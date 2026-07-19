from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import Settings
from .inventory import CavoSheetClient, InventoryItem, size_sort_key
from .preview import candidate_collage
from .quality import ImageQuality, inspect_image
from .recognizer import (
    ProductRecognizer,
    ResNet18HybridEmbedder,
    confidence_gate,
)

LOGGER = logging.getLogger(__name__)

VERSION = "3.0.0"
SYSTEM_IDENTITY = "Cavo & Diva Vision V3"
PRODUCT_CODE_RE = re.compile(r"(?i)\bCAVO[- ]?(\d{1,4})\b")
SIZE_RE = re.compile(r"(?<!\d)(\d{2}(?:[.,]\d)?)(?!\d)")
KEY_LAST_PRODUCT = "last_product_id"

QUALITY_MESSAGES = {
    "empty": "الصورة فارغة. ابعت صورة المنتج من جديد.",
    "too_large": "حجم الصورة كبير جدًا. ابعت نسخة مضغوطة أو صورة من كاميرا تليجرام.",
    "invalid": "الملف مش صورة صالحة. ابعت JPG أو PNG واضحة.",
    "too_small": "دقة الصورة قليلة. قرّب المنتج وابعت صورة أوضح.",
    "too_dark": "الصورة مظلمة جدًا. صوّر المنتج في إضاءة أفضل.",
    "too_bright": "الإضاءة قوية وبتخفي لون المنتج. ابعت صورة بإضاءة طبيعية.",
    "too_blurry": "الصورة مهزوزة أو التفاصيل غير واضحة. ثبّت الكاميرا وجرّب تاني.",
}


@dataclass(slots=True)
class PendingMatch:
    image_bytes: bytes
    candidates: tuple[str, ...]
    created_at: float


def _quantity_label(quantity: int) -> str:
    if quantity == 1:
        return "قطعة واحدة"
    if quantity == 2:
        return "قطعتين"
    return f"{quantity} قطع"


def _format_stock_dashboard(item: InventoryItem, elapsed: float = 0.0) -> str:
    lines = []
    for size in sorted(item.sizes, key=size_sort_key):
        quantity = item.sizes[size]
        if quantity <= 0:
            continue
        icon = "⚠️" if quantity <= 2 else "🔹"
        lines.append(f"{icon} مقاس {size} ← ({_quantity_label(quantity)})")

    if not lines:
        body = "🔴 غير متوفر في المخزن حاليًا"
    else:
        body = "\n".join(lines) + f"\n\n📦 إجمالي المخزون: {item.computed_quantity} قطعة"

    warning = ""
    if not item.is_consistent or item.validation != "OK":
        warning = "\n⚠️ بيانات الكمية تحتاج مراجعة من الإدارة"
    return (
        "👟 Cavo & Diva | فحص المخزن 👟\n"
        f"🆔 الموديل: {item.product_id}\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"{body}{warning}\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"⏱️ الاستجابة: {elapsed:.2f} ثانية"
    )


def _extract_product_code(text: str) -> str | None:
    match = PRODUCT_CODE_RE.search(text)
    return f"CAVO-{match.group(1).zfill(4)}" if match else None


def _deterministic_inventory_answer(item: InventoryItem, text: str) -> str | None:
    normalized = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    size_match = SIZE_RE.search(normalized)
    if size_match:
        size = size_match.group(1).replace(",", ".")
        if size.endswith(".0"):
            size = size[:-2]
        quantity = item.sizes.get(size, 0)
        if quantity:
            return f"✅ مقاس {size} متوفر: {_quantity_label(quantity)} في {item.product_id}."
        return f"🔴 مقاس {size} غير متوفر حاليًا في {item.product_id}."
    if any(word in normalized for word in ("إجمالي", "اجمالي", "الكمية", "كام قطعة")):
        return f"📦 إجمالي {item.product_id}: {item.computed_quantity} قطعة."
    return None


class CavoBot:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.inventory = CavoSheetClient(settings.sheet_id, settings.sheet_name)
        self.recognizer = ProductRecognizer.load(
            settings.index_path,
            embedder=ResNet18HybridEmbedder(),
            min_score=settings.min_match_score,
            min_margin=settings.min_match_margin,
            top_k=settings.top_k,
            catalog_dir=settings.catalog_dir,
        )
        self.pending: dict[int, PendingMatch] = {}
        self.user_locks: dict[int, asyncio.Lock] = {}
        self.inference_slots = asyncio.Semaphore(settings.inference_concurrency)
        self.refresh_task: asyncio.Task[None] | None = None
        self.gemini_client = self._create_gemini_client()

    def _create_gemini_client(self) -> object | None:
        if not self.settings.gemini_api_key:
            return None
        try:
            from google import genai

            return genai.Client(api_key=self.settings.gemini_api_key)
        except Exception:
            LOGGER.exception("Gemini client initialization failed; deterministic mode active")
            return None

    def _user_lock(self, user_id: int) -> asyncio.Lock:
        return self.user_locks.setdefault(user_id, asyncio.Lock())

    def _has_access(self, update: Update) -> bool:
        user = update.effective_user
        return bool(
            user
            and (
                self.settings.allow_public
                or user.id in self.settings.allowed_user_ids
                or user.id in self.settings.admin_user_ids
            )
        )

    async def _require_access(self, update: Update) -> bool:
        if self._has_access(update):
            return True
        if update.callback_query:
            await update.callback_query.answer("غير مصرح لك باستخدام النظام", show_alert=True)
        elif update.effective_message:
            await update.effective_message.reply_text("⛔ هذا النظام داخلي وغير متاح لهذا الحساب.")
        return False

    def _prune_pending(self) -> None:
        cutoff = time.monotonic() - self.settings.pending_ttl_seconds
        expired = [user_id for user_id, item in self.pending.items() if item.created_at < cutoff]
        for user_id in expired:
            self.pending.pop(user_id, None)

    async def post_init(self, _: Application) -> None:
        try:
            await self.inventory.refresh()
        except Exception:
            LOGGER.exception("Initial inventory refresh failed; retry loop will continue")
        self.refresh_task = asyncio.create_task(
            self.inventory.refresh_forever(
                self.settings.sheet_refresh_seconds,
                self.settings.sheet_max_backoff_seconds,
            ),
            name="cavo-sheet-refresh",
        )
        LOGGER.info(
            "CAVO Vision V3 ready: %d products, %d references",
            self.recognizer.product_count,
            self.recognizer.index_size,
        )

    async def post_shutdown(self, _: Application) -> None:
        if self.refresh_task:
            self.refresh_task.cancel()
            await asyncio.gather(self.refresh_task, return_exceptions=True)

    async def cmd_start(
        self,
        update: Update,
        _: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not await self._require_access(update) or not update.effective_message:
            return
        await update.effective_message.reply_text(
            f"👟 {SYSTEM_IDENTITY}\n\n"
            "📸 ابعت صورة واضحة لمنتج واحد، وهحلل الشكل واللون وأعرض المخزون.\n"
            "🔎 أو اكتب كود الموديل مثل CAVO-0012.\n"
            "💬 بعد النتيجة اسأل عن أي مقاس.\n\n"
            "/status — حالة النظام\n"
            "/clean — بدء جلسة جديدة\n"
            f"/dev — الإصدار {VERSION}"
        )

    async def cmd_status(
        self,
        update: Update,
        _: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not await self._require_access(update) or not update.effective_message:
            return
        age = self.inventory.snapshot_age_seconds
        age_text = "غير متاح" if age is None else f"{age:.0f} ثانية"
        sheet_state = "🟢 سليم" if self.inventory.ready else "🔴 يحتاج انتباه"
        ai_state = "🟢 متاح" if self.gemini_client else "⚪ وضع حتمي بدون AI"
        error = (
            f"\n⚠️ آخر خطأ: {self.inventory.last_error[:300]}" if self.inventory.last_error else ""
        )
        await update.effective_message.reply_text(
            f"⚙️ {SYSTEM_IDENTITY}\n\n"
            f"📊 المخزون: {sheet_state} ({self.inventory.item_count} منتج)\n"
            f"🕒 عمر النسخة: {age_text}\n"
            f"🧠 الفهرس: 🟢 {self.recognizer.index_size} صورة / "
            f"{self.recognizer.product_count} منتج\n"
            f"💬 المساعد: {ai_state}\n"
            f"📌 الإصدار: {VERSION}{error}"
        )

    async def cmd_clean(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not await self._require_access(update) or not update.effective_message:
            return
        if update.effective_user:
            self.pending.pop(update.effective_user.id, None)
        context.user_data.clear()
        await update.effective_message.reply_text("🧹 تم تنظيف الجلسة. ابعت صورة جديدة.")

    async def cmd_dev(
        self,
        update: Update,
        _: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not await self._require_access(update) or not update.effective_message:
            return
        await update.effective_message.reply_text(
            "💻 CAVO Vision V3\n"
            "محرك هجين للشكل واللون مع رفض النتائج غير المؤكدة.\n"
            "👤 Developer: Mohammed Saeed (MEZO)\n"
            f"📌 Version: {VERSION}"
        )

    async def _gemini_reply(self, item: InventoryItem, message: str) -> str | None:
        if self.gemini_client is None:
            return None
        sizes = ", ".join(
            f"{size}×{item.sizes[size]}" for size in sorted(item.sizes, key=size_sort_key)
        )
        prompt = (
            "أجب بالمصرية في سطرين بحد أقصى. استخدم البيانات التالية فقط ولا تضف سعرًا "
            "أو لونًا أو مقاسًا غير موجود. إذا لم تكف البيانات فاطلب من المستخدم سؤالًا أوضح.\n"
            f"المنتج: {item.product_id}\nالمقاسات: {sizes}\n"
            f"الإجمالي: {item.computed_quantity}\nسؤال المستخدم: {message}"
        )

        def generate() -> str:
            response = self.gemini_client.models.generate_content(
                model=self.settings.gemini_model,
                contents=prompt,
            )
            return str(response.text or "").strip()

        try:
            reply = await asyncio.wait_for(
                asyncio.to_thread(generate),
                timeout=self.settings.gemini_timeout_seconds,
            )
            return reply[:3000] or None
        except Exception:
            LOGGER.exception("Gemini reply failed")
            return None

    async def text_handler(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not await self._require_access(update):
            return
        message = update.effective_message
        user = update.effective_user
        if not message or not user or not message.text:
            return
        async with self._user_lock(user.id):
            started = time.perf_counter()
            text = message.text.strip()
            await context.bot.send_chat_action(message.chat_id, ChatAction.TYPING)
            product_id = _extract_product_code(text)
            if product_id:
                item = self.inventory.get(product_id)
                if item is None:
                    await message.reply_text(f"🔴 {product_id} غير متوفر أو غير مفعّل حاليًا.")
                    return
                context.user_data[KEY_LAST_PRODUCT] = product_id
                await message.reply_text(
                    _format_stock_dashboard(item, time.perf_counter() - started)
                )
                return

            last_product = context.user_data.get(KEY_LAST_PRODUCT)
            item = self.inventory.get(str(last_product)) if last_product else None
            if item is None:
                await message.reply_text(
                    "📸 ابعت صورة المنتج أو اكتب كوده الأول، وبعدها اسأل عن المقاس."
                )
                return
            reply = _deterministic_inventory_answer(item, text)
            if reply is None:
                reply = await self._gemini_reply(item, text)
            await message.reply_text(reply or _format_stock_dashboard(item))

    async def photo_handler(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not await self._require_access(update):
            return
        message = update.effective_message
        user = update.effective_user
        if not message or not user or not message.photo:
            return
        async with self._user_lock(user.id):
            started = time.perf_counter()
            await context.bot.send_chat_action(message.chat_id, ChatAction.TYPING)
            progress = await message.reply_text("🔍 جاري فحص جودة الصورة وتحليل المنتج...")
            try:
                telegram_file = await message.photo[-1].get_file()
                payload = bytes(await telegram_file.download_as_bytearray())
                quality: ImageQuality = await asyncio.to_thread(
                    inspect_image,
                    payload,
                    max_bytes=self.settings.max_image_bytes,
                )
                if not quality.accepted:
                    await progress.edit_text(
                        f"📸 {QUALITY_MESSAGES.get(quality.reason, 'ابعت صورة أوضح.')}"
                    )
                    return

                async with self.inference_slots:
                    result = await asyncio.to_thread(self.recognizer.match_bytes, payload)

                eligible = tuple(
                    candidate
                    for candidate in result.candidates
                    if self.inventory.get(candidate.product_id) is not None
                )
                confident, _, _ = confidence_gate(
                    [candidate.score for candidate in eligible],
                    self.settings.min_match_score,
                    self.settings.min_match_margin,
                )
                if not eligible:
                    await progress.edit_text(
                        "❌ لم أجد منتجًا مفعّلًا مطابقًا. ابعت صورة أوضح أو اكتب الكود."
                    )
                    return
                if confident:
                    candidate = eligible[0]
                    item = self.inventory.get(candidate.product_id)
                    if item is None:
                        await progress.edit_text("🔴 المنتج غير متوفر أو غير مفعّل حاليًا.")
                        return
                    context.user_data[KEY_LAST_PRODUCT] = item.product_id
                    await progress.edit_text(
                        _format_stock_dashboard(item, time.perf_counter() - started)
                    )
                    return

                self._prune_pending()
                self.pending[user.id] = PendingMatch(
                    image_bytes=payload,
                    candidates=tuple(candidate.product_id for candidate in eligible),
                    created_at=time.monotonic(),
                )
                collage = await asyncio.to_thread(candidate_collage, eligible)
                keyboard = InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                f"{candidate.product_id} — {candidate.score:.1%}",
                                callback_data=f"pick:{candidate.product_id}",
                            )
                        ]
                        for candidate in eligible
                    ]
                )
                await progress.delete()
                await message.reply_photo(
                    collage,
                    caption="⚠️ النتائج متقاربة. اختر الصورة المطابقة بدل التخمين.",
                    reply_markup=keyboard,
                )
            except Exception:
                LOGGER.exception("Photo processing failed")
                await progress.edit_text("⚠️ حصل خطأ أثناء تحليل الصورة. جرّب تاني بصورة واضحة.")

    async def pick_handler(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not await self._require_access(update):
            return
        query = update.callback_query
        user = update.effective_user
        if not query or not user:
            return
        await query.answer()
        async with self._user_lock(user.id):
            self._prune_pending()
            pending = self.pending.get(user.id)
            product_id = (query.data or "").partition(":")[2].upper()
            if pending is None or product_id not in pending.candidates:
                await query.edit_message_caption(
                    caption="⌛ انتهت جلسة الاختيار. ابعت الصورة من جديد."
                )
                return
            self.pending.pop(user.id, None)
            item = self.inventory.get(product_id)
            if item is None:
                await query.edit_message_caption(caption="🔴 المنتج غير متوفر أو غير مفعّل حاليًا.")
                return
            learned = False
            if user.id in self.settings.admin_user_ids:
                try:
                    async with self.inference_slots:
                        await asyncio.to_thread(
                            self.recognizer.learn_reference,
                            self.settings.index_path,
                            product_id,
                            pending.image_bytes,
                        )
                    learned = True
                except Exception:
                    LOGGER.exception("Failed to save confirmed reference for %s", product_id)
            context.user_data[KEY_LAST_PRODUCT] = product_id
            suffix = "\n\n🧠 تم حفظ زاوية جديدة بعد تأكيد الأدمن." if learned else ""
            await query.edit_message_caption(caption=_format_stock_dashboard(item) + suffix)

    async def error_handler(
        self,
        update: object,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        LOGGER.exception("Telegram update failed", exc_info=context.error)
        if isinstance(update, Update) and update.effective_message:
            try:
                await update.effective_message.reply_text("⚠️ حدث خطأ مؤقت. جرّب مرة ثانية.")
            except Exception:
                LOGGER.debug("Could not send user-facing error", exc_info=True)


def build_application(settings: Settings) -> Application:
    service = CavoBot(settings)
    application = (
        ApplicationBuilder()
        .token(settings.telegram_bot_token)
        .concurrent_updates(settings.concurrent_updates)
        .post_init(service.post_init)
        .post_shutdown(service.post_shutdown)
        .build()
    )
    application.add_handler(CommandHandler(["start", "help"], service.cmd_start))
    application.add_handler(CommandHandler("status", service.cmd_status))
    application.add_handler(CommandHandler(["clean", "reset"], service.cmd_clean))
    application.add_handler(CommandHandler(["dev", "about", "mezo"], service.cmd_dev))
    application.add_handler(
        CallbackQueryHandler(service.pick_handler, pattern=r"^pick:CAVO-\d{4}$")
    )
    application.add_handler(MessageHandler(filters.PHOTO, service.photo_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, service.text_handler))
    application.add_error_handler(service.error_handler)
    return application


def main() -> None:
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings.catalog_dir.mkdir(parents=True, exist_ok=True)
    settings.learning_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Starting %s version %s", SYSTEM_IDENTITY, VERSION)
    build_application(settings).run_polling(
        allowed_updates=["message", "callback_query"],
        drop_pending_updates=settings.drop_pending_updates,
    )


if __name__ == "__main__":
    main()
