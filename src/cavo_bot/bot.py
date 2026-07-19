from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    Update,
)
from telegram.constants import ChatAction
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .catalog import representative_image
from .config import Settings
from .inventory import CavoSheetClient, InventoryItem, format_inventory_result
from .preview import candidate_collage
from .recognizer import MatchCandidate, ProductRecognizer, ResNet18HybridEmbedder

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PendingMatch:
    image_bytes: bytes
    candidate_ids: tuple[str, ...]
    created_at: float


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
        self.refresh_task: asyncio.Task[None] | None = None

    async def post_init(self, _: Application) -> None:
        await self.inventory.refresh()
        self.refresh_task = asyncio.create_task(
            self.inventory.refresh_forever(self.settings.sheet_refresh_seconds),
            name="cavo-sheet-refresh",
        )

    async def post_shutdown(self, _: Application) -> None:
        if self.refresh_task:
            self.refresh_task.cancel()
            await asyncio.gather(self.refresh_task, return_exceptions=True)

    async def start(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_message:
            return
        await update.effective_message.reply_text(
            "👟 **مرحباً بك في CAVO Vision Bot**\n\n"
            "ابعت صورة منتج واحد واضحة، وهبعت لك صورة الكتالوج المطابقة مع المقاسات الحالية.\n\n"
            "الأوامر المتاحة:\n"
            "🧹 `/clean` - إلغاء الاختيار الحالي وبدء بحث جديد\n"
            "📊 `/status` - حالة الكتالوج والمخزون\n"
            "👨‍💻 `/dev` - معلومات المطور",
            parse_mode="Markdown",
        )

    async def status(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_message:
            return
        age = self.inventory.snapshot_age_seconds
        age_text = "غير متاح" if age is None else f"{age:.0f} ثانية"
        await update.effective_message.reply_text(
            "🟢 **حالة CAVO Vision Bot**\n\n"
            "🧠 المطابقة: شكل + لون بترتيب مستقل\n"
            f"📦 المنتجات المفعلة: {self.inventory.item_count}\n"
            f"🕒 عمر نسخة المخزون: {age_text}\n"
            "👨‍💻 التطوير والدمج: **MEZO**",
            parse_mode="Markdown",
        )

    async def clean(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_message or not update.effective_user:
            return
        removed = self.pending.pop(update.effective_user.id, None)
        text = (
            "🧹 تم إلغاء الاختيار السابق. ابعت صورة جديدة."
            if removed
            else "✨ مفيش اختيار معلق. ابعت صورة المنتج."
        )
        await update.effective_message.reply_text(text)

    async def developer(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_message:
            return
        await update.effective_message.reply_text(
            "💻 **CAVO Vision AI Assistant**\n\n"
            "⚡ **Developed & Engineered by: MEZO**\n"
            "🚀 ResNet18 Hybrid Recognition\n"
            "✨ *DeadZone By MEZO*",
            parse_mode="Markdown",
        )

    def _is_pending_fresh(self, pending: PendingMatch) -> bool:
        return time.time() - pending.created_at <= self.settings.pending_ttl_seconds

    @staticmethod
    def _caption(item: InventoryItem, candidate: MatchCandidate | None = None) -> str:
        caption = format_inventory_result(item)
        if candidate is not None:
            caption += f"\n\n🎯 نسبة المطابقة: {candidate.score * 100:.1f}%"
        return caption

    async def _reply_product(
        self,
        message,
        item: InventoryItem,
        candidate: MatchCandidate | None = None,
    ) -> None:
        image_path = representative_image(self.settings.catalog_dir, item.product_id)
        caption = self._caption(item, candidate)
        if image_path is None:
            await message.reply_text(
                caption + "\n\n⚠️ صورة الكتالوج الأصلية غير موجودة على السيرفر."
            )
            return
        with image_path.open("rb") as image_file:
            await message.reply_photo(photo=image_file, caption=caption)

    async def photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        user = update.effective_user
        if not message or not user or not message.photo:
            return

        self.pending.pop(user.id, None)
        await context.bot.send_chat_action(message.chat_id, ChatAction.TYPING)
        progress = await message.reply_text("🔍 جاري مقارنة الشكل واللون مع كتالوج CAVO...")

        telegram_file = await message.photo[-1].get_file()
        payload = bytes(await telegram_file.download_as_bytearray())
        result = await asyncio.to_thread(self.recognizer.match_bytes, payload)

        enabled_candidates = tuple(
            candidate
            for candidate in result.candidates
            if self.inventory.get(candidate.product_id) is not None
        )
        if not enabled_candidates:
            await progress.edit_text(
                "❌ لم أجد منتجًا مفعلاً مطابقًا. ابعت صورة أوضح بإضاءة طبيعية ومن غير فلاتر."
            )
            return

        best = enabled_candidates[0]
        enabled_confident = (
            result.confident
            and best.product_id == result.candidates[0].product_id
        )
        if enabled_confident:
            item = self.inventory.get(best.product_id)
            if item is None:  # Defensive; filtered above.
                await progress.edit_text("⚠️ بيانات المنتج غير متاحة حاليًا.")
                return
            await progress.delete()
            await self._reply_product(message, item, best)
            LOGGER.info(
                "Confident match user=%s product=%s score=%.4f shape=%.4f color=%.4f",
                user.id,
                best.product_id,
                best.score,
                best.shape_score,
                best.color_score,
            )
            return

        candidates = enabled_candidates[: self.settings.top_k]
        self.pending[user.id] = PendingMatch(
            image_bytes=payload,
            candidate_ids=tuple(candidate.product_id for candidate in candidates),
            created_at=time.time(),
        )
        collage = await asyncio.to_thread(candidate_collage, candidates)
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        f"{candidate.product_id} — {candidate.score * 100:.1f}%",
                        callback_data=f"pick:{candidate.product_id}",
                    )
                ]
                for candidate in candidates
            ]
        )
        await progress.delete()
        await message.reply_photo(
            collage,
            caption=(
                "⚠️ النتيجة غير مؤكدة، لذلك لن أخمّن. اختر نفس صورة المنتج بالضبط.\n"
                "الاختيار ينتهي تلقائيًا بعد 10 دقائق."
            ),
            reply_markup=keyboard,
        )

    async def pick(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        user = update.effective_user
        if not query or not user or not query.message:
            return

        product_id = (query.data or "").partition(":")[2].upper()
        pending = self.pending.get(user.id)
        if pending is None or not self._is_pending_fresh(pending):
            self.pending.pop(user.id, None)
            await query.answer("الاختيار انتهى. ابعت الصورة من جديد.", show_alert=True)
            return
        if product_id not in pending.candidate_ids:
            await query.answer("اختيار غير صالح.", show_alert=True)
            LOGGER.warning("Rejected forged pick user=%s product=%s", user.id, product_id)
            return

        item = self.inventory.get(product_id)
        if item is None:
            await query.answer("المنتج غير متاح حاليًا.", show_alert=True)
            return

        await query.answer()
        self.pending.pop(user.id, None)

        if user.id in self.settings.admin_user_ids:
            await asyncio.to_thread(
                self.recognizer.learn_reference,
                self.settings.catalog_dir,
                self.settings.index_path,
                product_id,
                pending.image_bytes,
            )
            LOGGER.info("Saved admin-confirmed reference product=%s", product_id)

        image_path = representative_image(self.settings.catalog_dir, product_id)
        caption = self._caption(item)
        if image_path is None:
            await query.edit_message_caption(
                caption=caption + "\n\n⚠️ صورة الكتالوج الأصلية غير موجودة على السيرفر."
            )
            return

        try:
            with image_path.open("rb") as image_file:
                await query.edit_message_media(
                    media=InputMediaPhoto(media=image_file, caption=caption),
                    reply_markup=None,
                )
        except BadRequest:
            LOGGER.exception("Could not replace candidate collage with product image")
            await self._reply_product(query.message, item)

    async def error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        LOGGER.exception("Telegram update failed", exc_info=context.error)


def build_application(settings: Settings) -> Application:
    service = CavoBot(settings)
    application = (
        ApplicationBuilder()
        .token(settings.telegram_bot_token)
        .post_init(service.post_init)
        .post_shutdown(service.post_shutdown)
        .build()
    )
    application.add_handler(CommandHandler("start", service.start))
    application.add_handler(CommandHandler("status", service.status))
    application.add_handler(CommandHandler(["clean", "reset"], service.clean))
    application.add_handler(CommandHandler(["dev", "about", "mezo"], service.developer))
    application.add_handler(MessageHandler(filters.PHOTO, service.photo))
    application.add_handler(CallbackQueryHandler(service.pick, pattern=r"^pick:CAVO-\d{4}$"))
    application.add_error_handler(service.error)
    return application


def main() -> None:
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings.catalog_dir.mkdir(parents=True, exist_ok=True)
    settings.learning_dir.mkdir(parents=True, exist_ok=True)
    build_application(settings).run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=False,
    )


if __name__ == "__main__":
    main()
