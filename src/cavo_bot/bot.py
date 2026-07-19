from __future__ import annotations

import asyncio
import logging
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
from .inventory import CavoSheetClient, format_inventory_result
from .preview import candidate_collage
from .recognizer import ProductRecognizer, ResNet18HybridEmbedder

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PendingMatch:
    image_bytes: bytes
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
            "👟 ابعت صورة منتج واحد من CAVO، وأنا هحدد اللون والموديل وأجيب المقاسات الحالية."
        )

    async def status(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_message:
            return
        age = self.inventory.snapshot_age_seconds
        age_text = "غير متاح" if age is None else f"{age:.0f} ثانية"
        await update.effective_message.reply_text(
            "🟢 النظام يعمل\n"
            f"📦 المنتجات المحملة: {self.inventory.item_count}\n"
            f"🕒 عمر آخر نسخة مخزون: {age_text}"
        )

    async def photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        user = update.effective_user
        if not message or not user or not message.photo:
            return
        await context.bot.send_chat_action(message.chat_id, ChatAction.TYPING)
        progress = await message.reply_text("🔍 جاري التعرّف على المنتج...")

        telegram_file = await message.photo[-1].get_file()
        payload = bytes(await telegram_file.download_as_bytearray())
        result = await asyncio.to_thread(self.recognizer.match_bytes, payload)
        if not result.candidates:
            await progress.edit_text("❌ لم أستطع العثور على منتج مطابق. أرسل صورة أوضح.")
            return

        if result.confident:
            candidate = result.candidates[0]
            item = self.inventory.get(candidate.product_id)
            if item is None:
                await progress.edit_text(
                    f"⚠️ تعرفت على {candidate.product_id} لكن بيانات المقاسات غير موجودة حاليًا."
                )
                return
            await progress.edit_text(format_inventory_result(item))
            return

        self.pending[user.id] = PendingMatch(image_bytes=payload, created_at=time.time())
        collage = await asyncio.to_thread(candidate_collage, result.candidates)
        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton(candidate.product_id, callback_data=f"pick:{candidate.product_id}")]
             for candidate in result.candidates]
        )
        await progress.delete()
        await message.reply_photo(
            collage,
            caption="⚠️ الموديلات متقاربة. اختر الصورة المطابقة بدل التخمين:",
            reply_markup=keyboard,
        )

    async def pick(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        user = update.effective_user
        if not query or not user:
            return
        await query.answer()
        product_id = (query.data or "").partition(":")[2].upper()
        item = self.inventory.get(product_id)
        if item is None:
            await query.edit_message_caption(caption=f"⚠️ لا توجد مقاسات حالية لـ {product_id}")
            return

        pending = self.pending.pop(user.id, None)
        if pending and user.id in self.settings.admin_user_ids:
            await asyncio.to_thread(
                self.recognizer.learn_reference,
                self.settings.catalog_dir,
                self.settings.index_path,
                product_id,
                pending.image_bytes,
            )
            LOGGER.info("Saved an admin-confirmed reference for %s", product_id)

        await query.edit_message_caption(caption=format_inventory_result(item))

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
