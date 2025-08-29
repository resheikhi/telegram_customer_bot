from dotenv import load_dotenv
import os
import re
import sqlite3
import secrets
import logging
from datetime import datetime

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardRemove,
)
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
load_dotenv()
# -----------------------------
# 🔧 Configuration
# -----------------------------
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_FILE = os.getenv("DB_FILE", "raffle.db")
BOT_NAME_FA = os.getenv("BOT_NAME_FA", "باشگاه مشتریان")

# Funds mapping (callback code -> display name)
FUNDS = {
    "SAM": "صندوق سام",
    "ROUIN": "صندوق رویین",
    "GHALAK": "صندوق قلک",
    "SOURNAFOOD": "صندوق سورنافود",
}

# -----------------------------
# 🗂️ Database helpers
# -----------------------------

def init_db() -> None:
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS draws (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            username TEXT,
            full_name TEXT,
            fund TEXT NOT NULL,
            units INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            phone TEXT,
            status TEXT NOT NULL DEFAULT 'awaiting_phone'
        );
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_draws_user_created
        ON draws(user_id, created_at DESC);
        """
    )
    conn.commit()
    conn.close()


def insert_draw(user_id: int, username: str | None, full_name: str | None, fund: str, units: int) -> int:
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO draws (user_id, username, full_name, fund, units)
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, username, full_name, fund, units),
    )
    conn.commit()
    draw_id = cur.lastrowid
    conn.close()
    return draw_id


def set_phone_for_latest_pending(user_id: int, phone: str) -> int | None:
    """Attach phone to the most recent pending draw for this user. Returns draw_id or None."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id FROM draws
        WHERE user_id = ? AND phone IS NULL AND status = 'awaiting_phone'
        ORDER BY datetime(created_at) DESC
        LIMIT 1
        """,
        (user_id,),
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return None

    draw_id = int(row[0])
    cur.execute(
        "UPDATE draws SET phone = ?, status = 'phone_received' WHERE id = ?",
        (phone, draw_id),
    )
    conn.commit()
    conn.close()
    return draw_id


# -----------------------------
# 🧼 Utilities
# -----------------------------

def sanitize_phone(text: str) -> str | None:
    """Extract a phone-like string: 9-15 digits with optional leading '+'."""
    if not text:
        return None
    # Convert Persian/Arabic digits to English
    digits_map = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
    t = text.translate(digits_map)
    t = re.sub(r"[\s\-()]+", "", t)
    m = re.fullmatch(r"\+?\d{9,15}", t)
    return t if m else None


async def send_typing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    except Exception:
        pass


# -----------------------------
# 🤖 Bot Handlers
# -----------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_typing(update, context)
    kb = InlineKeyboardMarkup(
        [[InlineKeyboardButton("🎲 شرکت در قرعه‌کشی", callback_data="start_draw")]]
    )
    text = (
        f"به ربات {BOT_NAME_FA} خوش اومدی!\n\n"
        "با کلیک روی دکمه زیر در قرعه‌کشی شرکت کن و بین ۱ تا ۱۰۰ واحد از یکی از صندوق‌ها جایزه بگیر.\n"
        "(برای تحویل جایزه، شماره موبایلت ازت گرفته میشه و در دیتابیس ذخیره می‌کنیم.)"
    )
    await update.message.reply_text(text, reply_markup=kb)


async def start_draw_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    # Ask user to choose a fund
    rows = [
        [InlineKeyboardButton("سام", callback_data="fund:SAM"), InlineKeyboardButton("رویین", callback_data="fund:ROUIN")],
        [InlineKeyboardButton("قلک", callback_data="fund:GHAK"), InlineKeyboardButton("سورنافود", callback_data="fund:SOURNAFOOD")],
        [InlineKeyboardButton("🔁 فرقی نداره (تصادفی)", callback_data="fund:RANDOM")],
    ]
    # Fix key for GHALAK
    rows[1][0] = InlineKeyboardButton("قلک", callback_data="fund:GHALAK")

    await query.edit_message_text(
        "یکی از صندوق‌ها رو انتخاب کن:",
        reply_markup=InlineKeyboardMarkup(rows),
    )


async def fund_selected_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    _, code = query.data.split(":", 1)
    fund_code = code
    if code == "RANDOM":
        fund_code = secrets.choice(list(FUNDS.keys()))

    fund_name = FUNDS.get(fund_code, "صندوق")

    # Draw 1..100 uniformly
    units = secrets.randbelow(100) + 1

    user = query.from_user
    full_name = (user.full_name or "").strip() if hasattr(user, "full_name") else None
    draw_id = insert_draw(
        user_id=user.id,
        username=user.username,
        full_name=full_name,
        fund=fund_name,
        units=units,
    )

    # Ask for phone number
    phone_btn = KeyboardButton("ارسال شماره موبایل 📱", request_contact=True)
    cancel_btn = KeyboardButton("انصراف")
    reply_kb = ReplyKeyboardMarkup([[phone_btn], [cancel_btn]], resize_keyboard=True, one_time_keyboard=True)

    msg = (
        f"🎉 تبریک!\n\n"
        f"شما <b>{units}</b> واحد از <b>{fund_name}</b> برنده شدی! 🏆\n\n"
        "برای تحویل جایزه، لطفاً شماره موبایل‌ت رو ارسال کن یا دکمه زیر رو بزن."
        f"\nشناسه قرعه‌کشی: <code>{draw_id}</code>"
    )
    await query.edit_message_text(msg, parse_mode="HTML")
    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text="شماره موبایل رو بفرست:",
        reply_markup=reply_kb,
    )


async def contact_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    contact = update.message.contact
    user_id = update.effective_user.id
    phone = contact.phone_number

    draw_id = set_phone_for_latest_pending(user_id, phone)

    if draw_id is None:
        await update.message.reply_text(
            "هیچ قرعه‌کشی در انتظار شماره برای شما پیدا نشد. ابتدا در قرعه‌کشی شرکت کنید.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return

    await update.message.reply_text(
        f"✅ شماره شما ثبت شد.\nشناسه قرعه‌کشی: {draw_id}\nاز طرف تیم {BOT_NAME_FA} با شما تماس می‌گیریم.",
        reply_markup=ReplyKeyboardRemove(),
    )


async def phone_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if text == "انصراف":
        await update.message.reply_text("لغو شد.", reply_markup=ReplyKeyboardRemove())
        return

    phone = sanitize_phone(text)
    if not phone:
        await update.message.reply_text(
            "❌ فرمت شماره موبایل معتبر نیست. نمونه‌های قابل قبول: 09121234567 یا +989121234567",
        )
        return

    user_id = update.effective_user.id
    draw_id = set_phone_for_latest_pending(user_id, phone)

    if draw_id is None:
        await update.message.reply_text(
            "هیچ قرعه‌کشی در انتظار شماره برای شما پیدا نشد. ابتدا در قرعه‌کشی شرکت کنید.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return

    await update.message.reply_text(
        f"✅ شماره شما ثبت شد.\nشناسه قرعه‌کشی: {draw_id}\nاز طرف تیم {BOT_NAME_FA} با شما تماس می‌گیریم.",
        reply_markup=ReplyKeyboardRemove(),
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "دستورات:\n/start شروع\n/help راهنما\nبرای شرکت در قرعه‌کشی روی دکمه‌ها کلیک کنید."
    )


# -----------------------------
# 🚀 Bootstrap
# -----------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if not TOKEN:
        raise RuntimeError("Please set TELEGRAM_BOT_TOKEN environment variable")

    init_db()

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))

    app.add_handler(CallbackQueryHandler(start_draw_cb, pattern=r"^start_draw$"))
    app.add_handler(CallbackQueryHandler(fund_selected_cb, pattern=r"^fund:(.+)$"))

    app.add_handler(MessageHandler(filters.CONTACT, contact_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, phone_text_handler))

    logging.info("Bot is running…")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
