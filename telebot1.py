from dotenv import load_dotenv
import os
import re
import ssl
import secrets
import sqlite3
import logging
import warnings
from datetime import datetime

from pydantic_ai import Agent
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

# -----------------------------
# 🔧 SSL Fix (optional, for restrictive networks)
# -----------------------------
def setup_simple_ssl_fix():
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except ImportError:
        pass
    print("✅ SSL verification disabled")

setup_simple_ssl_fix()
load_dotenv()
# -----------------------------
# 🔧 Configuration
# -----------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DB_FILE = os.getenv("DB_FILE", "raffle.db")
BOT_NAME_FA = os.getenv("BOT_NAME_FA", "باشگاه مشتریان")

FUNDS = {
    "SAM": "صندوق سام",
    "ROUIN": "صندوق رویین",
    "GHALAK": "صندوق قلک",
    "SOURNAFOOD": "صندوق سورنافود",
}

# OpenAI agent
agent: Agent | None = None
if OPENAI_API_KEY:
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

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
    digits_map = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
    t = text.translate(digits_map)
    t = re.sub(r"[\s\-()]+", "", t)
    return t if re.fullmatch(r"\+?\d{9,15}", t) else None

async def send_typing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    except Exception:
        pass

# -----------------------------
# 🤖 Handlers: Raffle
# -----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_typing(update, context)
    kb = InlineKeyboardMarkup(
        [[InlineKeyboardButton("🎲 شرکت در قرعه‌کشی", callback_data="start_draw")]]
    )
    text = (
        f"به ربات {BOT_NAME_FA} خوش اومدی!\n\n"
        "🔹 می‌تونی در قرعه‌کشی شرکت کنی و جایزه بگیری.\n"
        "🔹 یا هر سوالی داشتی با من چت کن (من از هوش مصنوعی قدرت می‌گیرم 🤖)."
    )
    await update.message.reply_text(text, reply_markup=kb)


async def start_draw_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    rows = [
        [InlineKeyboardButton("سام", callback_data="fund:SAM"), InlineKeyboardButton("رویین", callback_data="fund:ROUIN")],
        [InlineKeyboardButton("قلک", callback_data="fund:GHALAK"), InlineKeyboardButton("سورنافود", callback_data="fund:SOURNAFOOD")],
        [InlineKeyboardButton("🔁 فرقی نداره (تصادفی)", callback_data="fund:RANDOM")],
    ]

    await query.edit_message_text(
        "یکی از صندوق‌ها رو انتخاب کن:",
        reply_markup=InlineKeyboardMarkup(rows),
    )


async def fund_selected_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    _, code = query.data.split(":", 1)
    fund_code = secrets.choice(list(FUNDS.keys())) if code == "RANDOM" else code
    fund_name = FUNDS.get(fund_code, "صندوق")

    units = secrets.randbelow(100) + 1
    user = query.from_user
    draw_id = insert_draw(user.id, user.username, user.full_name, fund_name, units)

    phone_btn = KeyboardButton("ارسال شماره موبایل 📱", request_contact=True)
    cancel_btn = KeyboardButton("انصراف")
    reply_kb = ReplyKeyboardMarkup([[phone_btn], [cancel_btn]], resize_keyboard=True, one_time_keyboard=True)

    msg = (
        f"🎉 تبریک!\n\n"
        f"شما <b>{units}</b> واحد از <b>{fund_name}</b> برنده شدی! 🏆\n\n"
        "برای تحویل جایزه، لطفاً شماره موبایل‌ت رو ارسال کن."
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
    draw_id = set_phone_for_latest_pending(update.effective_user.id, contact.phone_number)

    if draw_id:
        await update.message.reply_text(
            f"✅ شماره شما ثبت شد.\nشناسه قرعه‌کشی: {draw_id}",
            reply_markup=ReplyKeyboardRemove(),
        )
    else:
        await update.message.reply_text(
            "هیچ قرعه‌کشی در انتظار شماره برای شما پیدا نشد.",
            reply_markup=ReplyKeyboardRemove(),
        )


async def phone_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if text == "انصراف":
        await update.message.reply_text("لغو شد.", reply_markup=ReplyKeyboardRemove())
        return

    phone = sanitize_phone(text)
    if not phone:
        await update.message.reply_text("❌ فرمت شماره موبایل معتبر نیست.")
        return

    draw_id = set_phone_for_latest_pending(update.effective_user.id, phone)
    if draw_id:
        await update.message.reply_text(
            f"✅ شماره شما ثبت شد.\nشناسه قرعه‌کشی: {draw_id}",
            reply_markup=ReplyKeyboardRemove(),
        )
    else:
        await update.message.reply_text("هیچ قرعه‌کشی در انتظار شماره برای شما پیدا نشد.")

# -----------------------------
# 🤖 Handlers: AI Chat
# -----------------------------
async def handle_ai_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global agent
    if not agent:
        agent = Agent('openai:gpt-4o')

    user_message = update.message.text
    await update.message.chat.send_action(action="typing")
    try:
        response = await agent.run(user_message)
        await update.message.reply_text(str(response.data))
    except Exception as e:
        await update.message.reply_text(f"❌ خطا در پردازش پیام: {e}")

# -----------------------------
# 🚀 Bootstrap
# -----------------------------
def main():
    logging.basicConfig(level=logging.INFO)

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("❌ TELEGRAM_BOT_TOKEN is missing")

    init_db()
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Raffle
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(start_draw_cb, pattern=r"^start_draw$"))
    app.add_handler(CallbackQueryHandler(fund_selected_cb, pattern=r"^fund:(.+)$"))
    app.add_handler(MessageHandler(filters.CONTACT, contact_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, phone_text_handler))

    # AI chat fallback (only when not matched above)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_ai_message))

    logging.info("🤖 Bot is running…")
    app.run_polling()

if __name__ == "__main__":
    main()
