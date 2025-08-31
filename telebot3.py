from dotenv import load_dotenv
import os
import re
import ssl
import secrets
import sqlite3
import logging
import warnings
from datetime import datetime

from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Literal
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
    ConversationHandler,
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

# Conversation states
WAITING_PHONE = range(1)

# OpenAI agent
agent: Agent | None = None
if OPENAI_API_KEY:
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# -----------------------------
# 🤖 AI Response Models
# -----------------------------
class FinancialResponse(BaseModel):
    """Structured response for financial questions"""
    response_type: Literal["financial", "non_financial"] = Field(
        description="Whether the question is about finance/investment or not"
    )
    message: str = Field(
        max_length=200,
        description="Short response message in Persian. If non-financial, use exact message: 'من دستیار هوش مصنوعی سبدگردان سورنا هستم و فقط در زمینه مالی و سرمایه گذاری می توانم به شما کمک کنم.'"
    )

def create_financial_agent():
    """Create the financial assistant agent with structured responses"""
    
    system_prompt = """You are Sourena, a specialized financial assistant for an Iranian investment company.

STRICT INSTRUCTIONS:
1. Analyze if the user's question is about finance, investments, funds, markets, or economics
2. If YES (financial): Give a short, helpful answer in Persian (max 2-3 sentences)
3. If NO (non-financial): Use EXACTLY this Persian message: "من دستیار هوش مصنوعی سبدگردان سورنا هستم و فقط در زمینه مالی و سرمایه گذاری می توانم به شما کمک کنم."

Available investment funds to discuss:
- صندوق سام (SAM Fund)
- صندوق رویین (ROUIN Fund) 
- صندوق قلک (GHALAK Fund)
- صندوق سورنافود (SOURNAFOOD Fund)

Always respond in Persian and keep answers brief and professional."""

    return Agent('openai:gpt-4o', result_type=FinancialResponse, system_prompt=system_prompt)

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
    
    # Add chat history table for AI conversations
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            message TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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

def save_chat_history(user_id: int, message: str, response: str):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO chat_history (user_id, message, response)
        VALUES (?, ?, ?)
        """,
        (user_id, message, response),
    )
    conn.commit()
    conn.close()

def has_pending_phone_request(user_id: int) -> bool:
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 1 FROM draws
        WHERE user_id = ? AND phone IS NULL AND status = 'awaiting_phone'
        LIMIT 1
        """,
        (user_id,),
    )
    result = cur.fetchone() is not None
    conn.close()
    return result

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
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🎲 شرکت در قرعه‌کشی", callback_data="start_draw")],
        [InlineKeyboardButton("💬 چت با ربات", callback_data="start_chat")]
    ])
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
        [InlineKeyboardButton("🔙 بازگشت", callback_data="back_to_main")]
    ]

    await query.edit_message_text(
        "یکی از صندوق‌ها رو انتخاب کن:",
        reply_markup=InlineKeyboardMarkup(rows),
    )

async def start_chat_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔙 بازگشت به منوی اصلی", callback_data="back_to_main")]
    ])
    
    await query.edit_message_text(
        "💬 حالت چت فعال شد!\n\n"
        "حالا می‌تونی هر سوالی بپرسی و من باهات صحبت کنم. "
        "فقط پیام‌ت رو بفرست! 🤖",
        reply_markup=kb
    )

async def back_to_main_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🎲 شرکت در قرعه‌کشی", callback_data="start_draw")],
        [InlineKeyboardButton("💬 چت با ربات", callback_data="start_chat")]
    ])
    text = (
        f"به ربات {BOT_NAME_FA} خوش اومدی!\n\n"
        "🔹 می‌تونی در قرعه‌کشی شرکت کنی و جایزه بگیری.\n"
        "🔹 یا هر سوالی داشتی با من چت کن (من از هوش مصنوعی قدرت می‌گیرم 🤖)."
    )
    await query.edit_message_text(text, reply_markup=kb)

async def fund_selected_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    _, code = query.data.split(":", 1)
    fund_code = secrets.choice(list(FUNDS.keys())) if code == "RANDOM" else code
    fund_name = FUNDS.get(fund_code, "صندوق")

    units = secrets.randbelow(100) + 1
    user = query.from_user
    draw_id = insert_draw(user.id, user.username, user.full_name, fund_name, units)

    # Store draw_id in user_data for the conversation
    context.user_data['pending_draw_id'] = draw_id
    
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
    
    # Start phone conversation
    return WAITING_PHONE

# -----------------------------
# 📞 Phone collection conversation handler
# -----------------------------
async def phone_received(update: Update, context: ContextTypes.DEFAULT_TYPE):
    contact = update.message.contact
    if contact:
        draw_id = set_phone_for_latest_pending(update.effective_user.id, contact.phone_number)
        if draw_id:
            await update.message.reply_text(
                f"✅ شماره شما ثبت شد. از طرف واحد ارتباط با مشتریان سبدگردان سورنا با شما تماس گرفته خواهد شد.\nشناسه قرعه‌کشی: {draw_id}",
                reply_markup=ReplyKeyboardRemove(),
            )
        else:
            await update.message.reply_text(
                "هیچ قرعه‌کشی در انتظار شماره برای شما پیدا نشد.",
                reply_markup=ReplyKeyboardRemove(),
            )
        return ConversationHandler.END

    # Handle text input for phone
    text = (update.message.text or "").strip()
    if text == "انصراف":
        await update.message.reply_text("لغو شد.", reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END

    phone = sanitize_phone(text)
    if not phone:
        await update.message.reply_text("❌ فرمت شماره موبایل معتبر نیست. دوباره تلاش کن:")
        return WAITING_PHONE

    draw_id = set_phone_for_latest_pending(update.effective_user.id, phone)
    if draw_id:
        await update.message.reply_text(
            f"✅ شماره شما ثبت شد. از طرف واحد ارتباط با مشتریان سبدگردان سورنا با شما تماس گرفته خواهد شد.\nشناسه قرعه‌کشی: {draw_id}",
            reply_markup=ReplyKeyboardRemove(),
        )
    else:
        await update.message.reply_text(
            "هیچ قرعه‌کشی در انتظار شماره برای شما پیدا نشد.",
            reply_markup=ReplyKeyboardRemove(),
        )
    return ConversationHandler.END

async def cancel_phone(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("عملیات لغو شد.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END

# -----------------------------
# 🤖 Handlers: AI Chat
# -----------------------------
async def handle_ai_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global agent

    if not OPENAI_API_KEY:
        await update.message.reply_text(
            "❌ متأسفانه سرویس چت فعلاً در دسترس نیست.\n"
            "کلید API تنظیم نشده است."
        )
        return

    user_message = update.message.text
    user_id = update.effective_user.id
    
    await send_typing(update, context)
    
    try:
        # Initialize financial agent if needed
        if not agent:
            logging.info("Initializing financial AI agent...")
            agent = create_financial_agent()
            logging.info("Financial AI agent initialized successfully")

        # Run the AI agent with the financial-specific prompt
        response = await agent.run(user_message)
        response_text = response.data.message
        
        logging.info(f"AI response received: {response_text[:100]}...")
        
        # Save chat history
        save_chat_history(user_id, user_message, response_text)
        
        # Add option to return to main menu
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 بازگشت به منوی اصلی", callback_data="back_to_main")]
        ])
        
        await update.message.reply_text(response_text, reply_markup=kb)
        
    except ImportError as e:
        logging.error(f"Import error - pydantic_ai not installed: {e}")
        await update.message.reply_text(
            "❌ ربات هوشمند فعلاً غیرفعال است.\n"
            "نیاز به نصب pydantic-ai دارد."
        )
    except Exception as e:
        logging.error(f"AI chat error: {str(e)}", exc_info=True)
        error_msg = str(e)
        
        # More specific error messages
        if "api key" in error_msg.lower():
            await update.message.reply_text(
                "❌ مشکل در کلید API. لطفاً تنظیمات را بررسی کنید."
            )
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            await update.message.reply_text(
                "❌ مشکل اتصال به شبکه. لطفاً دوباره تلاش کنید."
            )
        elif "rate limit" in error_msg.lower():
            await update.message.reply_text(
                "❌ محدودیت استفاده از API. لطفاً کمی صبر کنید."
            )
        else:
            await update.message.reply_text(
                f"❌ خطا در پردازش پیام: {error_msg}\n"
                "لطفاً دوباره تلاش کنید."
            )

# -----------------------------
# 📱 Commands
# -----------------------------
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        f"🤖 راهنمای {BOT_NAME_FA}\n\n"
        "دستورات موجود:\n"
        "/start - شروع و نمایش منوی اصلی\n"
        "/help - نمایش این راهنما\n"
        "/status - وضعیت قرعه‌کشی‌های شما\n\n"
        "امکانات:\n"
        "🎲 قرعه‌کشی - شرکت در قرعه و برنده شدن واحدهای صندوق\n"
        "💬 چت هوشمند - صحبت با ربات AI\n\n"
        "برای شروع /start را بزنید."
    )
    await update.message.reply_text(help_text)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    
    # Get user's draws
    cur.execute(
        """
        SELECT id, fund, units, status, created_at FROM draws
        WHERE user_id = ?
        ORDER BY datetime(created_at) DESC
        LIMIT 10
        """,
        (user_id,),
    )
    draws = cur.fetchall()
    
    # Get chat count
    cur.execute(
        "SELECT COUNT(*) FROM chat_history WHERE user_id = ?",
        (user_id,),
    )
    chat_count = cur.fetchone()[0]
    
    conn.close()
    
    if not draws and chat_count == 0:
        await update.message.reply_text("شما هنوز در هیچ فعالیتی شرکت نکرده‌اید.")
        return
    
    status_text = f"📊 وضعیت شما:\n\n"
    
    if draws:
        status_text += f"🎲 قرعه‌کشی‌ها ({len(draws)} مورد):\n"
        for draw in draws:
            draw_id, fund, units, status, created = draw
            status_fa = "✅ تکمیل شده" if status == "phone_received" else "⏳ در انتظار شماره"
            status_text += f"• شناسه {draw_id}: {units} واحد {fund} - {status_fa}\n"
        status_text += "\n"
    
    if chat_count > 0:
        status_text += f"💬 تعداد گفتگوها: {chat_count}\n"
    
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔙 بازگشت به منوی اصلی", callback_data="back_to_main")]
    ])
    
    await update.message.reply_text(status_text, reply_markup=kb)

# -----------------------------
# 🚀 Bootstrap
# -----------------------------
def main():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("❌ TELEGRAM_BOT_TOKEN is missing")

    init_db()
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Phone collection conversation handler
    phone_conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(fund_selected_cb, pattern=r"^fund:(.+)$")],
        states={
            WAITING_PHONE: [
                MessageHandler(filters.CONTACT, phone_received),
                MessageHandler(filters.TEXT & ~filters.COMMAND, phone_received),
            ],
        },
        fallbacks=[
            CommandHandler("cancel", cancel_phone),
            MessageHandler(filters.Regex("^انصراف$"), cancel_phone)
        ],
        per_user=True,
        per_chat=True,
    )

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    
    # Phone collection conversation handler (high priority)
    app.add_handler(phone_conv_handler)
    
    # Callback handlers
    app.add_handler(CallbackQueryHandler(start_draw_cb, pattern=r"^start_draw$"))
    app.add_handler(CallbackQueryHandler(start_chat_cb, pattern=r"^start_chat$"))
    app.add_handler(CallbackQueryHandler(back_to_main_cb, pattern=r"^back_to_main$"))

    # AI chat handler (lowest priority - catches remaining text messages)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_ai_message))

    logging.info("🤖 Bot is running with AI chat enabled...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
