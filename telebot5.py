#This code have just investment chat with monthly reports.
#------------------------------------
from dotenv import load_dotenv
import os
import re
import ssl
import secrets
import sqlite3
import logging
import warnings
import glob
from datetime import datetime
from pathlib import Path

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
    "ghollak": "صندوق قلک",
    "SOURNAFOOD": "صندوق سورنافود",
}

# Fund name mapping for image search
FUND_SEARCH_KEYWORDS = {
    "SAM": ["sam", "saam", "سام"],
    "ROUIN": ["rouin", "royin", "رویین"],
    "ghollak": ["ghollak", "ghalagh", "قلک"],
    "SOURNAFOOD": ["sourenafood", "sourena", "سورنافود", "سورنا"],
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
- صندوق قلک (ghollak Fund)
- صندوق سورنافود (SOURNAFOOD Fund)

Always respond in Persian and keep answers brief and professional."""

    return Agent('openai:gpt-4o', result_type=FinancialResponse, system_prompt=system_prompt)

# -----------------------------
# 📊 Report Image Helper Functions
# -----------------------------
def find_fund_report_image(fund_code: str) -> str | None:
    """
    Find the report image file for a specific fund by searching for keywords in filenames.
    Returns the path to the image file if found, None otherwise.
    """
    current_dir = Path.cwd()
    keywords = FUND_SEARCH_KEYWORDS.get(fund_code, [fund_code.lower()])
    
    # Common image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']
    
    for ext in extensions:
        image_files = list(current_dir.glob(ext)) + list(current_dir.glob(ext.upper()))
        
        for image_file in image_files:
            filename_lower = image_file.name.lower()
            
            # Check if any keyword is in the filename
            for keyword in keywords:
                if keyword.lower() in filename_lower:
                    logging.info(f"Found report image for {fund_code}: {image_file}")
                    return str(image_file)
    
    logging.warning(f"No report image found for fund {fund_code}")
    return None

def get_available_fund_reports() -> dict:
    """
    Get a dictionary of funds that have report images available.
    Returns {fund_code: image_path} for funds with available images.
    """
    available_reports = {}
    
    for fund_code in FUNDS.keys():
        image_path = find_fund_report_image(fund_code)
        if image_path:
            available_reports[fund_code] = image_path
    
    return available_reports

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
    
    # Add report views tracking table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS report_views (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            fund_code TEXT NOT NULL,
            fund_name TEXT NOT NULL,
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

def save_report_view(user_id: int, fund_code: str, fund_name: str):
    """Save when a user views a monthly report"""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO report_views (user_id, fund_code, fund_name)
        VALUES (?, ?, ?)
        """,
        (user_id, fund_code, fund_name),
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
# Helper function to safely handle message editing
# -----------------------------
async def safe_edit_or_send_message(query, text, reply_markup=None, parse_mode="HTML"):
    """
    Safely edit a message or send a new one if editing fails
    """
    try:
        # Try to edit the message
        await query.edit_message_text(
            text=text,
            reply_markup=reply_markup,
            parse_mode=parse_mode
        )
    except Exception as e:
        logging.info(f"Could not edit message (likely photo message): {str(e)}")
        # If editing fails, delete the original message and send a new one
        try:
            await query.delete_message()
        except Exception:
            pass
        
        # Send new message
        await query.message.reply_text(
            text,
            reply_markup=reply_markup,
            parse_mode=parse_mode
        )

# -----------------------------
# 🤖 Handlers: Main Menu
# -----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_typing(update, context)
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🎲 شرکت در قرعه‌کشی", callback_data="start_draw")],
        [InlineKeyboardButton("📊 گزارش‌های ماهانه", callback_data="monthly_reports")],
        [InlineKeyboardButton("💬 گفتگو با ربات", callback_data="start_chat")]
    ])
    text = (
        f"به ربات {BOT_NAME_FA} خوش اومدی!\n\n"
        "🔹 می‌تونی در قرعه‌کشی شرکت کنی و جایزه بگیری.\n"
        "🔹 گزارش‌های ماهانه صندوق‌ها رو مشاهده کن.\n"
        "🔹 یا هر سوالی داشتی با من گفتگو کن (من از هوش مصنوعی قدرت می‌گیرم 🤖)."
    )
    await update.message.reply_text(text, reply_markup=kb)

async def back_to_main_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🎲 شرکت در قرعه‌کشی", callback_data="start_draw")],
        [InlineKeyboardButton("📊 گزارش‌های ماهانه", callback_data="monthly_reports")],
        [InlineKeyboardButton("💬 گفتگو با ربات", callback_data="start_chat")]
    ])
    text = (
        f"به ربات {BOT_NAME_FA} خوش اومدی!\n\n"
        "🔹 می‌تونی در قرعه‌کشی شرکت کنی و جایزه بگیری.\n"
        "🔹 گزارش‌های ماهانه صندوق‌ها رو مشاهده کن.\n"
        "🔹 یا هر سوالی داشتی با من گفتگو کن (من از هوش مصنوعی قدرت می‌گیرم 🤖)."
    )
    
    # Use safe edit function
    await safe_edit_or_send_message(query, text, kb)

# -----------------------------
# 📊 Handlers: Monthly Reports
# -----------------------------
async def monthly_reports_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    # Get available fund reports
    available_reports = get_available_fund_reports()
    
    if not available_reports:
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 بازگشت", callback_data="back_to_main")]
        ])
        text = (
            "❌ هیچ گزارش ماهانه‌ای در حال حاضر موجود نیست.\n"
            "لطفاً بعداً دوباره تلاش کنید."
        )
        await safe_edit_or_send_message(query, text, kb)
        return
    
    # Create buttons for available funds
    rows = []
    fund_buttons = []
    
    for fund_code in available_reports.keys():
        fund_name = FUNDS.get(fund_code, fund_code)
        # Remove "صندوق" prefix for button text
        button_text = fund_name.replace("صندوق ", "")
        fund_buttons.append(InlineKeyboardButton(button_text, callback_data=f"report:{fund_code}"))
        
        # Add row every 2 buttons
        if len(fund_buttons) == 2:
            rows.append(fund_buttons)
            fund_buttons = []
    
    # Add remaining buttons
    if fund_buttons:
        rows.append(fund_buttons)
    
    # Add back button
    rows.append([InlineKeyboardButton("🔙 بازگشت", callback_data="back_to_main")])
    
    kb = InlineKeyboardMarkup(rows)
    text = (
        "📊 گزارش ماهانه کدام صندوق را می‌خواهید مشاهده کنید؟\n\n"
        f"تعداد گزارش‌های موجود: {len(available_reports)}"
    )
    
    await safe_edit_or_send_message(query, text, kb)

async def show_fund_report_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    _, fund_code = query.data.split(":", 1)
    fund_name = FUNDS.get(fund_code, fund_code)
    
    # Find the report image
    image_path = find_fund_report_image(fund_code)
    
    if not image_path:
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 بازگشت", callback_data="monthly_reports")]
        ])
        text = (
            f"❌ گزارش ماهانه {fund_name} در حال حاضر موجود نیست.\n"
            "لطفاً بعداً دوباره تلاش کنید."
        )
        await safe_edit_or_send_message(query, text, kb)
        return
    
    try:
        # Save the report view to database
        save_report_view(query.from_user.id, fund_code, fund_name)
        
        # Create the keyboard with proper callback data
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 گزارش‌های دیگر", callback_data="monthly_reports")],
            [InlineKeyboardButton("🔙 منوی اصلی", callback_data="back_to_main")]
        ])
        
        with open(image_path, 'rb') as photo:
            caption = f"📊 گزارش ماهانه {fund_name}\n\n📅 تاریخ مشاهده: {datetime.now().strftime('%Y/%m/%d - %H:%M')}"
            
            # Send photo with keyboard
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=photo,
                caption=caption,
                reply_markup=kb
            )
        
        # Delete the original message AFTER sending the photo
        try:
            await query.delete_message()
        except Exception as delete_error:
            logging.info(f"Could not delete original message: {str(delete_error)}")
            
        logging.info(f"Report sent for fund {fund_code} to user {query.from_user.id}")
        
    except FileNotFoundError:
        logging.error(f"Report image file not found: {image_path}")
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 بازگشت", callback_data="monthly_reports")]
        ])
        error_text = f"❌ فایل گزارش {fund_name} پیدا نشد.\nلطفاً به مدیر سیستم اطلاع دهید."
        await safe_edit_or_send_message(query, error_text, kb)
    except Exception as e:
        logging.error(f"Error sending report image: {str(e)}")
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 بازگشت", callback_data="monthly_reports")]
        ])
        error_text = f"❌ خطا در ارسال گزارش {fund_name}.\nلطفاً دوباره تلاش کنید."
        await safe_edit_or_send_message(query, error_text, kb)

# -----------------------------
# 🤖 Handlers: Raffle
# -----------------------------
async def start_draw_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    rows = [
        [InlineKeyboardButton("سام", callback_data="fund:SAM"), InlineKeyboardButton("رویین", callback_data="fund:ROUIN")],
        [InlineKeyboardButton("قلک", callback_data="fund:ghollak"), InlineKeyboardButton("سورنافود", callback_data="fund:SOURNAFOOD")],
        [InlineKeyboardButton("🔁 فرقی نداره (تصادفی)", callback_data="fund:RANDOM")],
        [InlineKeyboardButton("🔙 بازگشت", callback_data="back_to_main")]
    ]

    text = "یکی از صندوق‌ها رو انتخاب کن:"
    kb = InlineKeyboardMarkup(rows)
    
    await safe_edit_or_send_message(query, text, kb)

async def start_chat_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔙 بازگشت به منوی اصلی", callback_data="back_to_main")]
    ])
    
    text = (
        "💬 حالت گفتگو فعال شد!\n\n"
        "حالا می‌تونی هر سوالی بپرسی و من باهات صحبت کنم. "
        "فقط پیام‌ت رو بفرست! 🤖"
    )
    
    await safe_edit_or_send_message(query, text, kb)

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
    
    # Use safe edit function
    await safe_edit_or_send_message(query, msg, None)
    
    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text="شماره موبایل رو بفرست:",
        reply_markup=reply_kb,
        parse_mode="HTML"
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
            "❌ متأسفانه سرویس گفتگو فعلاً در دسترس نیست.\n"
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
        "/status - وضعیت قرعه‌کشی‌های شما\n"
        "/reports - دسترسی سریع به گزارش‌های ماهانه\n\n"
        "امکانات:\n"
        "🎲 قرعه‌کشی - شرکت در قرعه و برنده شدن واحدهای صندوق\n"
        "📊 گزارش‌های ماهانه - مشاهده گزارش‌های صندوق‌ها\n"
        "💬 گفتگو هوشمند - صحبت با ربات AI\n\n"
        "برای شروع /start را بزنید."
    )
    await update.message.reply_text(help_text)

async def reports_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick access to monthly reports"""
    # Get available fund reports
    available_reports = get_available_fund_reports()
    
    if not available_reports:
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 منوی اصلی", callback_data="back_to_main")]
        ])
        await update.message.reply_text(
            "❌ هیچ گزارش ماهانه‌ای در حال حاضر موجود نیست.\n"
            "لطفاً بعداً دوباره تلاش کنید.",
            reply_markup=kb
        )
        return
    
    # Create buttons for available funds
    rows = []
    fund_buttons = []
    
    for fund_code in available_reports.keys():
        fund_name = FUNDS.get(fund_code, fund_code)
        # Remove "صندوق" prefix for button text
        button_text = fund_name.replace("صندوق ", "")
        fund_buttons.append(InlineKeyboardButton(button_text, callback_data=f"report:{fund_code}"))
        
        # Add row every 2 buttons
        if len(fund_buttons) == 2:
            rows.append(fund_buttons)
            fund_buttons = []
    
    # Add remaining buttons
    if fund_buttons:
        rows.append(fund_buttons)
    
    # Add back button
    rows.append([InlineKeyboardButton("🔙 منوی اصلی", callback_data="back_to_main")])
    
    kb = InlineKeyboardMarkup(rows)
    
    await update.message.reply_text(
        "📊 گزارش ماهانه کدام صندوق را می‌خواهید مشاهده کنید؟\n\n"
        f"تعداد گزارش‌های موجود: {len(available_reports)}",
        reply_markup=kb
    )

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
    
    # Get report views count
    cur.execute(
        "SELECT COUNT(*) FROM report_views WHERE user_id = ?",
        (user_id,),
    )
    report_count = cur.fetchone()[0]
    
    conn.close()
    
    if not draws and chat_count == 0 and report_count == 0:
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
    
    if report_count > 0:
        status_text += f"📊 تعداد گزارش‌های مشاهده شده: {report_count}\n"
    
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔙 بازگشت به منوی اصلی", callback_data="back_to_main")]
    ])
    
    await update.message.reply_text(status_text, reply_markup=kb)

# -----------------------------
# 🚀 Bootstrap
# -----------------------------
# In your main() function, make sure handlers are registered in this order:

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

    # Commands (register first)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("reports", reports_command))
    
    # Phone collection conversation handler (high priority)
    app.add_handler(phone_conv_handler)
    
    # Specific callback handlers (register before generic ones)
    app.add_handler(CallbackQueryHandler(show_fund_report_cb, pattern=r"^report:(.+)$"))
    app.add_handler(CallbackQueryHandler(monthly_reports_cb, pattern=r"^monthly_reports$"))
    app.add_handler(CallbackQueryHandler(start_draw_cb, pattern=r"^start_draw$"))
    app.add_handler(CallbackQueryHandler(start_chat_cb, pattern=r"^start_chat$"))
    app.add_handler(CallbackQueryHandler(back_to_main_cb, pattern=r"^back_to_main$"))

    # AI chat handler (lowest priority - catches remaining text messages)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_ai_message))

    logging.info("🤖 Bot is running with AI chat and Monthly Reports enabled...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
