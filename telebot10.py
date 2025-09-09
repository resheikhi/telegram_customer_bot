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
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Literal, List
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

# Initialize embedding model for RAG
try:
    embedder = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
except Exception as e:
    logging.error(f"Failed to load embedding model: {str(e)}")
    embedder = None

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

# -----------------------------
# 📄 RAG Document Store and Retrieval
# -----------------------------
def init_rag_db():
    """Initialize the RAG documents table and populate with updated sample data"""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    # Create table if it doesn't exist
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    
    # Clear existing documents to ensure new ones are used
    cur.execute("DELETE FROM rag_documents")
    
    # Insert updated sample documents
    if embedder:
        sample_docs = [
            "صندوق رویین: سرمایه‌گذاری در سهام شرکت‌های فلزات اساسی با ریسک متوسط و بازده سالانه 25-30 درصد.",
            "صندوق سام: تمرکز بر اوراق با درآمد ثابت، مناسب برای سرمایه‌گذاری کم‌ریسک با بازده 36 درصد.",
            "صندوق قلک: مناسب برای پس‌انداز بلندمدت با ترکیب سهام و اوراق، ریسک پایین و بازده 35 درصد.",
            "صندوق سورنافود: سرمایه‌گذاری در شرکتهای صنایع غذایی و کشاورزی، با ریسک متوسط و بازده بالقوه 25-40 درصد.",
            "برای شرکت در قرعه‌کشی، باید شماره موبایل خود را ثبت کنید و منتظر تماس تیم پشتیبانی باشید."
        ]
        for doc in sample_docs:
            embedding = embedder.encode(doc).tobytes()
            cur.execute(
                "INSERT INTO rag_documents (content, embedding) VALUES (?, ?)",
                (doc, embedding)
            )
    conn.commit()
    conn.close()

def get_relevant_documents(query: str, top_k: int = 3) -> List[str]:
    """Retrieve top_k relevant documents based on query embedding"""
    if not embedder:
        return []
    
    query_embedding = embedder.encode(query)
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT content, embedding FROM rag_documents")
    docs = cur.fetchall()
    conn.close()
    
    if not docs:
        return []
    
    # Calculate cosine similarity
    similarities = []
    for content, emb_blob in docs:
        doc_embedding = np.frombuffer(emb_blob, dtype=np.float32)
        similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
        similarities.append((content, similarity))
    
    # Sort by similarity and return top_k documents
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in similarities[:top_k]]

def create_financial_agent():
    """Create the financial assistant agent with enhanced RAG support"""
    system_prompt = """You are Sourena, a specialized financial assistant for an Iranian investment company.

STRICT INSTRUCTIONS:
1. Analyze if the user's question is about finance, investments, funds, markets, or economics.
2. If YES (financial): Use the provided document context as the primary source to generate a comprehensive, accurate answer in Persian (max 2-3 sentences). Combine information from all relevant documents to provide a complete response, staying as close as possible to the document content.
3. If NO (non-financial): Use EXACTLY this Persian message: "من دستیار هوش مصنوعی سبدگردان سورنا هستم و فقط در زمینه مالی و سرمایه گذاری می توانم به شما کمک کنم."
4. If no relevant document context is provided for a financial question, respond with: "اطلاعات کافی برای پاسخ به این سوال در دسترس نیست."
5. Always respond in Persian, keep answers concise, professional, and directly tied to the provided document context when available.

Available investment funds to discuss:
- صندوق سام (SAM Fund)
- صندوق رویین (ROUIN Fund) 
- صندوق قلک (ghollak Fund)
- صندوق سورنافود (SOURNAFOOD Fund)

Document context will be provided with the user query. Use it as the primary source for financial questions, combining multiple documents if necessary to provide a complete answer."""

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
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']
    
    for ext in extensions:
        image_files = list(current_dir.glob(ext)) + list(current_dir.glob(ext.upper()))
        for image_file in image_files:
            filename_lower = image_file.name.lower()
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
    init_rag_db()

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

def is_phone_duplicate(phone: str) -> bool:
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 1 FROM draws
        WHERE phone = ?
        LIMIT 1
        """,
        (phone,),
    )
    result = cur.fetchone() is not None
    conn.close()
    return result

def set_phone_for_latest_pending(user_id: int, phone: str) -> int | None:
    if is_phone_duplicate(phone):
        return None
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

async def safe_edit_or_send_message(query, text, reply_markup=None, parse_mode="HTML"):
    try:
        await query.edit_message_text(
            text=text,
            reply_markup=reply_markup,
            parse_mode=parse_mode
        )
    except Exception as e:
        logging.info(f"Could not edit message (likely photo message): {str(e)}")
        try:
            await query.delete_message()
        except Exception:
            pass
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
    await safe_edit_or_send_message(query, text, kb)

# -----------------------------
# 📊 Handlers: Monthly Reports
# -----------------------------
async def monthly_reports_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
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
    rows = []
    fund_buttons = []
    for fund_code in available_reports.keys():
        fund_name = FUNDS.get(fund_code, fund_code)
        button_text = fund_name.replace("صندوق ", "")
        fund_buttons.append(InlineKeyboardButton(button_text, callback_data=f"report:{fund_code}"))
        if len(fund_buttons) == 2:
            rows.append(fund_buttons)
            fund_buttons = []
    if fund_buttons:
        rows.append(fund_buttons)
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
        save_report_view(query.from_user.id, fund_code, fund_name)
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 گزارش‌های دیگر", callback_data="monthly_reports")],
            [InlineKeyboardButton("🔙 منوی اصلی", callback_data="back_to_main")]
        ])
        with open(image_path, 'rb') as photo:
            caption = f"📊 گزارش ماهانه {fund_name}\n\n📅 تاریخ مشاهده: {datetime.now().strftime('%Y/%m/%d - %H:%M')}"
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=photo,
                caption=caption,
                reply_markup=kb
            )
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
    await safe_edit_or_send_message(query, msg, None)
    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text="شماره موبایل رو بفرست:",
        reply_markup=reply_kb,
        parse_mode="HTML"
    )
    return WAITING_PHONE

# -----------------------------
# 📞 Phone collection conversation handler
# -----------------------------
async def phone_received(update: Update, context: ContextTypes.DEFAULT_TYPE):
    contact = update.message.contact
    text = (update.message.text or "").strip() if not contact else None
    if text == "انصراف":
        await update.message.reply_text("لغو شد.", reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END
    phone = None
    if contact:
        phone = sanitize_phone(contact.phone_number)
    elif text:
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
        return ConversationHandler.END
    else:
        if is_phone_duplicate(phone):
            await update.message.reply_text(
                "❌ شماره تماس تکراری است، لطفا با شماره تماس جدید در قرعه کشی شرکت کنید.",
            )
        else:
            await update.message.reply_text(
                "هیچ قرعه‌کشی در انتظار شماره برای شما پیدا نشد.",
                reply_markup=ReplyKeyboardRemove(),
            )
            return ConversationHandler.END
        return WAITING_PHONE

async def cancel_phone(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("عملیات لغو شد.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END

# -----------------------------
# 🤖 Handlers: AI Chat with Enhanced RAG
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
        if not agent:
            logging.info("Initializing financial AI agent...")
            agent = create_financial_agent()
            logging.info("Financial AI agent initialized successfully")
        # Retrieve relevant documents for RAG
        relevant_docs = get_relevant_documents(user_message, top_k=3)
        if relevant_docs:
            # Combine all relevant documents for comprehensive context
            context_str = "\n\nDocument Context:\n" + "\n".join(relevant_docs)
            augmented_message = f"{user_message}\n{context_str}"
        else:
            # If no relevant documents, pass only the user message
            augmented_message = user_message
        response = await agent.run(augmented_message)
        response_text = response.data.message
        logging.info(f"AI response received: {response_text[:100]}...")
        save_chat_history(user_id, user_message, response_text)
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 بازگشت به منوی اصلی", callback_data="back_to_main")]
        ])
        await update.message.reply_text(response_text, reply_markup=kb)
    except ImportError as e:
        logging.error(f"Import error - pydantic_ai or sentence_transformers not installed: {e}")
        await update.message.reply_text(
            "❌ ربات هوشمند فعلاً غیرفعال است.\n"
            "نیاز به نصب pydantic-ai یا sentence-transformers دارد."
        )
    except Exception as e:
        logging.error(f"AI chat error: {str(e)}", exc_info=True)
        error_msg = str(e)
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
    rows = []
    fund_buttons = []
    for fund_code in available_reports.keys():
        fund_name = FUNDS.get(fund_code, fund_code)
        button_text = fund_name.replace("صندوق ", "")
        fund_buttons.append(InlineKeyboardButton(button_text, callback_data=f"report:{fund_code}"))
        if len(fund_buttons) == 2:
            rows.append(fund_buttons)
            fund_buttons = []
    if fund_buttons:
        rows.append(fund_buttons)
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
    cur.execute(
        "SELECT COUNT(*) FROM chat_history WHERE user_id = ?",
        (user_id,),
    )
    chat_count = cur.fetchone()[0]
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
def main():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("❌ TELEGRAM_BOT_TOKEN is missing")
    init_db()
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
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
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("reports", reports_command))
    app.add_handler(phone_conv_handler)
    app.add_handler(CallbackQueryHandler(show_fund_report_cb, pattern=r"^report:(.+)$"))
    app.add_handler(CallbackQueryHandler(monthly_reports_cb, pattern=r"^monthly_reports$"))
    app.add_handler(CallbackQueryHandler(start_draw_cb, pattern=r"^start_draw$"))
    app.add_handler(CallbackQueryHandler(start_chat_cb, pattern=r"^start_chat$"))
    app.add_handler(CallbackQueryHandler(back_to_main_cb, pattern=r"^back_to_main$"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_ai_message))
    logging.info("🤖 Bot is running with AI chat, Monthly Reports, and Enhanced RAG enabled...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
