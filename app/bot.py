from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler, ConversationHandler
from datetime import datetime, timedelta
from typing import Tuple
import logging
import asyncio
import os
from pathlib import Path
import hashlib
import re
import time
import requests

log = logging.getLogger(__name__)

from app.matcher import PreferenceMatcher, match_resume_with_job_embedding
from app.db import SessionLocal
from app.models import User, Preference, Delivery, ChannelPost, PreferredJobPosition
from app.config import settings

from app.classification import job_classifier
from app.langfuse_client import LangfuseSingleton
log.info("Using OpenAI-based job classifier")

matcher = PreferenceMatcher(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", threshold=0.62)
langfuse_client = LangfuseSingleton()

from telegram.constants import ChatType

# LangChain document loaders for PDFs
from langchain_community.document_loaders import PyPDFLoader
from sqlalchemy import or_

# Conversation states
EXTRACTING_KEYWORDS = 1
AWAITING_POSITION_UPDATE = 2

# Store admin keyword extraction sessions in memory
admin_keyword_sessions = {}

async def _notify_admins(ctx: ContextTypes.DEFAULT_TYPE, text: str, reply_to_message_id: int | None = None):
    for admin_id in settings.ADMIN_IDS:
        try:
            await ctx.bot.send_message(chat_id=admin_id, text=text, parse_mode="HTML", reply_to_message_id=reply_to_message_id)
        except Exception as e:
            log.warning(f"Failed to notify admin {admin_id}: {e}")

def _sanitize_username(username: str | None) -> str:
    if not username or not str(username).strip():
        return "unknown"
    s = str(username).strip().lower()
    s = re.sub(r"[^a-z0-9_\-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"

async def _build_telegram_post_link(ctx: ContextTypes.DEFAULT_TYPE, channel_chat_id: int | None, message_id: int | None) -> str | None:
    if not channel_chat_id or not message_id:
        return None
    # Try to fetch channel username for public link t.me/{username}/{message_id}
    try:
        chat = await ctx.bot.get_chat(chat_id=channel_chat_id)
        channel_name = settings.CHANNEL_ID
        if chat.username:
            return f"https://t.me/{channel_name}/{message_id}"
    except Exception as e:
        print(f"Failed to get channel username for {channel_chat_id}: {e}")
        return None

def _format_classification_message(
    employment_type: str | None,
    position: str | None,
    industry: str | None,
    seniority_level: str | None,
    years_experience: int | None,
    work_location: str | None,
    skills_technologies: str | None,
    bonuses: bool | None,
    health_insurance: bool | None,
    stock_options: bool | None,
    work_schedule: str | None,
    company_size: str | None,
):
    def fmt_bool(v: bool | None) -> str:
        if v is True:
            return "بله"
        if v is False:
            return "خیر"
        return "مهم نیست"
    def fmt_val(v) -> str:
        if v is None:
            return "مهم نیست"
        s = str(v).strip()
        return s if s else "مهم نیست"
    lines = [
        "<b>📌 شغل طبقه‌بندی شده</b>",
        f"• 🧑‍💼 <b>نوع اشتغال</b>: {fmt_val(employment_type)}",
        f"• 💼 <b>موقعیت شغلی</b>: {fmt_val(position)}",
        f"• 🏭 <b>صنعت</b>: {fmt_val(industry)}",
        f"• 📈 <b>سطح ارشدیت</b>: {fmt_val(seniority_level)}",
        f"• ⌛️ <b>سال‌های تجربه</b>: {fmt_val(years_experience)}",
        f"• 📍 <b>محل کار</b>: {fmt_val(work_location)}",
        f"• 🛠️ <b>مهارت‌ها/تکنولوژی‌ها</b>: {fmt_val(skills_technologies)}",
        f"• 💰 <b>پاداش</b>: {fmt_bool(bonuses)}",
        f"• 🏥 <b>بیمه درمانی</b>: {fmt_bool(health_insurance)}",
        f"• 📊 <b>سهام شرکت</b>: {fmt_bool(stock_options)}",
        f"• 🗓️ <b>برنامه کاری</b>: {fmt_val(work_schedule)}",
        f"• 🏢 <b>اندازه شرکت</b>: {fmt_val(company_size)}",
    ]
    return "\n".join(lines)

async def extract_keywords(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in settings.ADMIN_IDS:
        await update.message.reply_text("این دستور فقط برای ادمین‌ها است.")
        return ConversationHandler.END

    # Start a new session for this admin
    admin_keyword_sessions[user_id] = []
    await update.message.reply_text(
        "تمام پست‌های شغلی کانال را که می‌خواهید کلمات کلیدی از آن‌ها استخراج شود، فوروارد کنید. "
        "وقتی تمام شد، /extract_keywords_end را ارسال کنید."
    )
    return EXTRACTING_KEYWORDS

async def extract_keywords_end(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in settings.ADMIN_IDS:
        await update.message.reply_text("این دستور فقط برای ادمین‌ها است.")
        return ConversationHandler.END

    posts = admin_keyword_sessions.pop(user_id, [])
    if not posts:
        await update.message.reply_text("هیچ پیام فوروارد شده‌ای دریافت نشد. لطفاً ابتدا پست‌های کانال را فوروارد کنید.")
        return ConversationHandler.END

    # Process each post: extract keywords and classify
    all_keywords = set()
    classified_posts = 0
    
    with SessionLocal() as db:
        for post_data in posts:
            if post_data and 'text' in post_data:
                post_text = post_data['text']
                
                try:
                    classification = job_classifier.classify_job(post_text)
                    
                    # Update the existing post with classification data
                    existing_post = db.query(ChannelPost).filter(
                        ChannelPost.channel_msg_id == post_data['msg_id']
                    ).first()
                    
                    if existing_post:
                        # Store individual fields for indexing and querying
                        existing_post.employment_type = classification.employment_type
                        existing_post.position = classification.position
                        existing_post.industry = classification.industry
                        existing_post.seniority_level = classification.seniority_level
                        # Save numeric years of experience if provided
                        try:
                            existing_post.years_experience = int(classification.years_experience) if classification.years_experience is not None else None
                        except Exception:
                            existing_post.years_experience = None
                        existing_post.work_location = classification.work_location
                        existing_post.skills_technologies = classification.skills_technologies
                        existing_post.bonuses = classification.bonuses
                        existing_post.health_insurance = classification.health_insurance
                        existing_post.stock_options = classification.stock_options
                        existing_post.work_schedule = classification.work_schedule
                        existing_post.company_size = classification.company_size
                        existing_post.company_name = classification.company_name
                        existing_post.is_classified = True
                        
                        # Store complete Pydantic data as JSON
                        existing_post.classification_data = classification.model_dump()
                        
                        classified_posts += 1
                        
                except Exception as e:
                    log.error(f"Classification failed for post {post_data['msg_id']}: {e}")
        
        # Save extracted keywords to database (if not already present)
        for kw in all_keywords:
            exists = db.query(Preference).filter(Preference.user_id == 0, Preference.text == kw).first()
            if not exists:
                pref = Preference(user_id=0, text=kw)
                db.add(pref)
        
        db.commit()

    # Send summary
    summary = f"✅ پردازش تکمیل شد!\n\n"
    summary += f"📊 {classified_posts} پست شغلی طبقه‌بندی شد\n"
    summary += f"🔑 {len(all_keywords)} کلمه کلیدی استخراج شد\n\n"
    
    if all_keywords:
        summary += f"کلمات کلیدی: {', '.join(sorted(all_keywords))}"
    
    await update.message.reply_text(summary)
    return ConversationHandler.END

# Handler to collect forwarded messages from admin during extraction session
async def collect_forwarded_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in settings.ADMIN_IDS:
        return EXTRACTING_KEYWORDS
    if user_id not in admin_keyword_sessions:
        return EXTRACTING_KEYWORDS
    # Only accept forwarded messages from a channel
    msg = update.message
    if msg and msg.forward_from_chat and msg.forward_from_chat.type == ChatType.CHANNEL:
        text = msg.text or msg.caption
        if text:
            admin_keyword_sessions[user_id].append({
                'text': text,
                'msg_id': msg.forward_from_message_id,
                'chat_id': msg.forward_from_chat.id,
                'date': msg.date
            })
            # Store the post in database
            with SessionLocal() as db:
                # Check if post already exists
                existing = db.query(ChannelPost).filter(
                    ChannelPost.channel_msg_id == msg.forward_from_message_id
                ).first()
                if not existing:
                    post = ChannelPost(
                        channel_msg_id=msg.forward_from_message_id,
                        channel_chat_id=msg.forward_from_chat.id,
                        text=text or "",
                        caption=msg.caption or "",
                        posted_at=msg.date
                    )
                    db.add(post)
                    db.commit()
                else:
                    # Backfill missing forwarding metadata and content if absent
                    changed = False
                    try:
                        if getattr(existing, "channel_chat_id", None) is None and msg.forward_from_chat:
                            existing.channel_chat_id = msg.forward_from_chat.id
                            changed = True
                    except Exception:
                        pass
                    if (not getattr(existing, "text", None)) and text:
                        existing.text = text
                        changed = True
                    if (not getattr(existing, "caption", None)) and msg.caption:
                        existing.caption = msg.caption
                        changed = True
                    if changed:
                        db.add(existing)
                        db.commit()

                # Classify this message immediately and update the stored record
                try:
                    classification = job_classifier.classify_job(text)
                    target = existing or post
                    target.employment_type = classification.employment_type
                    target.position = classification.position
                    target.industry = classification.industry
                    target.seniority_level = classification.seniority_level
                    try:
                        target.years_experience = int(classification.years_experience) if classification.years_experience is not None else None
                    except Exception:
                        target.years_experience = None
                    target.work_location = classification.work_location
                    target.skills_technologies = classification.skills_technologies
                    target.bonuses = classification.bonuses
                    target.health_insurance = classification.health_insurance
                    target.stock_options = classification.stock_options
                    target.work_schedule = classification.work_schedule
                    target.company_size = classification.company_size
                    target.company_name = classification.company_name
                    target.is_classified = True
                    target.classification_data = classification.model_dump()
                    db.add(target)
                    db.commit()

                    # Reply with classification summary right away (reply to the forwarded post)
                    await update.message.reply_text(
                        _format_classification_message(
                            target.employment_type,
                            target.position,
                            target.industry,
                            target.seniority_level,
                            target.years_experience,
                            target.work_location,
                            target.skills_technologies,
                            target.bonuses,
                            target.health_insurance,
                            target.stock_options,
                            target.work_schedule,
                            target.company_size,
                        ),
                        parse_mode="HTML",
                        quote=True,
                    )
                except Exception as e:
                    log.error(f"Classification failed for post {msg.forward_from_message_id}: {e}")
                    await update.message.reply_text("طبقه‌بندی این پیام ناموفق بود.")
            # Keep session active until /extract_keywords_end
            return EXTRACTING_KEYWORDS
    return EXTRACTING_KEYWORDS


async def select_keyword(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    selected_keyword = query.data
    user_id = query.from_user.id
    with SessionLocal() as db:
        pref = Preference(user_id=user_id, text=selected_keyword)
        db.add(pref)
        db.commit()
    await query.answer()
    await query.edit_message_text(text=f"شما انتخاب کردید: {selected_keyword}. پست‌های شغلی مرتبط را برای شما فوروارد خواهم کرد.")

async def handle_text_messages(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle all text messages in private chats - route to appropriate handler"""
    if not update.message or not update.message.text:
        return
    # If we're waiting for text after /update_my_position, route to that handler
    if ctx.user_data.get("awaiting_position_update"):
        await handle_position_text_if_waiting(update, ctx)
        return
    # Otherwise, guide the user to commands
    help_hint = (
        "لطفاً از دستورات برای تعامل با بات استفاده کنید:\n"
        "• /update_my_position – تنظیم یا به‌روزرسانی موقعیت شغلی مورد نظر\n"
        "• /my_position – مشاهده تنظیمات ذخیره شده\n"
        "• /help – مشاهده تمام دستورات و نحوه استفاده"
    )
    await help_command(update, ctx)


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    welcome_text = (
        "🤖 <b>به بات شغلی CE خوش آمدید!</b>\n\n"
        "📋 <b>چگونه کار می‌کند:</b>\n"
        "1️⃣ <b>تنظیم ترجیحات شغلی:</b>\n"
        "   • با زدن دستور /update_my_position می‌توانید فایل PDF رزومه خود را آپلود کنید، یا\n"
        "   • نوع شغل مورد نظرتان را به صورت متن بفرستید\n\n"
        "2️⃣ <b>جستجو در مشاغل اخیر:</b>\n"
        "   • از دستور /match_positions استفاده کنید\n"
        "   • مشاغل مرتبط با ترجیحات شما در یک ماه اخیر را می‌بینید\n\n"
        "3️⃣ <b>دریافت مشاغل جدید:</b>\n"
        "   • به صورت خودکار مشاغل جدید مرتبط را دریافت می‌کنید\n"
        "   • برای فعال‌سازی: /activate_new_positions\n"
        "   • برای غیرفعال‌سازی: /deactivate_new_positions\n\n"
        "📱 <b>دستورات مفید:</b>\n"
        "• /my_position - مشاهده ترجیحات ذخیره شده\n"
        "• /help - راهنمای کامل\n\n"
        "🚀 <b>شروع کنید:</b> /update_my_position"
    )
    await update.message.reply_text(welcome_text, parse_mode="HTML")
    user = update.effective_user
    username = f"@{user.username}" if user.username else f"user_id {user.id}"
    await _notify_admins(ctx, f"🚀 کاربر {username} بات را شروع کرد (/start)")

async def stop(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("دیگر به‌روزرسانی‌های شغلی دریافت نخواهید کرد. برای راه‌اندازی مجدد، از /start استفاده کنید.")

async def fetch_channel_posts(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Admin command to fetch recent posts from the channel and store them"""
    user_id = update.effective_user.id
    if user_id not in settings.ADMIN_IDS:
        await update.message.reply_text("این دستور فقط برای ادمین‌ها است.")
        return

    try:
        # Get recent messages from the channel (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        
        # Note: This is a simplified approach. In practice, you'd need to:
        # 1. Use Telegram Client API (not Bot API) to get channel history
        # 2. Or have a separate service that monitors the channel
        # 3. Or manually forward posts to the bot
        
        await update.message.reply_text(
            "دریافت پست‌های کانال نیاز به دسترسی Telegram Client API دارد. "
            "فعلاً لطفاً پست‌های کانال را به صورت دستی با استفاده از دستور /extract_keywords فوروارد کنید."
        )
        
    except Exception as e:
        log.error(f"Error fetching channel posts: {e}")
        await update.message.reply_text(f"خطا در دریافت پست‌های کانال: {str(e)}")

def build_app():
    """Build Telegram bot application with comprehensive logging"""
    log.info("Building Telegram bot application...")
    start_time = time.time()
    
    # Test Telegram bot token connectivity
    _test_telegram_connectivity()
    
    async def post_init(app):
        log.info("Running post-initialization tasks...")
        try:
            # Test bot connection
            bot_info = await app.bot.get_me()
            log.info(f"✅ Bot connected successfully: @{bot_info.username} ({bot_info.first_name})")
            log.info(f"Bot ID: {bot_info.id}")
            
            # Set bot commands
            await app.bot.set_my_commands([
                BotCommand("start", "شروع بات"),
                BotCommand("help", "نمایش راهنما و دستورات موجود"),
                BotCommand("update_my_position", "تنظیم/به‌روزرسانی موقعیت شغلی مورد نظر"),
                BotCommand("my_position", "نمایش موقعیت شغلی ذخیره شده"),
                BotCommand("match_positions", "جستجو و فوروارد مشاغل مطابق اخیر"),
                BotCommand("activate_new_positions", "فعال‌سازی دریافت مشاغل مطابق جدید"),
                BotCommand("deactivate_new_positions", "غیرفعال‌سازی دریافت مشاغل مطابق جدید"),
                BotCommand("stop", "توقف دریافت به‌روزرسانی‌ها"),
            ])
            log.info("✅ Bot commands set successfully")
            
        except Exception as e:
            log.error(f"❌ Post-initialization failed: {e}")
            log.error(f"Error type: {type(e).__name__}")

    try:
        application = ApplicationBuilder().token(settings.TELEGRAM_BOT_TOKEN).post_init(post_init).build()
        
        # Conversation handler for admin keyword extraction
        keyword_extraction_conv = ConversationHandler(
            entry_points=[CommandHandler("extract_keywords", extract_keywords)],
            states={
                EXTRACTING_KEYWORDS: [
                    CommandHandler("extract_keywords_end", extract_keywords_end),
                    MessageHandler(filters.TEXT & ~filters.COMMAND, collect_forwarded_message),
                ],
            },
            fallbacks=[CommandHandler("extract_keywords_end", extract_keywords_end)],
        )
        
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("stop", stop))
        application.add_handler(CommandHandler("fetch_posts", fetch_channel_posts))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(keyword_extraction_conv)
        application.add_handler(CommandHandler("update_my_position", update_my_position))
        application.add_handler(CommandHandler("my_position", my_position))
        application.add_handler(CommandHandler("match_positions", match_positions))
        application.add_handler(CommandHandler("activate_new_positions", activate_new_positions))
        application.add_handler(CommandHandler("deactivate_new_positions", deactivate_new_positions))
        application.add_handler(CallbackQueryHandler(select_keyword))
        # Handle channel posts (bot must be admin in the channel) BEFORE generic text handlers
        application.add_handler(MessageHandler(filters.ChatType.CHANNEL, on_channel_post))
        # Private chat-only handlers
        application.add_handler(MessageHandler(filters.ChatType.PRIVATE & filters.Document.PDF, handle_position_document))
        application.add_handler(MessageHandler(filters.ChatType.PRIVATE & (filters.TEXT & ~filters.COMMAND), handle_text_messages))
        
        build_time = time.time() - start_time
        log.info(f"✅ Telegram bot application built successfully in {build_time:.2f}s")
        return application
    except Exception as e:
        build_time = time.time() - start_time
        log.error(f"❌ Failed to build Telegram bot application after {build_time:.2f}s: {e}")
        raise

def _test_telegram_connectivity():
    """Test Telegram bot API connectivity"""
    log.info("Testing Telegram bot API connectivity...")
    start_time = time.time()
    
    try:
        # Test basic connectivity to Telegram API
        test_url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/getMe"
        response = requests.get(test_url, timeout=10)
        response_time = time.time() - start_time
        
        log.info(f"Telegram API connectivity test - Status: {response.status_code}, Time: {response_time:.2f}s")
        
        if response.status_code == 200:
            bot_info = response.json()
            if bot_info.get('ok'):
                log.info("✅ Telegram bot API connectivity test passed")
                log.info(f"Bot info: {bot_info.get('result', {})}")
            else:
                log.warning(f"⚠️ Telegram API returned error: {bot_info}")
        else:
            log.warning(f"⚠️ Telegram API returned status {response.status_code}: {response.text[:200]}")
            
    except requests.exceptions.ConnectionError as e:
        log.error(f"❌ Telegram API connection failed: {e}")
    except requests.exceptions.Timeout as e:
        log.error(f"❌ Telegram API request timeout: {e}")
    except Exception as e:
        log.error(f"❌ Telegram API connectivity test failed: {e}")


async def ensure_user_record(update: Update):
    user = update.effective_user
    chat = update.effective_chat
    with SessionLocal() as db:
        db_user = db.query(User).filter(User.user_id == user.id).first()
        if not db_user:
            db_user = User(user_id=user.id, username=user.username, enabled=True, chat_id=chat.id)
            db.add(db_user)
        else:
            # Update username/chat id if changed
            changed = False
            if db_user.username != user.username:
                db_user.username = user.username
                changed = True
            if db_user.chat_id != chat.id:
                db_user.chat_id = chat.id
                changed = True
            if changed:
                db.add(db_user)
        db.commit()


async def update_my_position(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    await update.message.reply_text(
        "جزئیات موقعیت شغلی مورد نظر خود را به صورت متن ارسال کنید، یا رزومه PDF آپلود کنید."
    )
    # Flag the context that we're awaiting an update
    ctx.user_data["awaiting_position_update"] = True


def _classify_and_save_preference(db, user_id: int, text: str, resume_file_id: str | None = None, resume_file_path: str | None = None) -> PreferredJobPosition:
    classification = job_classifier.classify_job(text or "")
    # Upsert PreferredJobPosition
    preferred = db.query(PreferredJobPosition).filter(PreferredJobPosition.user_id == user_id).first()
    if not preferred:
        preferred = PreferredJobPosition(user_id=user_id)
        db.add(preferred)

    # Update fields if provided by classification
    preferred.employment_type = classification.employment_type or preferred.employment_type
    preferred.position = classification.position or preferred.position
    preferred.industry = classification.industry or preferred.industry
    preferred.seniority_level = classification.seniority_level or preferred.seniority_level
    try:
        preferred.years_experience = int(classification.years_experience) if classification.years_experience is not None else preferred.years_experience
    except Exception:
        pass
    preferred.work_location = classification.work_location or preferred.work_location
    preferred.skills_technologies = classification.skills_technologies or preferred.skills_technologies
    preferred.bonuses = classification.bonuses if classification.bonuses is not None else preferred.bonuses
    preferred.health_insurance = classification.health_insurance if classification.health_insurance is not None else preferred.health_insurance
    preferred.stock_options = classification.stock_options if classification.stock_options is not None else preferred.stock_options
    preferred.work_schedule = classification.work_schedule or preferred.work_schedule
    preferred.company_size = classification.company_size or preferred.company_size
    preferred.classification_data = classification.model_dump()

    # Append raw user text for traceability and future refinements
    if text and text.strip():
        if preferred.preferred_position_text and preferred.preferred_position_text.strip():
            preferred.preferred_position_text = preferred.preferred_position_text + " ---> " + text.strip()
        else:
            preferred.preferred_position_text = text.strip()

    # Persist resume metadata on user if provided
    if resume_file_id is not None or resume_file_path is not None:
        user = db.query(User).filter(User.user_id == user_id).first()
        if user:
            if resume_file_id is not None:
                user.resume_file_id = resume_file_id
            if resume_file_path is not None:
                user.resume_file_path = resume_file_path
            db.add(user)

    db.add(preferred)
    db.commit()
    return preferred


def _format_preferred_position_message(preferred: PreferredJobPosition) -> str:
    def fmt_bool(v: bool | None) -> str:
        if v is True:
            return "بله"
        if v is False:
            return "خیر"
        return "مهم نیست"

    def fmt_val(v) -> str:
        if v is None:
            return "مهم نیست"
        s = str(v).strip()
        return s if s else "مهم نیست"

    lines = [
        "<b>🎯 موقعیت شغلی مورد نظر شما</b>",
        f"• 🧑‍💼 <b>نوع اشتغال</b>: {fmt_val(preferred.employment_type)}",
        f"• 💼 <b>موقعیت شغلی</b>: {fmt_val(preferred.position)}",
        f"• 🏭 <b>صنعت</b>: {fmt_val(preferred.industry)}",
        f"• 📈 <b>سطح ارشدیت</b>: {fmt_val(preferred.seniority_level)}",
        f"• ⌛️ <b>سال‌های تجربه</b>: {fmt_val(preferred.years_experience)}",
        f"• 📍 <b>محل کار</b>: {fmt_val(preferred.work_location)}",
        f"• 🛠️ <b>مهارت‌ها/تکنولوژی‌ها</b>: {fmt_val(preferred.skills_technologies)}",
        f"• 💰 <b>پاداش</b>: {fmt_bool(preferred.bonuses)}",
        f"• 🏥 <b>بیمه درمانی</b>: {fmt_bool(preferred.health_insurance)}",
        f"• 📊 <b>سهام شرکت</b>: {fmt_bool(preferred.stock_options)}",
        f"• 🗓️ <b>برنامه کاری</b>: {fmt_val(preferred.work_schedule)}",
        f"• 🏢 <b>اندازه شرکت</b>: {fmt_val(preferred.company_size)}",
    ]
    return "\n".join(lines)


async def handle_position_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    # Always forward any received PDF to admins
    try:
        for admin_id in settings.ADMIN_IDS:
            try:
                await ctx.bot.forward_message(
                    chat_id=admin_id,
                    from_chat_id=update.effective_chat.id,
                    message_id=update.message.message_id,
                )
            except Exception as e:
                log.warning(f"Failed to forward user PDF to admin {admin_id}: {e}")
    except Exception as e:
        log.warning(f"Failed forwarding PDF to admins: {e}")

    if not (getattr(ctx, "user_data", None) and ctx.user_data.get("awaiting_position_update")):
        return
    await ensure_user_record(update)
    user_id = update.effective_user.id
    document = update.message.document
    if not document or document.mime_type != "application/pdf":
        await update.message.reply_text("لطفاً یک سند PDF آپلود کنید.")
        return
    # Download PDF to a temp folder
    base_dir = Path(os.getenv("RESUME_DIR", "./resumes")).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    # Build a filename with username and short hashes
    username_label = _sanitize_username(update.effective_user.username) or f"user{user_id}"
    uid_part = (document.file_unique_id[-6:] if document.file_unique_id else "nofuid")
    ts = int(datetime.utcnow().timestamp())
    hash_input = f"{user_id}:{document.file_unique_id}:{ts}".encode("utf-8")
    short_hash = hashlib.sha256(hash_input).hexdigest()[:10]
    file_path = base_dir / f"resume_{username_label}_{uid_part}_{short_hash}.pdf"
    file = await document.get_file()
    await file.download_to_drive(str(file_path))

    # Load text via LangChain PyPDFLoader
    try:
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
        text_content = "\n\n".join(page.page_content for page in pages)
        log.info(f"OCRed text_content: {text_content}")
    except Exception as e:
        text_content = ""
        log.error(f"Error loading PDF: {e}")

    with SessionLocal() as db:
        preferred = _classify_and_save_preference(
            db,
            user_id=user_id,
            text=text_content,
            resume_file_id=document.file_id,
            resume_file_path=str(file_path),
        )

    ctx.user_data["awaiting_position_update"] = False
    await update.message.reply_text(
        "موقعیت شغلی مورد نظر شما از رزومه به‌روزرسانی شد."
    )
    await update.message.reply_text(_format_preferred_position_message(preferred), parse_mode="HTML")
    # Notify admins about the update (source and result)
    user = update.effective_user
    username = f"@{user.username}" if user.username else f"user_id {user.id}"
    await _notify_admins(
        ctx,
        ("✍️ به‌روزرسانی موقعیت شغلی توسط " + username + " (از PDF)\n\n"
         + "<b>نتیجه به‌روزرسانی:</b>\n" + _format_preferred_position_message(preferred))
    )


async def handle_position_text_if_waiting(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not (getattr(ctx, "user_data", None) and ctx.user_data.get("awaiting_position_update")):
        return
    await ensure_user_record(update)
    user_id = update.effective_user.id
    text = update.message.text or ""
    with SessionLocal() as db:
        preferred = _classify_and_save_preference(db, user_id=user_id, text=text)
    ctx.user_data["awaiting_position_update"] = False
    await update.message.reply_text("موقعیت شغلی مورد نظر شما به‌روزرسانی شد.")
    await update.message.reply_text(_format_preferred_position_message(preferred), parse_mode="HTML")
    # Notify admins about the update (source and result)
    user = update.effective_user
    username = f"@{user.username}" if user.username else f"user_id {user.id}"
    preview = (text[:300] + ("…" if len(text) > 300 else "")) if text else "(متن خالی)"
    await _notify_admins(
        ctx,
        ("✍️ به‌روزرسانی موقعیت شغلی توسط " + username + " (متن)\n\n"
         + "<b>متن ارسالی:</b>\n" + preview + "\n\n"
         + "<b>نتیجه به‌روزرسانی:</b>\n" + _format_preferred_position_message(preferred))
    )


async def my_position(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    with SessionLocal() as db:
        preferred = db.query(PreferredJobPosition).filter(PreferredJobPosition.user_id == user_id).first()
        if not preferred:
            await update.message.reply_text("هنوز موقعیت شغلی مورد نظر تنظیم نشده است. از /update_my_position برای ارائه جزئیات استفاده کنید.")
            return
        await update.message.reply_text(_format_preferred_position_message(preferred), parse_mode="HTML")


async def help_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = (
        "<b>🤖 نحوه استفاده از بات شغلی CE</b>\n\n"
        "• /start – شروع بات\n"
        "• /help – نمایش این راهنما\n"
        "• /update_my_position – ارسال متن یا آپلود رزومه PDF برای تنظیم/به‌روزرسانی ترجیحات\n"
        "• /my_position – مشاهده موقعیت شغلی مورد نظر ذخیره شده\n"
        "• /match_positions – جستجو و فوروارد مشاغل مطابق اخیر بر اساس ترجیحات شما\n"
        "• /activate_new_positions – فعال‌سازی دریافت مشاغل مطابق جدید\n"
        "• /deactivate_new_positions – غیرفعال‌سازی دریافت مشاغل مطابق جدید\n"
        "• /stop – توقف دریافت به‌روزرسانی‌ها\n\n"
        "نکته: پس از /update_my_position، پیام بعدی شما (متن یا PDF) برای به‌روزرسانی ترجیحات استفاده خواهد شد."
    )
    await update.message.reply_text(text, parse_mode="HTML")


def _skills_overlap_ratio(candidate: str | None, required: str | None) -> float:
    """
    Returns 1.0 if there is at least one overlapping skill/technology between candidate and required,
    otherwise returns 0.0. If either is empty, returns 0.0.
    """
    if not candidate or not required:
        return 0.0
    def tokenize(s: str) -> set[str]:
        tokens = [t.strip().lower() for t in s.replace("/", ",").replace(";", ",").split(",")]
        return {t for t in tokens if t}
    cand = tokenize(candidate)
    req = tokenize(required)
    if not cand or not req:
        return 0.0
    inter = cand.intersection(req)
    return 1.0 if inter else 0.0


def _seniority_rank(value: str | None) -> int:
    order = {
        "intern": 0,
        "entry-level": 1,
        "junior": 1,
        "mid": 2,
        "mid-level": 2,
        "senior": 3,
        "lead": 4,
        "principal": 5,
        "manager": 6,
        "director": 7,
        "vp": 8,
    }
    if not value:
        return None
    key = value.strip().lower()
    return order.get(key, None)


def _get_user_resume_text(user_id: int, db) -> str:
    """Get resume text for a user from database or file"""
    try:
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            return ""
        
        # Try to get text from preferred_position_text first
        preferred = db.query(PreferredJobPosition).filter(PreferredJobPosition.user_id == user_id).first()
        if preferred and preferred.preferred_position_text:
            return preferred.preferred_position_text
        
        # If resume file path exists, try to read it
        if user.resume_file_path and os.path.exists(user.resume_file_path):
            try:
                loader = PyPDFLoader(user.resume_file_path)
                pages = loader.load()
                text_content = "\n\n".join(page.page_content for page in pages)
                return text_content
            except Exception as e:
                log.warning(f"Failed to load resume from file {user.resume_file_path}: {e}")
                return ""
        
        return ""
    except Exception as e:
        log.error(f"Error getting resume text: {e}")
        return ""


async def _generate_match_explanation(resume_text: str, job_text: str) -> str:
    """Generate detailed match explanation using Langfuse prompt"""
    try:
        # Use the Langfuse singleton to get the match explanation
        explanation = langfuse_client.ask(
            prompt_name="match-explanation",
            resume_text=resume_text,
            job_text=job_text,
            temperature=0.5
        )
        return explanation
    except Exception as e:
        log.error(f"Error generating match explanation: {e}")
        # Fallback to a simple explanation
        return f"این موقعیت شغلی با مهارت‌ها و تجربیات شما همخوانی دارد. مهارت‌های کلیدی شما شامل {resume_text[:100]}... می‌باشد که با نیازهای این شغل مطابقت دارد."


def _is_position_match(preferred: PreferredJobPosition, post: ChannelPost) -> bool:
    """Check if a job post matches user preferences, with detailed debug prints for mismatches"""
    def eq(a, b):
        if a is None or str(a).strip() == "":
            return True
        return (str(a).strip().lower() == str(b or "").strip().lower())

    # Basic matching criteria
    if not eq(preferred.employment_type, post.employment_type):
        print(
            f"[MATCH DEBUG] employment_type mismatch: "
            f"user='{preferred.employment_type}' vs post='{post.employment_type}'"
        )
        return False
    if not eq(preferred.position, post.position):
        print(
            f"[MATCH DEBUG] position mismatch: "
            f"user='{preferred.position}' vs post='{post.position}'"
        )
        return False
    if not eq(preferred.work_schedule, post.work_schedule):
        print(
            f"[MATCH DEBUG] work_schedule mismatch: "
            f"user='{preferred.work_schedule}' vs post='{post.work_schedule}'"
        )
        return False

    # Seniority level check - post level should not be higher than user's preferred level
    user_rank = _seniority_rank(preferred.seniority_level)
    post_rank = _seniority_rank(post.seniority_level)
    if user_rank is not None and post_rank is not None and post_rank > user_rank:
        print(
            f"[MATCH DEBUG] seniority_level mismatch: "
            f"user='{preferred.seniority_level}' (rank={user_rank}) vs post='{post.seniority_level}' (rank={post_rank})"
        )
        return False

    # Skills overlap check - at least 50% overlap required (only if user has skills specified)
    if preferred.skills_technologies:
        overlap = _skills_overlap_ratio(preferred.skills_technologies, post.skills_technologies)
        if overlap < 0.5:
            print(
                f"[MATCH DEBUG] skills_technologies overlap too low: "
                f"user='{preferred.skills_technologies}' vs post='{post.skills_technologies}' "
                f"(overlap={overlap:.2f})"
            )
            return False

    return True


async def _is_position_match_with_embedding(preferred: PreferredJobPosition, post: ChannelPost, db) -> Tuple[bool, float, str]:
    """
    Check if a job post matches user preferences using both old algorithm and embedding-based matching
    Returns Tuple of (matches, score, explanation)
    """
    # First check with old algorithm
    old_match = _is_position_match(preferred, post)
    
    # Get resume text and job text
    user = db.query(User).filter(User.user_id == preferred.user_id).first()
    resume_skills = preferred.skills_technologies or ""
    job_text = (post.text or "") or (post.caption or "")
    
    # Use the skills/technologies from the job post for embedding matching
    skills_text = post.skills_technologies or ""
    threshold = 0.7
    if resume_skills and (job_text or skills_text):
        # Try embedding matching
        try:
            embedding_score, basic_explanation = match_resume_with_job_embedding(
                resume_text=resume_skills,
                job_text=skills_text if skills_text else job_text,
                threshold=threshold
            )
        except Exception as e:
            log.error(f"Error in embedding matching: {e}")
            embedding_score, basic_explanation = 0.0, "خطا در محاسبه تطابق"
    else:
        embedding_score, basic_explanation = 0.0, "متن کافی برای محاسبه تطابق موجود نیست"
    
    # Combine old and new matching: must pass old match AND have embedding score >= 0.5
    final_match = embedding_score >= threshold
    
    # Generate detailed explanation if it's a match

    resume_text = _get_user_resume_text(user.user_id, db=db)
    if final_match and resume_text and job_text:
        try:
            detailed_explanation = await _generate_match_explanation(resume_text, job_text)
            return final_match, embedding_score, detailed_explanation
        except Exception as e:
            log.error(f"Error generating detailed explanation: {e}")
            return final_match, embedding_score, basic_explanation
    else:
        return final_match, embedding_score, basic_explanation


async def match_positions(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await update.message.reply_text("در حال جستجوی موقعیت‌های شغلی مرتبط با ترجیحات شما...")
    with SessionLocal() as db:
        preferred = db.query(PreferredJobPosition).filter(PreferredJobPosition.user_id == user_id).first()
        if not preferred:
            await update.message.reply_text("هنوز موقعیت شغلی مورد نظر ندارید. ابتدا از /update_my_position استفاده کنید.")
            return

        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        posts = db.query(ChannelPost).filter(
            ChannelPost.posted_at >= thirty_days_ago,
            ChannelPost.is_classified == True
        ).order_by(ChannelPost.posted_at.desc()).limit(200).all()
        
        sent = 0
        total_checked = 0
        total_posts = len(posts)
        
        # Send initial progress message
        if total_posts > 0:
            await update.message.reply_text(f"🔍 در حال بررسی {total_posts} پست شغلی...")
        
        for p in posts:
            total_checked += 1
            print(f"trying to match post {p.channel_msg_id}")
            is_match, score, explanation = await _is_position_match_with_embedding(preferred, p, db)
            
            if is_match:
                try:
                    if p.channel_chat_id:
                        # Forward the post immediately
                        await ctx.bot.forward_message(
                            chat_id=update.effective_chat.id,
                            from_chat_id=p.channel_chat_id,
                            message_id=p.channel_msg_id,
                        )
                        sent += 1
                        
                        # Send score and explanation immediately
                        score_message = f"📊 <b>امتیاز تطابق:</b> {int(score * 100)}%\n\n{explanation}"
                        await ctx.bot.send_message(
                            chat_id=update.effective_chat.id,
                            text=score_message,
                            parse_mode="HTML"
                        )
                        
                        # Admin notification about delivery
                        user = update.effective_user
                        username = f"@{user.username}" if user.username else f"user_id {user.id}"
                        link = await _build_telegram_post_link(ctx, p.channel_chat_id, p.channel_msg_id)
                        note = (
                            f"📨 پست شغلی به {username} ارسال شد (جستجوی دستی)\n"
                            + f"📊 امتیاز: {int(score * 100)}%\n"
                            + (f"🔗 لینک: {link}" if link else f"🆔 شناسه: #{p.channel_msg_id}")
                        )
                        await _notify_admins(ctx, note)
                    else:
                        # Fallback to sending raw text if we can't forward
                        text = (p.text or "") or (p.caption or "")
                        if text:
                            await update.message.reply_text(text[:4000])
                            # Send score and explanation immediately
                            score_message = f"📊 <b>امتیاز تطابق:</b> {int(score * 100)}%\n\n{explanation}"
                            await update.message.reply_text(score_message, parse_mode="HTML")
                            sent += 1
                            user = update.effective_user
                            username = f"@{user.username}" if user.username else f"user_id {user.id}"
                            link = await _build_telegram_post_link(ctx, p.channel_chat_id, p.channel_msg_id)
                            note = (
                                f"📨 متن پست شغلی به {username} ارسال شد (جستجوی دستی)\n"
                                + f"📊 امتیاز: {int(score * 100)}%\n"
                                + (f"🔗 لینک: {link}" if link else f"🆔 شناسه: #{p.channel_msg_id}")
                            )
                            await _notify_admins(ctx, note)
                except Exception as e:
                    log.warning(f"Failed to forward message {p.channel_msg_id}: {e}")
                    try:
                        text = (p.text or "") or (p.caption or "")
                        if text:
                            await update.message.reply_text(text[:4000])
                            score_message = f"📊 <b>امتیاز تطابق:</b> {int(score * 100)}%\n\n{explanation}"
                            await update.message.reply_text(score_message, parse_mode="HTML")
                            sent += 1
                    except Exception as e2:
                        log.warning(f"Also failed to send text fallback: {e2}")
            else:
                print(f"post {p.channel_msg_id} did not match user preferences with score of {score}")

        # Send summary message
        if sent == 0:
            await update.message.reply_text(f"❌ از {total_posts} پست بررسی شده، هیچ موقعیت شغلی مطابق یافت نشد.")
        else:
            await update.message.reply_text(f"✅ از {total_posts} پست بررسی شده، {sent} پست مطابق مهارت شما فوروارد شد.")


async def _notify_matched_users_for_post(ctx: ContextTypes.DEFAULT_TYPE, post: ChannelPost):
    """Notify users with active preferences that match this post and record deliveries."""
    with SessionLocal() as db:
        # Join preferences with users to get chat_id and enabled flag
        pairs = (
            db.query(PreferredJobPosition, User)
            .join(User, PreferredJobPosition.user_id == User.user_id)
            .filter(
                User.enabled == True,
                User.chat_id.isnot(None),
                # Treat NULL (older rows) as active=True
                or_(PreferredJobPosition.active == True, PreferredJobPosition.active.is_(None)),
            )
            .all()
        )

        for preferred, user in pairs:
            # Use the new embedding-based matching logic
            is_match, score, explanation = await _is_position_match_with_embedding(preferred, post, db)
            
            if not is_match:
                continue

            try:
                if user.chat_id:
                    if post.channel_chat_id:
                        await ctx.bot.forward_message(
                            chat_id=user.chat_id,
                            from_chat_id=post.channel_chat_id,
                            message_id=post.channel_msg_id,
                        )
                        
                        # Send score and explanation
                        score_message = f"📊 <b>امتیاز تطابق:</b> {int(score * 100)}%\n\n{explanation}"
                        await ctx.bot.send_message(
                            chat_id=user.chat_id,
                            text=score_message,
                            parse_mode="HTML"
                        )
                        
                        # Notify admins for auto-delivery (forward)
                        username = f"@{user.username}" if getattr(user, "username", None) else f"user_id {user.user_id}"
                        link = await _build_telegram_post_link(ctx, post.channel_chat_id, post.channel_msg_id)
                        note = (
                            f"📨 پست شغلی به {username} ارسال شد (اتوماتیک)\n"
                            + f"📊 امتیاز: {int(score * 100)}%\n"
                            + (f"🔗 لینک: {link}" if link else f"🆔 شناسه: #{post.channel_msg_id}")
                        )
                        await _notify_admins(ctx, note)
                    else:
                        # Fallback to sending raw text if we can't forward
                        text = (post.text or "") or (post.caption or "")
                        if text:
                            await ctx.bot.send_message(chat_id=user.chat_id, text=text[:4000])
                            # Send score and explanation
                            score_message = f"📊 <b>امتیاز تطابق:</b> {int(score * 100)}%\n\n{explanation}"
                            await ctx.bot.send_message(
                                chat_id=user.chat_id,
                                text=score_message,
                                parse_mode="HTML"
                            )
                            username = f"@{user.username}" if getattr(user, "username", None) else f"user_id {user.user_id}"
                            link = await _build_telegram_post_link(ctx, post.channel_chat_id, post.channel_msg_id)
                            note = (
                                f"📨 متن پست شغلی به {username} ارسال شد (اتوماتیک)\n"
                                + f"📊 امتیاز: {int(score * 100)}%\n"
                                + (f"🔗 لینک: {link}" if link else f"🆔 شناسه: #{post.channel_msg_id}")
                            )
                            await _notify_admins(ctx, note)

                    # Record delivery
                    db.add(Delivery(user_id=user.user_id, channel_msg_id=post.channel_msg_id))
                    db.commit()
            except Exception as e:
                log.warning(f"Failed to notify user {user.user_id} for post {post.channel_msg_id}: {e}")

async def on_channel_post(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle new posts published in channels where the bot is an admin."""
    msg = update.channel_post
    if not msg:
        return
    text = msg.text or msg.caption
    if not text:
        return

    with SessionLocal() as db:
        # Upsert channel post
        existing = db.query(ChannelPost).filter(
            ChannelPost.channel_msg_id == msg.message_id
        ).first()
        if not existing:
            post = ChannelPost(
                channel_msg_id=msg.message_id,
                channel_chat_id=msg.chat.id if msg.chat else None,
                text=msg.text or "",
                caption=msg.caption or "",
                posted_at=msg.date,
            )
            db.add(post)
            db.commit()
        else:
            post = existing
            changed = False
            if getattr(post, "channel_chat_id", None) is None and msg.chat:
                post.channel_chat_id = msg.chat.id
                changed = True
            if (not getattr(post, "text", None)) and msg.text:
                post.text = msg.text
                changed = True
            if (not getattr(post, "caption", None)) and msg.caption:
                post.caption = msg.caption
                changed = True
            if changed:
                db.add(post)
                db.commit()

        # Classify
        try:
            classification = job_classifier.classify_job(text)
            post.employment_type = classification.employment_type
            post.position = classification.position
            post.industry = classification.industry
            post.seniority_level = classification.seniority_level
            try:
                post.years_experience = int(classification.years_experience) if classification.years_experience is not None else None
            except Exception:
                post.years_experience = None
            post.work_location = classification.work_location
            post.skills_technologies = classification.skills_technologies
            post.bonuses = classification.bonuses
            post.health_insurance = classification.health_insurance
            post.stock_options = classification.stock_options
            post.work_schedule = classification.work_schedule
            post.company_size = classification.company_size
            post.company_name = classification.company_name
            post.is_classified = True
            post.classification_data = classification.model_dump()
            db.add(post)
            db.commit()
        except Exception as e:
            log.error(f"Classification failed for channel post {msg.message_id}: {e}")

    # Forward to admins and send formatted classification summary
    for admin_id in settings.ADMIN_IDS:
        try:
            fwd = await ctx.bot.forward_message(
                chat_id=admin_id,
                from_chat_id=msg.chat.id,
                message_id=msg.message_id,
            )
            formatted = _format_classification_message(
                post.employment_type,
                post.position,
                post.industry,
                post.seniority_level,
                post.years_experience,
                post.work_location,
                post.skills_technologies,
                post.bonuses,
                post.health_insurance,
                post.stock_options,
                post.work_schedule,
                post.company_size,
            )
            await ctx.bot.send_message(chat_id=admin_id, text=formatted, parse_mode="HTML", reply_to_message_id=fwd.message_id)
        except Exception as e:
            log.warning(f"Failed to forward to admin {admin_id}: {e}")

    # Notify matched users (don't block admin notifications on failures)
    try:
        await _notify_matched_users_for_post(ctx, post)
    except Exception as e:
        log.warning(f"Failed notifying users for post {post.channel_msg_id}: {e}")

async def activate_new_positions(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    with SessionLocal() as db:
        preferred = db.query(PreferredJobPosition).filter(PreferredJobPosition.user_id == user_id).first()
        if not preferred:
            preferred = PreferredJobPosition(user_id=user_id, active=True)
            db.add(preferred)
        else:
            preferred.active = True
            db.add(preferred)
        db.commit()
    await update.message.reply_text("✅ پست‌های شغلی مطابق جدید را دریافت خواهید کرد.")


async def deactivate_new_positions(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    with SessionLocal() as db:
        preferred = db.query(PreferredJobPosition).filter(PreferredJobPosition.user_id == user_id).first()
        if not preferred:
            preferred = PreferredJobPosition(user_id=user_id, active=False)
            db.add(preferred)
        else:
            preferred.active = False
            db.add(preferred)
        db.commit()
    await update.message.reply_text("🚫 پست‌های شغلی مطابق جدید را دریافت نخواهید کرد. از /activate_new_positions برای فعال‌سازی مجدد استفاده کنید.")
