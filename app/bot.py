from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler, ConversationHandler
from datetime import datetime, timedelta
import logging
import asyncio
import os
from pathlib import Path

log = logging.getLogger(__name__)

from app.matcher import keyword_extractor, PreferenceMatcher
from app.db import SessionLocal
from app.models import User, Preference, Delivery, ChannelPost, PreferredJobPosition
from app.config import settings

from app.classification import job_classifier
log.info("Using OpenAI-based job classifier")

matcher = PreferenceMatcher(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", threshold=0.62)

from telegram.constants import ChatType

# LangChain document loaders for PDFs
from langchain_community.document_loaders import PyPDFLoader
from sqlalchemy import or_

# Conversation states
EXTRACTING_KEYWORDS = 1
AWAITING_POSITION_UPDATE = 2

# Store admin keyword extraction sessions in memory
admin_keyword_sessions = {}

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
            return "yes"
        if v is False:
            return "no"
        return "don't care"
    def fmt_val(v) -> str:
        if v is None:
            return "don't care"
        s = str(v).strip()
        return s if s else "don't care"
    lines = [
        "<b>📌 Classified job</b>",
        f"• 🧑‍💼 <b>Employment type</b>: {fmt_val(employment_type)}",
        f"• 💼 <b>Position</b>: {fmt_val(position)}",
        f"• 🏭 <b>Industry</b>: {fmt_val(industry)}",
        f"• 📈 <b>Seniority level</b>: {fmt_val(seniority_level)}",
        f"• ⌛️ <b>Years experience</b>: {fmt_val(years_experience)}",
        f"• 📍 <b>Work location</b>: {fmt_val(work_location)}",
        f"• 🛠️ <b>Skills/technologies</b>: {fmt_val(skills_technologies)}",
        f"• 💰 <b>Bonuses</b>: {fmt_bool(bonuses)}",
        f"• 🏥 <b>Health insurance</b>: {fmt_bool(health_insurance)}",
        f"• 📊 <b>Stock options</b>: {fmt_bool(stock_options)}",
        f"• 🗓️ <b>Work schedule</b>: {fmt_val(work_schedule)}",
        f"• 🏢 <b>Company size</b>: {fmt_val(company_size)}",
    ]
    return "\n".join(lines)

async def extract_keywords(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in settings.ADMIN_IDS:
        await update.message.reply_text("This command is only for admins.")
        return ConversationHandler.END

    # Start a new session for this admin
    admin_keyword_sessions[user_id] = []
    await update.message.reply_text(
        "Forward all channel job posts you want to extract keywords from. "
        "When done, send /extract_keywords_end."
    )
    return EXTRACTING_KEYWORDS

async def extract_keywords_end(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in settings.ADMIN_IDS:
        await update.message.reply_text("This command is only for admins.")
        return ConversationHandler.END

    posts = admin_keyword_sessions.pop(user_id, [])
    if not posts:
        await update.message.reply_text("No forwarded messages received. Please forward channel posts first.")
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
    summary = f"✅ Processing completed!\n\n"
    summary += f"📊 Classified {classified_posts} job posts\n"
    summary += f"🔑 Extracted {len(all_keywords)} keywords\n\n"
    
    if all_keywords:
        summary += f"Keywords: {', '.join(sorted(all_keywords))}"
    
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
                    await update.message.reply_text("Failed to classify this message.")
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
    await query.edit_message_text(text=f"You've selected: {selected_keyword}. I'll forward related job posts to you.")

async def query_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    # If we're waiting for text after /update_position, route to that handler
    if ctx.user_data.get("awaiting_position_update"):
        await handle_position_text_if_waiting(update, ctx)
        return
    # Otherwise, guide the user to commands
    help_hint = (
        "Please use commands to interact with the bot:\n"
        "• /update_position – set or update your preferred job position\n"
        "• /my_position – view your saved preferences\n"
        "• /help – see all commands and how to use them"
    )
    await update.message.reply_text(help_hint)


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome to the job bot! Send me job preferences or ask for job posts.")

async def stop(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("You will no longer receive job updates. To restart, use /start.")

async def fetch_channel_posts(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Admin command to fetch recent posts from the channel and store them"""
    user_id = update.effective_user.id
    if user_id not in settings.ADMIN_IDS:
        await update.message.reply_text("This command is only for admins.")
        return

    try:
        # Get recent messages from the channel (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        
        # Note: This is a simplified approach. In practice, you'd need to:
        # 1. Use Telegram Client API (not Bot API) to get channel history
        # 2. Or have a separate service that monitors the channel
        # 3. Or manually forward posts to the bot
        
        await update.message.reply_text(
            "Channel post fetching requires Telegram Client API access. "
            "For now, please forward channel posts manually using /extract_keywords command."
        )
        
    except Exception as e:
        log.error(f"Error fetching channel posts: {e}")
        await update.message.reply_text(f"Error fetching channel posts: {str(e)}")

def build_app():
    async def post_init(app):
        try:
            await app.bot.set_my_commands([
                BotCommand("start", "Start the bot"),
                BotCommand("help", "Show help and available commands"),
                BotCommand("update_position", "Set/update your preferred job position"),
                BotCommand("my_position", "Show your saved preferred position"),
                BotCommand("search_recent_positions", "Find and forward recent matching jobs"),
                BotCommand("activate_new_jobs", "Enable receiving new matching jobs"),
                BotCommand("deactive_new_jobs", "Disable receiving new matching jobs"),
                BotCommand("stop", "Stop receiving updates"),
            ])
        except Exception as e:
            log.warning(f"Failed to set bot commands: {e}")

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
    application.add_handler(CommandHandler("update_position", update_position))
    application.add_handler(CommandHandler("my_position", my_position))
    application.add_handler(CommandHandler("search_recent_positions", search_recent_positions))
    application.add_handler(CommandHandler("activate_new_jobs", activate_new_jobs))
    application.add_handler(CommandHandler("deactive_new_jobs", deactive_new_jobs))
    application.add_handler(CallbackQueryHandler(select_keyword))
    # Handle channel posts (bot must be admin in the channel) BEFORE generic text handlers
    application.add_handler(MessageHandler(filters.ChatType.CHANNEL, on_channel_post))
    # Private chat-only handlers
    application.add_handler(MessageHandler(filters.ChatType.PRIVATE & filters.Document.PDF, handle_position_document))
    application.add_handler(MessageHandler(filters.ChatType.PRIVATE & (filters.TEXT & ~filters.COMMAND), handle_position_text_if_waiting))
    application.add_handler(MessageHandler(filters.ChatType.PRIVATE & (filters.TEXT & ~filters.COMMAND), query_handler))
    return application


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


async def update_position(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    await update.message.reply_text(
        "Send your preferred position details as text, or upload a PDF resume."
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
            return "yes"
        if v is False:
            return "no"
        return "don't care"

    def fmt_val(v) -> str:
        if v is None:
            return "don't care"
        s = str(v).strip()
        return s if s else "don't care"

    lines = [
        "<b>🎯 Your preferred position</b>",
        f"• 🧑‍💼 <b>Employment type</b>: {fmt_val(preferred.employment_type)}",
        f"• 💼 <b>Position</b>: {fmt_val(preferred.position)}",
        f"• 🏭 <b>Industry</b>: {fmt_val(preferred.industry)}",
        f"• 📈 <b>Seniority level</b>: {fmt_val(preferred.seniority_level)}",
        f"• ⌛️ <b>Years experience</b>: {fmt_val(preferred.years_experience)}",
        f"• 📍 <b>Work location</b>: {fmt_val(preferred.work_location)}",
        f"• 🛠️ <b>Skills/technologies</b>: {fmt_val(preferred.skills_technologies)}",
        f"• 💰 <b>Bonuses</b>: {fmt_bool(preferred.bonuses)}",
        f"• 🏥 <b>Health insurance</b>: {fmt_bool(preferred.health_insurance)}",
        f"• 📊 <b>Stock options</b>: {fmt_bool(preferred.stock_options)}",
        f"• 🗓️ <b>Work schedule</b>: {fmt_val(preferred.work_schedule)}",
        f"• 🏢 <b>Company size</b>: {fmt_val(preferred.company_size)}",
    ]
    return "\n".join(lines)


async def handle_position_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not (getattr(ctx, "user_data", None) and ctx.user_data.get("awaiting_position_update")):
        return
    await ensure_user_record(update)
    user_id = update.effective_user.id
    document = update.message.document
    if not document or document.mime_type != "application/pdf":
        await update.message.reply_text("Please upload a PDF document.")
        return
    # Download PDF to a temp folder
    base_dir = Path(os.getenv("RESUME_DIR", "./resumes")).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    file_path = base_dir / f"resume_{user_id}_{document.file_unique_id}.pdf"
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
        "Your preferred position has been updated from your resume."
    )
    await update.message.reply_text(_format_preferred_position_message(preferred), parse_mode="HTML")


async def handle_position_text_if_waiting(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not (getattr(ctx, "user_data", None) and ctx.user_data.get("awaiting_position_update")):
        return
    await ensure_user_record(update)
    user_id = update.effective_user.id
    text = update.message.text or ""
    with SessionLocal() as db:
        preferred = _classify_and_save_preference(db, user_id=user_id, text=text)
    ctx.user_data["awaiting_position_update"] = False
    await update.message.reply_text("Your preferred position has been updated.")
    await update.message.reply_text(_format_preferred_position_message(preferred), parse_mode="HTML")


async def my_position(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    with SessionLocal() as db:
        preferred = db.query(PreferredJobPosition).filter(PreferredJobPosition.user_id == user_id).first()
        if not preferred:
            await update.message.reply_text("No preferred position is set yet. Use /update_position to provide details.")
            return
        await update.message.reply_text(_format_preferred_position_message(preferred), parse_mode="HTML")


async def help_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = (
        "<b>🤖 How to use CE Job Bot</b>\n\n"
        "• /start – start the bot\n"
        "• /help – show this help\n"
        "• /update_position – send text or upload a PDF resume to set/update your preferences\n"
        "• /my_position – view your saved preferred position\n"
        "• /search_recent_positions – find and forward recent matching jobs based on your preferences\n"
        "• /activate_new_jobs – enable receiving new matching jobs\n"
        "• /deactive_new_jobs – disable receiving new matching jobs\n"
        "• /stop – stop receiving updates\n\n"
        "Tip: After /update_position, your next message (text or PDF) will be used to update your preferences."
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
        return 999
    key = value.strip().lower()
    return order.get(key, 999)


async def search_recent_positions(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await update.message.reply_text("Trying to find job position realted to your preferences...")
    with SessionLocal() as db:
        preferred = db.query(PreferredJobPosition).filter(PreferredJobPosition.user_id == user_id).first()
        if not preferred:
            await update.message.reply_text("You don't have a preferred position yet. Use /update_position first.")
            return

        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        posts = db.query(ChannelPost).filter(
            ChannelPost.posted_at >= thirty_days_ago,
            ChannelPost.is_classified == True
        ).order_by(ChannelPost.posted_at.desc()).limit(200).all()

        matches = []
        for p in posts:
            print(f"trying to match post {p.channel_msg_id}")
            def eq(a, b):
                if a is None or str(a).strip() == "":
                    return True
                return (str(a).strip().lower() == str(b or "").strip().lower())

            if not eq(preferred.employment_type, p.employment_type):
                print(f"employment_type mismatch: {preferred.employment_type} != {p.employment_type}")      
                continue
            if not eq(preferred.position, p.position):
                print(f"position mismatch: {preferred.position} != {p.position}")
                continue
            if not eq(preferred.work_schedule, p.work_schedule):
                print(f"work_schedule mismatch: {preferred.work_schedule} != {p.work_schedule}")
                continue

            user_rank = _seniority_rank(preferred.seniority_level)
            post_rank = _seniority_rank(p.seniority_level)
            if post_rank > user_rank:
                print(f"seniority_level mismatch: {preferred.seniority_level} != {p.seniority_level}")
                continue

            overlap = _skills_overlap_ratio(preferred.skills_technologies, p.skills_technologies)
            if overlap < 0.5:
                print(f"skills_technologies mismatch: {preferred.skills_technologies} != {p.skills_technologies}")
                continue

            matches.append(p)

    if not matches:
        await update.message.reply_text("No matching positions found in recent posts.")
        return

    sent = 0
    for p in matches[:10]:
        try:
            if p.channel_chat_id:
                await ctx.bot.forward_message(
                    chat_id=update.effective_chat.id,
                    from_chat_id=p.channel_chat_id,
                    message_id=p.channel_msg_id,
                )
                sent += 1
            else:
                # Fallback to sending raw text if we can't forward
                text = (p.text or "") or (p.caption or "")
                if text:
                    await update.message.reply_text(text[:4000])
                    sent += 1
        except Exception as e:
            log.warning(f"Failed to forward message {p.channel_msg_id}: {e}")
            try:
                text = (p.text or "") or (p.caption or "")
                if text:
                    await update.message.reply_text(text[:4000])
                    sent += 1
            except Exception as e2:
                log.warning(f"Also failed to send text fallback: {e2}")

    await update.message.reply_text(f"Forwarded {sent} matching post(s).")


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
            def eq(a, b):
                if a is None or str(a).strip() == "":
                    return True
                return (str(a).strip().lower() == str(b or "").strip().lower())

            # Basic matching similar to /search_recent_positions
            if not eq(preferred.employment_type, post.employment_type):
                continue
            if not eq(preferred.position, post.position):
                continue
            if not eq(preferred.work_schedule, post.work_schedule):
                continue

            user_rank = _seniority_rank(preferred.seniority_level)
            post_rank = _seniority_rank(post.seniority_level)
            if post_rank > user_rank:
                continue

            overlap = _skills_overlap_ratio(preferred.skills_technologies, post.skills_technologies)
            if overlap < 0.5:
                continue

            try:
                if user.chat_id:
                    if post.channel_chat_id:
                        await ctx.bot.forward_message(
                            chat_id=user.chat_id,
                            from_chat_id=post.channel_chat_id,
                            message_id=post.channel_msg_id,
                        )
                    else:
                        # Fallback to sending raw text if we can't forward
                        text = (post.text or "") or (post.caption or "")
                        if text:
                            await ctx.bot.send_message(chat_id=user.chat_id, text=text[:4000])

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


async def activate_new_jobs(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
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
    await update.message.reply_text("✅ You will receive new matching job posts.")


async def deactive_new_jobs(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
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
    await update.message.reply_text("🚫 You will not receive new matching job posts. Use /activate_new_jobs to re-enable.")
