from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler, ConversationHandler
from datetime import datetime, timedelta
import logging
import asyncio

log = logging.getLogger(__name__)

from app.matcher import keyword_extractor, PreferenceMatcher
from app.db import SessionLocal
from app.models import User, Preference, Delivery, ChannelPost
from app.config import settings

try:
    from app.classification import job_classifier
    log.info("Using OpenAI-based job classifier")
except Exception as e:
    log.warning(f"OpenAI classifier not available: {e}")
    from app.classification_mock import mock_job_classifier as job_classifier
    log.info("Using mock job classifier")

matcher = PreferenceMatcher(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", threshold=0.62)

from telegram.constants import ChatType

# Conversation states
EXTRACTING_KEYWORDS = 1

# Store admin keyword extraction sessions in memory
admin_keyword_sessions = {}

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
                
                # Extract keywords
                try:
                    if keyword_extractor:
                        post_keywords = keyword_extractor.extract_keywords(post_text)
                        all_keywords.update(post_keywords)
                except Exception as e:
                    log.error(f"Keyword extraction failed: {e}")
                
                # Classify the job posting
                try:
                    classification = job_classifier.classify_job(post_text)
                    
                    # Update the existing post with classification data
                    existing_post = db.query(ChannelPost).filter(
                        ChannelPost.channel_msg_id == post_data['msg_id']
                    ).first()
                    
                    if existing_post:
                        # Store individual fields for indexing and querying
                        existing_post.employment_type = classification.employment_type
                        existing_post.job_function = classification.job_function
                        existing_post.industry = classification.industry
                        existing_post.seniority_level = classification.seniority_level
                        existing_post.work_location = classification.work_location
                        existing_post.job_specialization = classification.job_specialization
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
                'msg_id': msg.message_id,
                'date': msg.date
            })
            # Store the post in database
            with SessionLocal() as db:
                # Check if post already exists
                existing = db.query(ChannelPost).filter(
                    ChannelPost.channel_msg_id == msg.message_id
                ).first()
                if not existing:
                    post = ChannelPost(
                        channel_msg_id=msg.message_id,
                        text=text or "",
                        caption=msg.caption or "",
                        posted_at=msg.date
                    )
                    db.add(post)
                    db.commit()
            # Optionally, acknowledge receipt
            await update.message.reply_text("Message received for keyword extraction.")
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
    
    user_query = update.message.text.strip()
    user_id = update.message.from_user.id
    
    if not user_query:
        return
    
    try:
        # Get recent channel posts (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        
        with SessionLocal() as db:
            recent_posts = db.query(ChannelPost).filter(
                ChannelPost.posted_at >= thirty_days_ago,
                ChannelPost.is_classified == True  # Only search classified posts
            ).order_by(ChannelPost.posted_at.desc()).limit(100).all()
        
        if not recent_posts:
            await update.message.reply_text(
                "No recent job posts found in the database. "
                "Admins need to forward channel posts first using /extract_keywords."
            )
            return
        
        # Prepare posts for matching
        posts_text = []
        posts_data = []
        for post in recent_posts:
            text = post.text or post.caption or ""
            if text:
                posts_text.append(text)
                posts_data.append({
                    'id': post.channel_msg_id,
                    'text': text,
                    'date': post.posted_at
                })
        
        if not posts_text:
            await update.message.reply_text("No text content found in recent posts.")
            return
        
        # Use the matcher to find relevant posts
        print(posts_data)
        matcher.rebuild_index(posts_text, [p['id'] for p in posts_data])
        matched_posts = matcher.search_users_by_embedding(user_query)
        
        if not matched_posts:
            await update.message.reply_text(
                f"No matching job posts found for: '{user_query}'\n\n"
                f"Try different keywords or be more specific about your requirements."
            )
            return
        
        # Sort by similarity score (descending)
        matched_posts.sort(key=lambda x: x[1], reverse=True)
        
        # Send top matches (limit to 5)
        top_matches = matched_posts[:5]
        
        response = f"Found {len(top_matches)} matching job post(s) for: '{user_query}'\n\n"
        
        for i, (msg_id, similarity) in enumerate(top_matches, 1):
            # Find the post data and classification
            post_data = next((p for p in posts_data if p['id'] == msg_id), None)
            post_obj = next((p for p in recent_posts if p.channel_msg_id == msg_id), None)
            
            if post_data and post_obj:
                response += f"**Match {i}** (Similarity: {similarity:.2f})\n"
                response += f"Posted: {post_data['date'].strftime('%Y-%m-%d %H:%M')}\n"
                
                # Add classification details if available
                if post_obj.job_function:
                    response += f"Role: {post_obj.job_function}\n"
                if post_obj.seniority_level:
                    response += f"Level: {post_obj.seniority_level}\n"
                if post_obj.work_location:
                    response += f"Location: {post_obj.work_location}\n"
                if post_obj.company_name:
                    response += f"Company: {post_obj.company_name}\n"
                if post_obj.skills_technologies:
                    response += f"Skills: {post_obj.skills_technologies[:100]}{'...' if len(post_obj.skills_technologies) > 100 else ''}\n"
                
                response += f"Content: {post_data['text'][:150]}{'...' if len(post_data['text']) > 150 else ''}\n\n"
        
        # Split long messages if needed
        if len(response) > 4000:
            response = response[:4000] + "\n\n... (truncated)"
        
        await update.message.reply_text(response)
        
        # Also save the user's query as a preference for future matching
        with SessionLocal() as db:
            pref = Preference(user_id=user_id, text=user_query)
            db.add(pref)
            db.commit()
            
    except Exception as e:
        log.error(f"Error in job search: {e}")
        await update.message.reply_text("Sorry, there was an error searching for jobs. Please try again.")


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
    application = ApplicationBuilder().token(settings.TELEGRAM_BOT_TOKEN).build()
    
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
    application.add_handler(keyword_extraction_conv)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, query_handler))
    application.add_handler(CallbackQueryHandler(select_keyword))
    return application
