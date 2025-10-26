from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import re
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging
import time
import requests
import numpy as np
from app.config import settings

log = logging.getLogger(__name__)

def norm(text: str) -> str:
    t = text.lower()
    t = t.replace("ي", "ی").replace("ك", "ک")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


class KeywordExtractor:
    def __init__(self, model_name: str):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured")

        log.info(f"Initializing KeywordExtractor with model: {model_name}")
        log.info(f"Using LLM base URL: {settings.LLM_BASE_URL}")
        
        # Test API connectivity
        self._test_api_connectivity()

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            base_url=settings.LLM_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
        )

        self.prompt = PromptTemplate(
            input_variables=["job_description"],
            template=(
                "Extract relevant keywords for a job description from the following text. "
                "Return a concise, comma-separated list of keywords only.\n\n"
                "{job_description}"
            ),
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def _test_api_connectivity(self):
        """Test API connectivity before making actual requests"""
        try:
            log.info("Testing API connectivity for KeywordExtractor...")
            start_time = time.time()
            
            # Test basic connectivity to the API endpoint
            test_url = f"{settings.LLM_BASE_URL.rstrip('/')}/models"
            headers = {
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(test_url, headers=headers, timeout=10)
            response_time = time.time() - start_time
            
            log.info(f"KeywordExtractor API connectivity test - Status: {response.status_code}, Time: {response_time:.2f}s")
            
            if response.status_code == 200:
                log.info("✅ KeywordExtractor API connectivity test passed")
            else:
                log.warning(f"⚠️ KeywordExtractor API returned status {response.status_code}: {response.text[:200]}")
                
        except requests.exceptions.ConnectionError as e:
            log.error(f"❌ KeywordExtractor API connection failed: {e}")
        except requests.exceptions.Timeout as e:
            log.error(f"❌ KeywordExtractor API request timeout: {e}")
        except Exception as e:
            log.error(f"❌ KeywordExtractor API connectivity test failed: {e}")

    def extract_keywords(self, text: str, max_retries: int = 3) -> List[str]:
        """Extracts keywords from job description text with comprehensive logging and retry mechanism"""
        log.info(f"Starting keyword extraction for text length: {len(text) if text else 0}")
        
        if not text or not text.strip():
            log.warning("Empty text provided for keyword extraction")
            return []
        
        log.info(f"Text preview: {text[:200]}...")
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                # Log the request details
                log.info(f"Making LLM request to {settings.LLM_BASE_URL} (attempt {attempt + 1}/{max_retries})")
                log.info(f"Using model: {self.llm.model_name}")
                
                # Invoke the LLM chain to get a response
                response = self.chain.run({"job_description": text}).strip()
                
                processing_time = time.time() - start_time
                log.info(f"✅ Successfully extracted keywords in {processing_time:.2f}s")
                log.info(f"Raw LLM response: {repr(response)}")

                # Try to extract keywords from various possible formats
                lines = [line.strip() for line in response.splitlines() if line.strip()]
                keywords = []

                for line in lines:
                    # Handle comma-separated
                    if "," in line:
                        keywords.extend([k.strip() for k in line.split(",") if k.strip()])
                    # Handle numbered or bulleted lists
                    elif re.match(r"^(\d+\.|-|\*)\s*", line):
                        kw = re.sub(r"^(\d+\.|-|\*)\s*", "", line)
                        if kw:
                            keywords.append(kw)
                    else:
                        # Fallback: treat as single keyword/phrase
                        keywords.append(line)

                # Remove duplicates and normalize
                normed_keywords = []
                seen = set()
                for k in keywords:
                    nk = norm(k)
                    if nk and nk not in seen:
                        normed_keywords.append(nk)
                        seen.add(nk)

                log.info(f"Extracted keywords: {normed_keywords}")
                return normed_keywords
                
            except Exception as e:
                processing_time = time.time() - start_time if 'start_time' in locals() else 0
                log.error(f"❌ Error extracting keywords (attempt {attempt + 1}/{max_retries}) after {processing_time:.2f}s: {e!r}")
                log.error(f"Error type: {type(e).__name__}")
                log.error(f"Error details: {str(e)}")
                
                # Log additional context for debugging
                if hasattr(e, 'response'):
                    log.error(f"HTTP Response: {e.response}")
                if hasattr(e, 'request'):
                    log.error(f"Request details: {e.request}")
                
                # If this is not the last attempt, wait before retrying
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    log.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    log.error(f"❌ All {max_retries} attempts failed, returning empty keywords")
                    return []
        
        # This should never be reached, but just in case
        return []


class PreferenceMatcher:
    def __init__(self, model_name: str, threshold: float):
        log.info(f"Initializing PreferenceMatcher with model: {model_name}, threshold: {threshold}")
        
        try:
            start_time = time.time()
            self.model = SentenceTransformer(model_name)
            load_time = time.time() - start_time
            log.info(f"✅ SentenceTransformer model loaded in {load_time:.2f}s")
            log.info(f"Model device: {self.model.device}")
            log.info(f"Model max sequence length: {self.model.max_seq_length}")
        except Exception as e:
            log.error(f"❌ Failed to load SentenceTransformer model: {e}")
            raise
            
        self.threshold = threshold
        self.messages = []
        self.message_ids = []
        self.users = []
        self.preferences = {}

    def rebuild_index(self, posts: List[str], ids: List[int]):
        log.info(f"Rebuilding index with {len(posts)} posts")
        start_time = time.time()
        
        self.messages = [norm(p) for p in posts]
        self.message_ids = ids
        
        try:
            embeddings = self.model.encode(self.messages)
            self.embeddings = embeddings
            processing_time = time.time() - start_time
            log.info(f"✅ Index rebuilt successfully in {processing_time:.2f}s")
            log.info(f"Embeddings shape: {embeddings.shape}")
        except Exception as e:
            log.error(f"❌ Failed to rebuild index: {e}")
            raise

    def add_preference(self, user_id: int, preference_text: str):
        log.info(f"Adding preference for user {user_id}: {preference_text[:100]}...")
        self.preferences.setdefault(user_id, []).append(norm(preference_text))

    def search_users_by_keyword(self, user_query: str) -> List[int]:
        log.info(f"Searching users by keyword: {user_query[:100]}...")
        matched_users = []
        for user_id, prefs in self.preferences.items():
            for pref in prefs:
                if pref in user_query:
                    matched_users.append(user_id)
        log.info(f"Found {len(matched_users)} users by keyword matching")
        return matched_users

    def search_users_by_embedding(self, user_query: str) -> List[int]:
        log.info(f"Searching users by embedding: {user_query[:100]}...")
        start_time = time.time()
        
        try:
            query_embedding = self.model.encode([norm(user_query)])
            similarities = cosine_similarity(query_embedding, self.embeddings)
            
            matched_posts = [
                (self.message_ids[i], similarities[0][i])
                for i in range(len(self.messages)) if similarities[0][i] >= self.threshold
            ]
            
            processing_time = time.time() - start_time
            log.info(f"✅ Embedding search completed in {processing_time:.2f}s")
            log.info(f"Found {len(matched_posts)} matches above threshold {self.threshold}")
            log.info(f"Similarity scores: {[f'{score:.3f}' for _, score in matched_posts[:5]]}")
            
            return matched_posts
            
        except Exception as e:
            processing_time = time.time() - start_time
            log.error(f"❌ Embedding search failed after {processing_time:.2f}s: {e}")
            return []


# Create instances for easy importing
# Only create instance if OpenAI API key is available
if settings.OPENAI_API_KEY:
    keyword_extractor = KeywordExtractor("gpt-5")
    openai_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.LLM_BASE_URL
    )
else:
    # Create a dummy instance for testing
    keyword_extractor = None
    openai_embeddings = None


def get_embedding(text: str) -> np.ndarray:
    """Get OpenAI embedding for text"""
    if not text or not text.strip():
        return None
    try:
        embedding = openai_embeddings.embed_query(text)
        return np.array(embedding)
    except Exception as e:
        log.error(f"Error generating embedding: {e}")
        return None


def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings"""
    if embedding1 is None or embedding2 is None:
        return 0.0
    
    try:
        # Compute cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    except Exception as e:
        log.error(f"Error calculating cosine similarity: {e}")
        return 0.0


def match_resume_with_job_embedding(
    resume_text: str,
    job_text: str,
    threshold: float = 0.5
) -> Tuple[float, str]:
    """
    Match resume with job position using OpenAI embeddings
    
    Returns:
        Tuple of (similarity_score, explanation)
    """
    if not resume_text or not job_text:
        return 0.0, "متن رزومه یا شغل موجود نیست"
    
    try:
        # Get embeddings
        resume_embedding = get_embedding(resume_text)
        job_embedding = get_embedding(job_text)
        
        if resume_embedding is None or job_embedding is None:
            return 0.0, "خطا در تولید embedding"
        
        # Calculate similarity
        similarity = calculate_cosine_similarity(resume_embedding, job_embedding)
        
        # Generate explanation
        if similarity >= threshold:
            percentage = int(similarity * 100)
            explanation = f"میزان تطابق: {percentage}% - این موقعیت شغلی با رزومه شما همخوانی بالایی دارد."
        else:
            percentage = int(similarity * 100)
            explanation = f"میزان تطابق: {percentage}% - این موقعیت شغلی با رزومه شما همخوانی پایینی دارد."
        
        print(f"Matching resume with job: {job_text}")
        print(f"Similarity: {similarity}")
        print(f"Explanation: {explanation}")
        return similarity, explanation
        
    except Exception as e:
        log.error(f"Error matching resume with job: {e}")
        return 0.0, f"خطا در تطابق: {str(e)}"
