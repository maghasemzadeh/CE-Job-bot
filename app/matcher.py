from sentence_transformers import SentenceTransformer
from typing import List
import re
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import openai  # Correct import for OpenAI API
from sklearn.metrics.pairwise import cosine_similarity
import os
from app.config import import_config

def norm(text: str) -> str:
    t = text.lower()
    t = t.replace("ي", "ی").replace("ك", "ک")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


class KeywordExtractor:
    def __init__(self, model_name: str):
        # Set the OpenAI API key directly from environment variable
        openai.api_key = import_config("OPENAI_API_KEY")  # This assumes you have the key set in your environment
        print(f"OPENAI_API_KEY: {openai.api_key}")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set!")

        self.llm = ChatOpenAI()  # Assuming this is your desired model

    def extract_keywords(self, text: str) -> List[str]:
        prompt = (f"Extract relevant keywords for a job description from the following text:"
                  f"{text})"
                  f"Keywords:")
        response = self.llm.responses.create(
            model="gpt-5-nano",  # You can update the model name if needed
            input=prompt
        ).output_text
        print("Raw LLM response:", repr(response))  # Debug print

        # Try to extract keywords from various possible formats
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
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

        print("Extracted keywords:", normed_keywords)  # Debug print
        return normed_keywords


class PreferenceMatcher:
    def __init__(self, model_name: str, threshold: float):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.messages = []
        self.message_ids = []
        self.users = []
        self.preferences = {}

    def rebuild_index(self, posts: List[str], ids: List[int]):
        self.messages = [norm(p) for p in posts]
        self.message_ids = ids
        embeddings = self.model.encode(self.messages)
        self.embeddings = embeddings

    def add_preference(self, user_id: int, preference_text: str):
        self.preferences.setdefault(user_id, []).append(norm(preference_text))

    def search_users_by_keyword(self, user_query: str) -> List[int]:
        matched_users = []
        for user_id, prefs in self.preferences.items():
            for pref in prefs:
                if pref in user_query:
                    matched_users.append(user_id)
        return matched_users

    def search_users_by_embedding(self, user_query: str) -> List[int]:
        query_embedding = self.model.encode([norm(user_query)])
        print(user_query)
        similarities = cosine_similarity(query_embedding, self.embeddings)
        print(similarities)
        matched_posts = [
            (self.message_ids[i], similarities[0][i])
            for i in range(len(self.messages)) if similarities[0][i] >= self.threshold
        ]

        return matched_posts


# Create instances for easy importing
# Only create instance if OpenAI API key is available
if import_config("OPENAI_API_KEY"):
    keyword_extractor = KeywordExtractor("gpt-5")
else:
    # Create a dummy instance for testing
    keyword_extractor = None
