"""
Job position classification using LangChain and Pydantic
"""

from pydantic import BaseModel, Field
from typing import Optional
from langchain_openai import ChatOpenAI  
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
import logging
import time
import requests
from app.config import settings
from app.langfuse_client import LangfuseSingleton

log = logging.getLogger(__name__)

class JobPositionClassification(BaseModel):
    # Employment Type
    employment_type: Optional[str] = Field(None, description="Type of employment")
    
    # Job Function or Department
    position: Optional[str] = Field(None, description="Primary job function or department")
    
    # Industry
    industry: Optional[str] = Field(None, description="Industry sector")
    
    # Seniority Level
    seniority_level: Optional[str] = Field(None, description="Required experience level")
    # Numeric years of experience required if explicitly mentioned (e.g., 3-5 -> 3)
    years_experience: Optional[int] = Field(None, description="Minimum years of experience required if mentioned")
    
    # Work Location
    work_location: Optional[str] = Field(None, description="Work location type")
    
    
    # Skills/Technologies
    skills_technologies: Optional[str] = Field(None, description="Required skills and technologies")
    
    # Bonuses & Benefits
    bonuses: Optional[bool] = Field(None, description="Whether the position includes bonuses or profit-sharing")
    health_insurance: Optional[bool] = Field(None, description="Whether health insurance is provided")
    stock_options: Optional[bool] = Field(None, description="Whether stock options are provided")
    
    # Work Schedule
    work_schedule: Optional[str] = Field(None, description="Work schedule type")
    
    # Company Size
    company_size: Optional[str] = Field(None, description="Size of the company")
    
    # Company Name
    company_name: Optional[str] = Field(None, description="Name of the company offering the job position")

class JobClassifier:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured")

        log.info(f"Initializing JobClassifier with model: {model_name}")
        log.info(f"Using LLM base URL: {settings.LLM_BASE_URL}")
        log.info(f"API Key configured: {'Yes' if settings.OPENAI_API_KEY else 'No'}")

        # Test API connectivity before initializing
        self._test_api_connectivity()

        

    def _test_api_connectivity(self):
        """Test API connectivity before making actual requests"""
        try:
            log.info("Testing API connectivity...")
            start_time = time.time()
            
            # Test basic connectivity to the API endpoint
            test_url = f"{settings.LLM_BASE_URL.rstrip('/')}/models"
            headers = {
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(test_url, headers=headers, timeout=10)
            response_time = time.time() - start_time
            
            log.info(f"API connectivity test - Status: {response.status_code}, Time: {response_time:.2f}s")
            
            if response.status_code == 200:
                log.info("✅ API connectivity test passed")
            else:
                log.warning(f"⚠️ API returned status {response.status_code}: {response.text[:200]}")
                
        except requests.exceptions.ConnectionError as e:
            log.error(f"❌ API connection failed: {e}")
        except requests.exceptions.Timeout as e:
            log.error(f"❌ API request timeout: {e}")
        except Exception as e:
            log.error(f"❌ API connectivity test failed: {e}")

    def classify_job(self, job_text: str, max_retries: int = 3) -> JobPositionClassification:
        """Classify a job posting text with comprehensive logging and retry mechanism"""
        langfuse = LangfuseSingleton()
        parser = PydanticOutputParser(pydantic_object=JobPositionClassification)
        # Use a named prompt from Langfuse
        job = langfuse.ask("job-classification", 
            parser=parser, 
            job_text=job_text,
            format_instructions=parser.get_format_instructions()
        )
        return job

# Create a global instance
job_classifier = JobClassifier(model_name=settings.LLM_MODEL_NAME)


