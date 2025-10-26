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

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            base_url=settings.LLM_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            timeout=60,
            max_retries=2,
        )
        self.parser = PydanticOutputParser(pydantic_object=JobPositionClassification)
        
        self.prompt = PromptTemplate(
            template="""
You are an expert job classification system. Analyze the following job posting text and extract structured information.

Job Posting Text:
{job_text}

{format_instructions}

Please classify this job posting based on the text content. If information is not explicitly mentioned or unclear, set the field to null.
Focus on extracting:
- Employment type (Full-time, Part-time, Internship, Freelance)
- Position (Backend Developer, Frontend Developer, Full-stack Developer, DevOps Engineer, Software Engineer, Data Scientist, Machine Learning Engineer, AI Specialist, Data Engineer, Cloud Engineer, Quality Assurance (QA) Engineer, Security Engineer)
- Industry (Technology, Software, IT, Fintech, Cybersecurity)
- Seniority level (Entry-level, Mid-level, Senior, Lead, Manager, Director, VP)
- Minimum years of experience if a numeric value is explicitly mentioned (e.g., "3+ years", "2 years", "3-5 years"). If a range is provided, return the minimum. If not present, set to null. Provide only an integer like 3.
- Work location (On-site, Remote, Hybrid)
- Skills/technologies mentioned (Python, JavaScript, Java, C++, Go, Ruby, PHP, Rust, SQL, NoSQL, React, Angular, Node.js, Django, Flask, Spring Boot, Vue.js, TensorFlow, PyTorch, Kubernetes, Docker, AWS, Azure, Google Cloud, Machine Learning, Data Analysis)
- Benefits (bonuses, health_insurance, stock_options)
- Work schedule (Flexible Hours, Fixed Hours, Shift work)
- Company size (Startup, SME, Large Corporation)
- Company name if mentioned

Return the classification in the specified JSON format.
""",
            input_variables=["job_text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        self.chain = self.prompt | self.llm | self.parser

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
        log.info(f"Starting job classification for text length: {len(job_text) if job_text else 0}")
        
        if not job_text or not job_text.strip():
            log.warning("Empty job text provided, returning empty classification")
            return JobPositionClassification()
        
        log.info(f"Job text preview: {job_text[:200]}...")
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                # Log the request details
                log.info(f"Making LLM request to {settings.LLM_BASE_URL} (attempt {attempt + 1}/{max_retries})")
                log.info(f"Using model: {self.llm.model_name}")
                
                result = self.chain.invoke({"job_text": job_text})
                
                processing_time = time.time() - start_time
                log.info(f"✅ Successfully classified job posting in {processing_time:.2f}s")
                log.info(f"Classification result: {result.model_dump()}")
                
                return result
                
            except Exception as e:
                processing_time = time.time() - start_time if 'start_time' in locals() else 0
                log.error(f"❌ Error classifying job posting (attempt {attempt + 1}/{max_retries}) after {processing_time:.2f}s: {e!r}")
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
                    log.error(f"❌ All {max_retries} attempts failed, returning empty classification")
                    return JobPositionClassification()
        
        # This should never be reached, but just in case
        return JobPositionClassification()

# Create a global instance
job_classifier = JobClassifier(model_name=settings.LLM_MODEL_NAME)


