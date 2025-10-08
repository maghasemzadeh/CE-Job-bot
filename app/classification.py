"""
Job position classification using LangChain and Pydantic
"""

from pydantic import BaseModel, Field
from typing import Optional
from langchain_openai import ChatOpenAI  
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
import logging
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

    def classify_job(self, job_text: str) -> JobPositionClassification:
        """Classify a job posting text"""
        try:
            if not job_text or not job_text.strip():
                return JobPositionClassification()
            
            result = self.chain.invoke({"job_text": job_text})
            log.info(f"Successfully classified job posting")
            return result
            
        except Exception as e:
            log.error(f"Error classifying job posting: {e!r}")
            return JobPositionClassification()

# Create a global instance
job_classifier = JobClassifier(model_name=settings.LLM_MODEL_NAME)


