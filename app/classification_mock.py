"""
Mock job classification system for testing when OpenAI API is not available
"""

from pydantic import BaseModel, Field
from typing import Optional
import logging

log = logging.getLogger(__name__)

class JobPositionClassification(BaseModel):
    # Employment Type
    employment_type: Optional[str] = Field(None, description="Type of employment")
    
    # Job Function or Department
    job_function: Optional[str] = Field(None, description="Primary job function or department")
    
    # Industry
    industry: Optional[str] = Field(None, description="Industry sector")
    
    # Seniority Level
    seniority_level: Optional[str] = Field(None, description="Required experience level")
    
    # Work Location
    work_location: Optional[str] = Field(None, description="Work location type")
    
    # Job Specialization
    job_specialization: Optional[str] = Field(None, description="Specific job specialization")
    
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

class MockJobClassifier:
    def __init__(self):
        log.info("Using Mock Job Classifier (OpenAI API not available)")
    
    def classify_job(self, job_text: str) -> JobPositionClassification:
        """Mock classification using simple keyword matching"""
        if not job_text or not job_text.strip():
            return JobPositionClassification()
        
        text_lower = job_text.lower()
        
        # Simple keyword-based classification
        classification = JobPositionClassification()
        
        # Employment type
        if any(word in text_lower for word in ['full-time', 'full time', 'fulltime']):
            classification.employment_type = "Full-time"
        elif any(word in text_lower for word in ['part-time', 'part time', 'parttime']):
            classification.employment_type = "Part-time"
        elif any(word in text_lower for word in ['contract', 'freelance']):
            classification.employment_type = "Contract"
        elif any(word in text_lower for word in ['internship', 'intern']):
            classification.employment_type = "Internship"
        
        # Job function
        if any(word in text_lower for word in ['backend', 'server', 'api']):
            classification.job_function = "Backend Developer"
        elif any(word in text_lower for word in ['frontend', 'front-end', 'ui', 'ux']):
            classification.job_function = "Frontend Developer"
        elif any(word in text_lower for word in ['full-stack', 'fullstack', 'full stack']):
            classification.job_function = "Full-stack Developer"
        elif any(word in text_lower for word in ['devops', 'dev-ops', 'deployment']):
            classification.job_function = "DevOps Engineer"
        elif any(word in text_lower for word in ['data scientist', 'data science', 'ml', 'machine learning']):
            classification.job_function = "Data Scientist"
        elif any(word in text_lower for word in ['qa', 'quality assurance', 'testing', 'test']):
            classification.job_function = "Quality Assurance (QA) Engineer"
        else:
            classification.job_function = "Software Engineer"
        
        # Industry
        if any(word in text_lower for word in ['fintech', 'finance', 'banking']):
            classification.industry = "Fintech"
        elif any(word in text_lower for word in ['cybersecurity', 'security', 'cyber']):
            classification.industry = "Cybersecurity"
        else:
            classification.industry = "Technology"
        
        # Seniority level
        if any(word in text_lower for word in ['senior', 'lead', 'principal', 'architect']):
            classification.seniority_level = "Senior"
        elif any(word in text_lower for word in ['junior', 'entry', 'graduate', 'trainee']):
            classification.seniority_level = "Entry-level"
        elif any(word in text_lower for word in ['manager', 'director', 'vp', 'vice president']):
            classification.seniority_level = "Manager"
        else:
            classification.seniority_level = "Mid-level"
        
        # Work location
        if any(word in text_lower for word in ['remote', 'work from home', 'wfh']):
            classification.work_location = "Remote"
        elif any(word in text_lower for word in ['hybrid', 'flexible']):
            classification.work_location = "Hybrid"
        else:
            classification.work_location = "On-site"
        
        # Skills/Technologies
        skills = []
        if 'python' in text_lower:
            skills.append('Python')
        if 'javascript' in text_lower or 'js' in text_lower:
            skills.append('JavaScript')
        if 'java' in text_lower:
            skills.append('Java')
        if 'react' in text_lower:
            skills.append('React')
        if 'django' in text_lower:
            skills.append('Django')
        if 'postgresql' in text_lower or 'postgres' in text_lower:
            skills.append('PostgreSQL')
        if 'aws' in text_lower:
            skills.append('AWS')
        if 'docker' in text_lower:
            skills.append('Docker')
        if 'kubernetes' in text_lower or 'k8s' in text_lower:
            skills.append('Kubernetes')
        
        if skills:
            classification.skills_technologies = ', '.join(skills)
        
        # Benefits
        classification.bonuses = any(word in text_lower for word in ['bonus', 'profit sharing', 'incentive'])
        classification.health_insurance = any(word in text_lower for word in ['health', 'medical', 'insurance'])
        classification.stock_options = any(word in text_lower for word in ['stock', 'equity', 'shares'])
        
        # Work schedule
        if any(word in text_lower for word in ['flexible', 'flex']):
            classification.work_schedule = "Flexible Hours"
        else:
            classification.work_schedule = "Fixed Hours"
        
        # Company size
        if any(word in text_lower for word in ['startup', 'start-up']):
            classification.company_size = "Startup"
        elif any(word in text_lower for word in ['enterprise', 'corporation', 'corp']):
            classification.company_size = "Large Corporation"
        else:
            classification.company_size = "SME"
        
        log.info(f"Mock classification completed for job posting")
        return classification

# Create a global instance
mock_job_classifier = MockJobClassifier()


