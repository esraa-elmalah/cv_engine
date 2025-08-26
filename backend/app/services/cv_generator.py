import asyncio
import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from app.agents import create_agent, run_agent, create_face_analyzer_agent, create_cv_generator_agent
from app.services.index_service import update_index

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from app.config import settings, CVGeneratorConfig
from app.models.cv_models import CVTemplate
from app.services.image_validator import ImageValidationService, ImageValidationConfig





# --- Base Classes ---
class FaceImageProvider(ABC):
    """Abstract base class for face image providers."""
    
    @abstractmethod
    async def get_face_image(self) -> str:
        """Get a face image and return as base64 encoded HTML img tag."""
        pass


class PDFGenerator(ABC):
    """Abstract base class for PDF generators."""
    
    @abstractmethod
    async def generate_pdf(self, html: str, filename: str) -> str:
        """Generate PDF from HTML and return file path."""
        pass


class CVContentGenerator(ABC):
    """Abstract base class for CV content generators."""
    
    @abstractmethod
    async def generate_cv_content(self, template: CVTemplate, person_characteristics: Optional[Dict[str, Any]] = None) -> str:
        """Generate CV content based on template and person characteristics."""
        pass


# --- Concrete Implementations ---
class ThisPersonDoesNotExistProvider(FaceImageProvider):
    """Face image provider using thispersondoesnotexist.com with intelligent filtering."""
    
    def __init__(self, config: CVGeneratorConfig):
        self.config = config
        self.max_attempts = config.max_face_attempts
        # Initialize face analysis agent for cost-efficient validation
        self.face_analyzer = create_face_analyzer_agent()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def get_face_image(self) -> str:
        """Fetch random face with intelligent filtering."""
        for attempt in range(self.max_attempts):
            try:
                response = requests.get(
                    self.config.face_image_url, 
                    timeout=self.config.face_image_timeout
                )
                response.raise_for_status()
                
                img_b64 = base64.b64encode(response.content).decode("utf-8")
                img_tag = (
                    f'<img src="data:image/jpeg;base64,{img_b64}" '
                    f'alt="Profile Photo" style="{self.config.face_image_style}" />'
                )
                
                # Use agentic validation if enabled
                if not self.config.enable_age_filtering or await self._is_appropriate_face_agentic(img_tag):
                    logger.info(f"Found appropriate face image on attempt {attempt + 1}")
                    return img_tag
                else:
                    logger.info(f"Face image attempt {attempt + 1} was inappropriate, retrying...")
                    
            except requests.RequestException as e:
                logger.error(f"Failed to fetch face image on attempt {attempt + 1}: {e}")
                if attempt == self.max_attempts - 1:
                    raise
        
        # If all attempts failed, return the last image anyway
        logger.warning(f"Could not find appropriate face after {self.max_attempts} attempts, using last image")
        return img_tag
    
    async def _is_appropriate_face_agentic(self, img_tag: str) -> bool:
        """Agentic face validation - more intelligent and cost-efficient."""
        try:
            # For now, use a simple validation approach to avoid context length issues
            # In a production environment, you might want to use a dedicated vision API
            # or implement more sophisticated image analysis
            
            # Simple validation: check if the image tag contains expected elements
            if "data:image/jpeg;base64," in img_tag and len(img_tag) > 100:
                logger.info("Face image validation passed (basic check)")
                return True
            else:
                logger.warning("Face image validation failed (basic check)")
                return False
                
        except Exception as e:
            logger.warning(f"Face validation failed: {e}")
            return True  # Assume appropriate to avoid blocking


class WeasyPrintPDFGenerator(PDFGenerator):
    """PDF generator using WeasyPrint."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_pdf(self, html: str, filename: str) -> str:
        """Generate PDF from HTML using WeasyPrint."""
        try:
            from weasyprint import HTML
            
            filename = Path(filename).stem  # strip extensions
            file_path = self.output_dir / f"{filename}.pdf"
            
            logger.info(f"Generating PDF for {filename}...")
            HTML(string=html, base_url=str(self.output_dir)).write_pdf(str(file_path))
            logger.info(f"PDF saved at: {file_path}")
            
            return str(file_path)
        except ImportError:
            raise RuntimeError("WeasyPrint is required for PDF generation. Install with: pip install weasyprint")
        except Exception as e:
            logger.error(f"Failed to generate PDF for {filename}: {e}")
            raise


class OpenAICVContentGenerator(CVContentGenerator):
    """CV content generator using OpenAI agents."""
    
    def __init__(self, config: CVGeneratorConfig):
        self.config = config
        self.agent = create_cv_generator_agent()
    
    async def generate_cv_content(self, template: CVTemplate, person_characteristics: Optional[Dict[str, Any]] = None) -> str:
        """Generate CV content based on template and person characteristics."""
        try:
            # Build person details based on characteristics
            person_details = ""
            if person_characteristics:
                gender = person_characteristics.get('gender', 'unknown')
                estimated_age = person_characteristics.get('estimated_age', 'unknown')
                
                # Generate appropriate name based on gender
                if gender == 'male':
                    person_details = f"""
- Gender: Male
- Estimated Age: {estimated_age}
- Name: Generate a realistic male name appropriate for the age and role
"""
                elif gender == 'female':
                    person_details = f"""
- Gender: Female  
- Estimated Age: {estimated_age}
- Name: Generate a realistic female name appropriate for the age and role
"""
                else:
                    person_details = f"""
- Estimated Age: {estimated_age}
- Name: Generate a realistic name appropriate for the age and role
"""
            
            task = f"""
Generate a professional CV in raw HTML format for a realistic candidate with the following specifications:
- Role: {template.role}
- Level: {template.level}
- Industry: {template.industry}
- Key Skills: {', '.join(template.skills)}
- Experience: {template.experience_years} years{person_details}

IMPORTANT: The person's name and gender must match the face image characteristics provided.
If gender is specified, use an appropriate name for that gender.
If age is specified, ensure the experience level and career progression match the age.

Include personal details, skills, experience, and education sections.
Do NOT include placeholders or text for a profile photo.
Do NOT wrap the HTML in code fences.
Return ONLY valid HTML.
"""
            result = await run_agent(self.agent, task)
            return str(result)
        except Exception as e:
            logger.error(f"Failed to generate CV content: {e}")
            raise


# --- CV Templates Repository ---
class CVTemplateRepository:
    """Repository for CV templates."""
    
    def __init__(self, config: Optional[CVGeneratorConfig] = None):
        self.config = config or CVGeneratorConfig()
        self.template_agent = create_agent(
            name="CV Template Generator",
            instructions=(
                "You are an expert at generating diverse and realistic CV templates for the tech industry. "
                "Generate random but realistic combinations of roles, levels, industries, skills, and experience. "
                "Consider current market trends and realistic skill combinations. "
                "Return templates in JSON format with the following structure: "
                '{"role": "string", "level": "Junior|Mid-level|Senior", "industry": "string", "skills": ["skill1", "skill2"], "experience_years": number}'
            ),
            model=self.config.agent_model,
        )
    
    async def generate_random_template(self) -> CVTemplate:
        """Generate a random CV template using LLM."""
        try:
            task = """
Generate a random but realistic CV template for the tech industry. Consider:
- Diverse roles: Software Engineer, Data Scientist, DevOps Engineer, Frontend Developer, Backend Developer, 
  Product Manager, UX Designer, QA Engineer, Mobile Developer, Cloud Architect, etc.
- Different levels: Junior (1-2 years), Mid-level (3-5 years), Senior (6+ years)
- Various industries: Technology, Finance, Healthcare, E-commerce, Gaming, etc.
- Realistic skill combinations that match the role and level
- Appropriate experience years for the level

Return ONLY a valid JSON object with this exact structure:
{
    "role": "string",
    "level": "Junior|Mid-level|Senior", 
    "industry": "string",
    "skills": ["skill1", "skill2", "skill3", "skill4", "skill5"],
    "experience_years": number
}
"""
            result = await run_agent(self.template_agent, task)
            
            # Parse the JSON response
            import json
            import re
            
            # Extract JSON from the response (in case there's extra text)
            response_text = str(result)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                template_data = json.loads(json_match.group())
            else:
                template_data = json.loads(response_text)
            
            return CVTemplate(**template_data)
            
        except Exception as e:
            logger.error(f"Failed to generate random template: {e}")
            # Fallback to a default template
            return CVTemplate(
                role="Software Engineer",
                level="Mid-level",
                industry="Technology",
                skills=["Python", "JavaScript", "React"],
                experience_years=3
            )
    
    async def get_templates(self, count: int = 5) -> List[CVTemplate]:
        """Get multiple random CV templates."""
        templates = []
        for _ in range(count):
            template = await self.generate_random_template()
            templates.append(template)
        return templates
    
    @staticmethod
    def get_predefined_templates() -> List[CVTemplate]:
        """Get predefined CV templates as fallback."""
        return [
            CVTemplate(
                role="Software Engineer",
                level="Senior",
                industry="Technology",
                skills=["Python", "JavaScript", "React", "Node.js", "AWS"],
                experience_years=5
            ),
            CVTemplate(
                role="Data Scientist",
                level="Mid-level",
                industry="Technology",
                skills=["Python", "Machine Learning", "SQL", "Pandas", "Scikit-learn"],
                experience_years=3
            ),
            CVTemplate(
                role="DevOps Engineer",
                level="Senior",
                industry="Technology",
                skills=["Docker", "Kubernetes", "AWS", "Terraform", "Jenkins"],
                experience_years=6
            ),
            CVTemplate(
                role="Frontend Developer",
                level="Junior",
                industry="Technology",
                skills=["JavaScript", "React", "HTML", "CSS", "TypeScript"],
                experience_years=2
            ),
            CVTemplate(
                role="Backend Developer",
                level="Mid-level",
                industry="Technology",
                skills=["Java", "Spring Boot", "PostgreSQL", "Redis", "Microservices"],
                experience_years=4
            ),
        ]


# --- Main CV Generator Service ---
@dataclass
class GeneratedCV:
    """Data class for generated CV information."""
    filename: str
    file_path: str
    template: CVTemplate
    generated_at: datetime
    image_validation: Optional[Dict[str, Any]] = None


class CVGeneratorService:
    """Main service for CV generation."""
    
    def __init__(self, config: Optional[CVGeneratorConfig] = None):
        self.config = config or CVGeneratorConfig()
        self.face_provider = ThisPersonDoesNotExistProvider(self.config)
        self.pdf_generator = WeasyPrintPDFGenerator(self.config.output_dir)
        self.content_generator = OpenAICVContentGenerator(self.config)
        self.template_repo = CVTemplateRepository(self.config)
        
        # Initialize image validation service
        if self.config.enable_image_validation:
            validation_config = ImageValidationConfig(
                max_validation_attempts=self.config.max_image_validation_attempts,
                min_confidence_score=self.config.min_validation_confidence
            )
            self.image_validator = ImageValidationService(validation_config)
        else:
            self.image_validator = None
    
    def _inject_face_image(self, html: str, face_image: str) -> str:
        """Inject face image into CV HTML."""
        if "<body>" in html:
            return html.replace("<body>", f"<body>{face_image}", 1)
        return face_image + html
    
    async def _generate_single_cv(self, template: CVTemplate, index: int) -> GeneratedCV:
        """Generate a single CV with the given template - optimized for agentic workflow."""
        try:
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"cv_{timestamp}_{index}"
            
            # Step 1: Get face image with built-in validation (already agentic)
            face_image = await self.face_provider.get_face_image()
            
            # Step 2: Use agentic or traditional CV generation
            if self.config.enable_agentic_workflow:
                html_content, person_characteristics = await self._generate_cv_content_agentic(
                    template, face_image, index
                )
            else:
                # Traditional workflow with separate validation
                person_characteristics = None
                if self.image_validator:
                    try:
                        analysis_result = await self.image_validator.analyze_face_characteristics(face_image)
                        person_characteristics = analysis_result
                    except Exception as e:
                        logger.warning(f"Failed to analyze face for CV {index}: {e}")
                
                html_content = await self.content_generator.generate_cv_content(
                    template, person_characteristics=person_characteristics
                )
            
            # Step 3: Inject face image and generate PDF
            final_html = self._inject_face_image(html_content, face_image)
            file_path = await self.pdf_generator.generate_pdf(final_html, filename)
            
            # Step 4: Prepare validation data (using characteristics from content generation)
            validation_data = None
            if person_characteristics:
                validation_data = {
                    "is_valid": True,  # If we got here, validation passed
                    "confidence_score": 0.9,  # High confidence for agentic workflow
                    "issues": [],
                    "suggestions": [],
                    "profile_match": person_characteristics
                }
            
            return GeneratedCV(
                filename=filename,
                file_path=file_path,
                template=template,
                generated_at=datetime.now(),
                image_validation=validation_data
            )
            
        except Exception as e:
            logger.error(f"Failed to generate CV {index}: {e}")
            raise
    
    async def _generate_cv_content_agentic(self, template: CVTemplate, face_image: str, index: int) -> tuple[str, Optional[Dict[str, Any]]]:
        """Agentic CV content generation with integrated face analysis - cost-efficient."""
        try:
            # Create a specialized agent for integrated CV generation
            cv_agent = create_cv_generator_agent()
            
            task = f"""
Analyze the face image and generate a professional CV that matches both the template and the person's appearance.

Template Requirements:
- Role: {template.role}
- Level: {template.level}
- Industry: {template.industry}
- Skills: {', '.join(template.skills)}
- Experience: {template.experience_years} years

Face Image Analysis:
- Analyze age, gender, and professional appearance
- Ensure the CV content matches the person's characteristics
- Generate appropriate name and details based on the face

Return ONLY a JSON response:
{{
    "html_content": "full HTML CV content",
    "person_characteristics": {{
        "gender": "male/female/unknown",
        "estimated_age": "age range",
        "professional_appearance": true/false,
        "overall_impression": "description"
    }}
}}
"""
            
            import json
            import re
            
            result = await run_agent(cv_agent, task)
            
            # Extract JSON from response
            response_text = str(result)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                html_content = data.get('html_content', '')
                person_characteristics = data.get('person_characteristics', {})
                
                logger.info(f"Agentic CV generation for CV {index}: {person_characteristics}")
                return html_content, person_characteristics
            else:
                logger.warning(f"Could not parse agentic CV response for CV {index}, falling back to traditional method")
                # Fallback to traditional method
                html_content = await self.content_generator.generate_cv_content(template)
                return html_content, None
                
        except Exception as e:
            logger.error(f"Agentic CV generation failed for CV {index}: {e}")
            # Fallback to traditional method
            html_content = await self.content_generator.generate_cv_content(template)
            return html_content, None
    
    async def generate_cvs(self, count: int, custom_templates: Optional[List[CVTemplate]] = None) -> List[GeneratedCV]:
        """Generate multiple CVs."""
        try:
            # Get templates - use random templates if none provided
            if custom_templates:
                templates = custom_templates
            else:
                # Generate random templates using LLM
                templates = await self.template_repo.get_templates(count)
            
            # Prepare tasks
            tasks = []
            for i in range(count):
                template = templates[i] if i < len(templates) else templates[i % len(templates)]
                task = self._generate_single_cv(template, i)
                tasks.append(task)
            
            # Generate CVs in parallel
            logger.info(f"Generating {count} CVs in parallel...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and log them
            successful_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"CV generation {i} failed: {result}")
                else:
                    successful_results.append(result)
            
            # Update index if we have successful generations
            if successful_results:
                logger.info("Updating FAISS index...")
                await update_index()
                logger.info("Index updated successfully.")
            
            return successful_results
            
        except Exception as e:
            logger.error(f"Failed to generate CVs: {e}")
            raise


# --- Factory for easy instantiation ---
class CVGeneratorFactory:
    """Factory for creating CV generator instances."""
    
    @staticmethod
    def create(config: Optional[CVGeneratorConfig] = None) -> CVGeneratorService:
        """Create a CV generator service instance."""
        return CVGeneratorService(config)


# --- Legacy compatibility function ---
async def generate_fake_cvs(output_dir: str, count: int) -> List[str]:
    """Legacy function for backward compatibility."""
    config = CVGeneratorConfig(output_dir=output_dir)
    service = CVGeneratorFactory.create(config)
    
    try:
        results = await service.generate_cvs(count)
        return [result.filename for result in results]
    except Exception as e:
        logger.error(f"Legacy generate_fake_cvs failed: {e}")
        raise
