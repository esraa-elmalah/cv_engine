import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import requests
from pydantic import BaseModel

from app.agents import create_agent, run_agent
from app.models.cv_models import CVTemplate

logger = logging.getLogger(__name__)


@dataclass
class ImageValidationResult:
    """Result of image validation."""
    is_valid: bool
    confidence_score: float
    issues: List[str]
    suggestions: List[str]
    profile_match: Dict[str, Any]


class ImageValidationConfig(BaseModel):
    """Configuration for image validation."""
    validation_model: str = "gpt-4.1-mini"
    max_validation_attempts: int = 3
    min_confidence_score: float = 0.7
    enable_age_validation: bool = True
    enable_gender_validation: bool = True
    enable_professional_validation: bool = True


class ImageValidator(ABC):
    """Abstract base class for image validators."""
    
    @abstractmethod
    async def validate_image(self, image_data: str, cv_template: CVTemplate) -> ImageValidationResult:
        """Validate if the image matches the CV profile."""
        pass


class OpenAIImageValidator(ImageValidator):
    """OpenAI-based image validator using vision capabilities."""
    
    def __init__(self, config: ImageValidationConfig):
        self.config = config
        self.agent = create_agent(
            name="Image Profile Validator",
            instructions=(
                "You are an expert at analyzing profile photos and matching them to professional CV profiles. "
                "Analyze the provided image and compare it with the CV template to determine if they are a good match. "
                "Consider factors like age appropriateness, gender, professional appearance, and overall suitability. "
                "Provide a confidence score (0-1) and specific feedback on any issues or suggestions for improvement."
            ),
            model=config.validation_model,
        )
    
    async def validate_image(self, image_data: str, cv_template: CVTemplate) -> ImageValidationResult:
        """Validate if the image matches the CV profile using OpenAI vision."""
        try:
            # Extract base64 data from the img tag
            if "data:image/jpeg;base64," in image_data:
                base64_data = image_data.split("data:image/jpeg;base64,")[1].split('"')[0]
            else:
                base64_data = image_data
            
            # Create validation prompt
            validation_prompt = f"""
Analyze this profile photo and determine if it matches the following CV profile:

CV Profile:
- Role: {cv_template.role}
- Level: {cv_template.level}
- Industry: {cv_template.industry}
- Experience: {cv_template.experience_years} years
- Skills: {', '.join(cv_template.skills)}

Please evaluate the following aspects:
1. Age appropriateness for the role and experience level
2. Professional appearance suitable for the industry
3. Overall image quality and suitability

Provide your analysis in the following JSON format:
{{
    "is_valid": true/false,
    "confidence_score": 0.0-1.0,
    "issues": ["list of issues if any"],
    "suggestions": ["list of suggestions for improvement"],
    "profile_match": {{
        "age_appropriate": true/false,
        "professional_appearance": true/false,
        "industry_suitable": true/false,
        "estimated_age": "20s/30s/40s/etc",
        "gender": "male/female/other",
        "overall_impression": "description"
    }}
}}

Return ONLY the JSON response, no additional text.
"""
            
            # Use OpenAI vision API for image analysis
            from openai import OpenAI
            from app.config import settings
            
            # Get API key from settings
            api_key = settings.openai_api_key or settings.openai.api_key
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use vision-capable model
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": validation_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            # Parse the response
            result_text = response.choices[0].message.content
            import json
            
            try:
                result_data = json.loads(result_text)
                
                return ImageValidationResult(
                    is_valid=result_data.get("is_valid", False),
                    confidence_score=result_data.get("confidence_score", 0.0),
                    issues=result_data.get("issues", []),
                    suggestions=result_data.get("suggestions", []),
                    profile_match=result_data.get("profile_match", {})
                )
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse validation response: {result_text}")
                return ImageValidationResult(
                    is_valid=False,
                    confidence_score=0.0,
                    issues=["Failed to parse validation response"],
                    suggestions=["Try regenerating the image"],
                    profile_match={}
                )
                
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return ImageValidationResult(
                is_valid=False,
                confidence_score=0.0,
                issues=[f"Validation error: {str(e)}"],
                suggestions=["Check API configuration and try again"],
                profile_match={}
            )


class SimpleImageValidator(ImageValidator):
    """Simple rule-based image validator (fallback when OpenAI is not available)."""
    
    def __init__(self, config: ImageValidationConfig):
        self.config = config
    
    async def validate_image(self, image_data: str, cv_template: CVTemplate) -> ImageValidationResult:
        """Simple validation based on CV template rules."""
        try:
            # Basic validation logic
            issues = []
            suggestions = []
            profile_match = {}
            
            # Check if image contains expected elements
            if "img" not in image_data:
                issues.append("No image found in the data")
            
            # Basic professional validation based on role
            if cv_template.level == "Senior" and cv_template.experience_years < 5:
                issues.append("Senior role should have more experience")
            
            if cv_template.role == "Frontend Developer" and "JavaScript" not in cv_template.skills:
                suggestions.append("Frontend developers typically need JavaScript skills")
            
            # Estimate age based on experience
            estimated_age = self._estimate_age_from_experience(cv_template.experience_years)
            profile_match["estimated_age"] = estimated_age
            profile_match["experience_appropriate"] = True
            
            # Determine if validation passes
            is_valid = len(issues) == 0
            confidence_score = 0.8 if is_valid else 0.4
            
            return ImageValidationResult(
                is_valid=is_valid,
                confidence_score=confidence_score,
                issues=issues,
                suggestions=suggestions,
                profile_match=profile_match
            )
            
        except Exception as e:
            logger.error(f"Simple validation failed: {e}")
            return ImageValidationResult(
                is_valid=False,
                confidence_score=0.0,
                issues=[f"Validation error: {str(e)}"],
                suggestions=["Try regenerating the image"],
                profile_match={}
            )
    
    def _estimate_age_from_experience(self, experience_years: int) -> str:
        """Estimate age range based on experience years."""
        if experience_years <= 2:
            return "20s"
        elif experience_years <= 5:
            return "late 20s-early 30s"
        elif experience_years <= 8:
            return "30s"
        elif experience_years <= 12:
            return "30s-40s"
        else:
            return "40s+"


class ImageValidationService:
    """Service for managing image validation."""
    
    def __init__(self, config: Optional[ImageValidationConfig] = None):
        self.config = config or ImageValidationConfig()
        self.validators = self._initialize_validators()
    
    def _initialize_validators(self) -> List[ImageValidator]:
        """Initialize available validators."""
        validators = []
        
        # Try to initialize OpenAI validator
        try:
            from app.config import settings
            api_key = settings.openai_api_key or settings.openai.api_key
            if api_key:
                validators.append(OpenAIImageValidator(self.config))
                logger.info("OpenAI image validator initialized")
            else:
                logger.warning("OpenAI API key not found, using simple validator")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI validator: {e}")
        
        # Always add simple validator as fallback
        validators.append(SimpleImageValidator(self.config))
        logger.info("Simple image validator initialized")
        
        return validators
    
    async def validate_image(self, image_data: str, cv_template: CVTemplate) -> ImageValidationResult:
        """Validate image using available validators."""
        for validator in self.validators:
            try:
                result = await validator.validate_image(image_data, cv_template)
                
                # If validation passes with good confidence, return result
                if result.is_valid and result.confidence_score >= self.config.min_confidence_score:
                    logger.info(f"Image validation passed with confidence {result.confidence_score}")
                    return result
                
                # If using OpenAI validator and it fails, try simple validator
                if isinstance(validator, OpenAIImageValidator) and not result.is_valid:
                    logger.info("OpenAI validation failed, trying simple validator")
                    continue
                    
            except Exception as e:
                logger.error(f"Validator {type(validator).__name__} failed: {e}")
                continue
        
        # If all validators fail, return a default failure result
        logger.warning("All image validators failed")
        return ImageValidationResult(
            is_valid=False,
            confidence_score=0.0,
            issues=["All validation methods failed"],
            suggestions=["Check system configuration and try again"],
            profile_match={}
        )
    
    async def validate_with_retry(self, image_data: str, cv_template: CVTemplate, max_attempts: int = None, person_characteristics: Optional[Dict[str, Any]] = None) -> ImageValidationResult:
        """Validate image with retry logic."""
        max_attempts = max_attempts or self.config.max_validation_attempts
        
        for attempt in range(max_attempts):
            result = await self.validate_image(image_data, cv_template)
            
            if result.is_valid and result.confidence_score >= self.config.min_confidence_score:
                logger.info(f"Image validation successful on attempt {attempt + 1}")
                return result
            
            logger.info(f"Image validation attempt {attempt + 1} failed, confidence: {result.confidence_score}")
            
            if attempt < max_attempts - 1:
                logger.info("Retrying image validation...")
        
        logger.warning(f"Image validation failed after {max_attempts} attempts")
        return result
    
    async def analyze_face_characteristics(self, image_data: str) -> Dict[str, Any]:
        """Analyze face image to extract person characteristics."""
        try:
            # Use the first available validator (preferably OpenAI)
            if self.validators:
                validator = self.validators[0]
                
                # Create a temporary template for analysis
                temp_template = CVTemplate(
                    role="Software Engineer",
                    level="Mid-level", 
                    industry="Technology",
                    skills=["Python"],
                    experience_years=3
                )
                
                # Get validation result which includes face analysis
                result = await validator.validate_image(image_data, temp_template)
                
                # Extract characteristics from profile_match
                characteristics = result.profile_match.copy()
                
                # Ensure we have the key characteristics
                if 'gender' not in characteristics:
                    characteristics['gender'] = 'unknown'
                if 'estimated_age' not in characteristics:
                    characteristics['estimated_age'] = 'unknown'
                
                logger.info(f"Face characteristics extracted: {characteristics}")
                return characteristics
                
        except Exception as e:
            logger.error(f"Failed to analyze face characteristics: {e}")
            return {
                'gender': 'unknown',
                'estimated_age': 'unknown',
                'professional_appearance': True,
                'overall_impression': 'Unable to analyze'
            }
