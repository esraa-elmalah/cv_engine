"""
Unit tests for Image Validation service.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import json

from app.services.image_validator import (
    ImageValidationService, 
    ImageValidationConfig,
    OpenAIImageValidator,
    SimpleImageValidator,
    ImageValidationResult
)
from app.models.cv_models import CVTemplate


class TestImageValidationConfig:
    """Test image validation configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ImageValidationConfig()
        
        assert config.validation_model == "gpt-4.1-mini"
        assert config.max_validation_attempts == 3
        assert config.min_confidence_score == 0.7
        assert config.enable_age_validation is True
        assert config.enable_gender_validation is True
        assert config.enable_professional_validation is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ImageValidationConfig(
            validation_model="gpt-4o-mini",
            max_validation_attempts=5,
            min_confidence_score=0.8,
            enable_age_validation=False
        )
        
        assert config.validation_model == "gpt-4o-mini"
        assert config.max_validation_attempts == 5
        assert config.min_confidence_score == 0.8
        assert config.enable_age_validation is False


class TestSimpleImageValidator:
    """Test simple image validator functionality."""
    
    def test_validator_initialization(self, image_validation_config):
        """Test validator initialization."""
        validator = SimpleImageValidator(image_validation_config)
        assert validator.config == image_validation_config
    
    @pytest.mark.asyncio
    async def test_validate_image_success(self, image_validation_config, sample_cv_template, mock_image_data):
        """Test successful image validation."""
        validator = SimpleImageValidator(image_validation_config)
        
        result = await validator.validate_image(mock_image_data, sample_cv_template)
        
        assert isinstance(result, ImageValidationResult)
        assert result.is_valid is True
        assert result.confidence_score == 0.8
        assert len(result.issues) == 0
        assert "estimated_age" in result.profile_match
        assert result.profile_match["experience_appropriate"] is True
    
    @pytest.mark.asyncio
    async def test_validate_image_with_invalid_data(self, image_validation_config, sample_cv_template):
        """Test image validation with invalid image data."""
        validator = SimpleImageValidator(image_validation_config)
        
        invalid_image_data = "not an image"
        result = await validator.validate_image(invalid_image_data, sample_cv_template)
        
        assert result.is_valid is False
        assert "No image found" in result.issues[0]
    
    @pytest.mark.asyncio
    async def test_validate_image_senior_role_validation(self, image_validation_config, senior_cv_template, mock_image_data):
        """Test validation for senior role with insufficient experience."""
        # Create a senior template with low experience
        senior_template = CVTemplate(
            role="Senior Software Engineer",
            level="Senior",
            industry="Technology",
            skills=["Python", "JavaScript"],
            experience_years=2  # Too low for senior
        )
        
        validator = SimpleImageValidator(image_validation_config)
        result = await validator.validate_image(mock_image_data, senior_template)
        
        assert result.is_valid is False
        assert "Senior role should have more experience" in result.issues[0]
    
    def test_estimate_age_from_experience(self, image_validation_config):
        """Test age estimation based on experience."""
        validator = SimpleImageValidator(image_validation_config)
        
        assert validator._estimate_age_from_experience(1) == "20s"
        assert validator._estimate_age_from_experience(3) == "late 20s-early 30s"
        assert validator._estimate_age_from_experience(6) == "30s"
        assert validator._estimate_age_from_experience(10) == "30s-40s"
        assert validator._estimate_age_from_experience(15) == "40s+"


class TestOpenAIImageValidator:
    """Test OpenAI image validator functionality."""
    
    def test_validator_initialization(self, image_validation_config):
        """Test validator initialization."""
        with patch('app.services.image_validator.create_agent'):
            validator = OpenAIImageValidator(image_validation_config)
            assert validator.config == image_validation_config
            assert validator.agent is not None
    
    @pytest.mark.asyncio
    async def test_validate_image_success(self, image_validation_config, sample_cv_template, mock_image_data):
        """Test successful OpenAI image validation."""
        with patch('app.services.image_validator.create_agent'):
            with patch('openai.OpenAI') as mock_openai:
                validator = OpenAIImageValidator(image_validation_config)
                
                # Mock OpenAI response
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message = Mock()
                mock_response.choices[0].message.content = json.dumps({
                    "is_valid": True,
                    "confidence_score": 0.85,
                    "issues": [],
                    "suggestions": [],
                    "profile_match": {
                        "age_appropriate": True,
                        "professional_appearance": True,
                        "industry_suitable": True,
                        "estimated_age": "30s",
                        "gender": "male",
                        "overall_impression": "Professional appearance"
                    }
                })
                
                mock_openai.return_value.chat.completions.create.return_value = mock_response
                
                result = await validator.validate_image(mock_image_data, sample_cv_template)
                
                assert result.is_valid is True
                assert result.confidence_score == 0.85
                assert len(result.issues) == 0
                assert result.profile_match["estimated_age"] == "30s"
                assert result.profile_match["gender"] == "male"
    
    @pytest.mark.asyncio
    async def test_validate_image_invalid_response(self, image_validation_config, sample_cv_template, mock_image_data):
        """Test validation with invalid JSON response."""
        with patch('app.services.image_validator.create_agent'):
            with patch('openai.OpenAI') as mock_openai:
                validator = OpenAIImageValidator(image_validation_config)
                
                # Mock invalid JSON response
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message = Mock()
                mock_response.choices[0].message.content = "invalid json"
                
                mock_openai.return_value.chat.completions.create.return_value = mock_response
                
                result = await validator.validate_image(mock_image_data, sample_cv_template)
                
                assert result.is_valid is False
                assert result.confidence_score == 0.0
                assert "Failed to parse validation response" in result.issues[0]
    
    @pytest.mark.asyncio
    async def test_validate_image_api_error(self, image_validation_config, sample_cv_template, mock_image_data):
        """Test validation when OpenAI API fails."""
        with patch('app.services.image_validator.create_agent'):
            with patch('openai.OpenAI') as mock_openai:
                validator = OpenAIImageValidator(image_validation_config)
                
                # Mock API error
                mock_openai.return_value.chat.completions.create.side_effect = Exception("API Error")
                
                result = await validator.validate_image(mock_image_data, sample_cv_template)
                
                assert result.is_valid is False
                assert result.confidence_score == 0.0
                assert "Validation error: API Error" in result.issues[0]


class TestImageValidationService:
    """Test image validation service functionality."""
    
    def test_service_initialization_with_openai(self, image_validation_config):
        """Test service initialization with OpenAI available."""
        with patch('os.getenv', return_value="test-key"):
            with patch('app.services.image_validator.OpenAIImageValidator') as mock_openai_validator:
                service = ImageValidationService(image_validation_config)
                
                assert len(service.validators) == 2  # OpenAI + Simple
                assert isinstance(service.validators[0], Mock)  # OpenAI validator
                assert isinstance(service.validators[1], SimpleImageValidator)
    
    def test_service_initialization_without_openai(self, image_validation_config):
        """Test service initialization without OpenAI."""
        with patch('os.getenv', return_value=None):
            service = ImageValidationService(image_validation_config)
            
            assert len(service.validators) == 1  # Only Simple
            assert isinstance(service.validators[0], SimpleImageValidator)
    
    @pytest.mark.asyncio
    async def test_validate_image_success(self, image_validation_service, sample_cv_template, mock_image_data):
        """Test successful image validation through service."""
        result = await image_validation_service.validate_image(mock_image_data, sample_cv_template)
        
        assert isinstance(result, ImageValidationResult)
        assert result.is_valid is True
        assert result.confidence_score >= 0.6
    
    @pytest.mark.asyncio
    async def test_validate_image_with_retry_success(self, image_validation_service, sample_cv_template, mock_image_data):
        """Test validation with retry logic."""
        result = await image_validation_service.validate_with_retry(
            mock_image_data, 
            sample_cv_template, 
            max_attempts=2
        )
        
        assert isinstance(result, ImageValidationResult)
        assert result.is_valid is True
    
    @pytest.mark.asyncio
    async def test_validate_image_with_retry_failure(self, image_validation_service, sample_cv_template):
        """Test validation with retry logic when all attempts fail."""
        # Use invalid image data to force failures
        invalid_image_data = "invalid"
        
        result = await image_validation_service.validate_with_retry(
            invalid_image_data, 
            sample_cv_template, 
            max_attempts=2
        )
        
        assert isinstance(result, ImageValidationResult)
        assert result.is_valid is False
    
    @pytest.mark.asyncio
    async def test_validate_image_all_validators_fail(self, image_validation_config, sample_cv_template, mock_image_data):
        """Test validation when all validators fail."""
        # Create service with failing validators
        with patch('app.services.image_validator.SimpleImageValidator.validate_image', 
                  side_effect=Exception("Validator failed")):
            service = ImageValidationService(image_validation_config)
            
            result = await service.validate_image(mock_image_data, sample_cv_template)
            
            assert result.is_valid is False
            assert result.confidence_score == 0.0
            assert "All validation methods failed" in result.issues[0]


class TestImageValidationResult:
    """Test image validation result data structure."""
    
    def test_result_creation(self):
        """Test validation result creation."""
        result = ImageValidationResult(
            is_valid=True,
            confidence_score=0.85,
            issues=["Issue 1"],
            suggestions=["Suggestion 1"],
            profile_match={"age": "30s", "gender": "male"}
        )
        
        assert result.is_valid is True
        assert result.confidence_score == 0.85
        assert result.issues == ["Issue 1"]
        assert result.suggestions == ["Suggestion 1"]
        assert result.profile_match == {"age": "30s", "gender": "male"}
    
    def test_result_defaults(self):
        """Test validation result with default values."""
        result = ImageValidationResult(
            is_valid=False,
            confidence_score=0.0,
            issues=[],
            suggestions=[],
            profile_match={}
        )
        
        assert result.is_valid is False
        assert result.confidence_score == 0.0
        assert len(result.issues) == 0
        assert len(result.suggestions) == 0
        assert len(result.profile_match) == 0
