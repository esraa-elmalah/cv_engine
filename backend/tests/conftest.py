"""
Pytest configuration and common fixtures for CV Engine tests.
"""

import os
import pytest
import asyncio
from pathlib import Path
from typing import Generator

from app.config import CVGeneratorConfig
from app.services.image_validator import ImageValidationConfig
from app.models.cv_models import CVTemplate
from app.services.cv_generator import CVGeneratorFactory
from app.services.image_validator import ImageValidationService


# OpenAI API key check
def pytest_configure(config):
    """Configure pytest with OpenAI API key checks."""
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not set. Tests requiring real API calls will be skipped.")

def requires_openai_api_key():
    """Decorator to skip tests that require an OpenAI API key."""
    return pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set"
    )


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def cv_generator_config() -> CVGeneratorConfig:
    """Provide a test CV generator configuration."""
    return CVGeneratorConfig(
        output_dir="tests/test_data/cvs",
        enable_image_validation=True,
        max_image_validation_attempts=2,
        min_validation_confidence=0.6,
        batch_size=2
    )


@pytest.fixture
def image_validation_config() -> ImageValidationConfig:
    """Provide a test image validation configuration."""
    return ImageValidationConfig(
        max_validation_attempts=2,
        min_confidence_score=0.6
    )


@pytest.fixture
def sample_cv_template() -> CVTemplate:
    """Provide a sample CV template for testing."""
    return CVTemplate(
        role="Software Engineer",
        level="Mid-level",
        industry="Technology",
        skills=["Python", "JavaScript", "React"],
        experience_years=3
    )


@pytest.fixture
def senior_cv_template() -> CVTemplate:
    """Provide a senior-level CV template for testing."""
    return CVTemplate(
        role="Senior Software Engineer",
        level="Senior",
        industry="Technology",
        skills=["Python", "JavaScript", "React", "Node.js", "AWS"],
        experience_years=7
    )


@pytest.fixture
def junior_cv_template() -> CVTemplate:
    """Provide a junior-level CV template for testing."""
    return CVTemplate(
        role="Junior Frontend Developer",
        level="Junior",
        industry="Technology",
        skills=["JavaScript", "React", "HTML", "CSS"],
        experience_years=1
    )


@pytest.fixture
def cv_generator_service(cv_generator_config) -> CVGeneratorFactory:
    """Provide a CV generator service instance for testing."""
    return CVGeneratorFactory.create(cv_generator_config)


@pytest.fixture
def image_validation_service(image_validation_config) -> ImageValidationService:
    """Provide an image validation service instance for testing."""
    return ImageValidationService(image_validation_config)


@pytest.fixture
def mock_image_data() -> str:
    """Provide mock image data for testing."""
    return '<img src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=" alt="Profile Photo" />'


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Create test directories
    test_dirs = [
        "tests/test_data/cvs",
        "tests/test_data/index",
        "tests/test_data/logs"
    ]
    
    for test_dir in test_dirs:
        Path(test_dir).mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Cleanup after tests (optional)
    # You can add cleanup logic here if needed
