"""
Integration tests for CV generation flow.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock

from app.services.cv_generator import CVGeneratorFactory
from app.services.cv_service import CVService
from app.models.cv_models import CVTemplate


class TestCVGenerationIntegration:
    """Test complete CV generation flow."""
    
    @pytest.mark.asyncio
    async def test_cv_service_integration(self, cv_generator_config):
        """Test CV service integration with generator."""
        service = CVService()
        
        # Test service initialization
        assert service.generator is not None
        assert hasattr(service.generator, 'generate_cvs')
    
    @pytest.mark.asyncio
    async def test_template_retrieval_integration(self):
        """Test template retrieval through service."""
        service = CVService()
        templates = await service.get_available_templates()
        
        assert len(templates) > 0
        assert all(isinstance(template, CVTemplate) for template in templates)
        
        # Check template diversity
        roles = set(template.role for template in templates)
        levels = set(template.level for template in templates)
        
        assert len(roles) > 1
        assert len(levels) > 1
    
    @pytest.mark.asyncio
    async def test_random_template_generation_integration(self):
        """Test random template generation through service."""
        service = CVService()
        
        # Mock the template generation to avoid API calls
        with patch('app.services.cv_generator.create_agent') as mock_create_agent:
            with patch('app.services.cv_generator.run_agent') as mock_run_agent:
                # Mock successful response
                mock_run_agent.return_value = '{"role": "Test Engineer", "level": "Mid-level", "industry": "Technology", "skills": ["Python", "Selenium"], "experience_years": 3}'
                
                templates = await service.get_random_templates(2)
                
                assert len(templates) == 2
                assert all(isinstance(template, CVTemplate) for template in templates)
    
    @pytest.mark.asyncio
    async def test_custom_template_creation_integration(self):
        """Test custom template creation and validation."""
        service = CVService()
        
        template = service.create_custom_template(
            role="Test Engineer",
            level="Mid-level",
            industry="Technology",
            skills=["Python", "Selenium", "Pytest"],
            experience_years=3
        )
        
        assert template.role == "Test Engineer"
        assert template.level == "Mid-level"
        assert template.skills == ["Python", "Selenium", "Pytest"]
        assert template.experience_years == 3
    
    @pytest.mark.asyncio
    async def test_stats_integration(self):
        """Test statistics generation."""
        service = CVService()
        stats = await service.get_generation_stats()
        
        assert "total_generated" in stats
        assert "available_templates" in stats
        assert "service_status" in stats
        assert stats["service_status"] == "active"
        assert stats["available_templates"] > 0
    
    @pytest.mark.asyncio
    async def test_cv_generation_with_mocks(self, cv_generator_config, sample_cv_template):
        """Test CV generation with mocked external dependencies."""
        with patch('app.services.cv_generator.ThisPersonDoesNotExistProvider.get_face_image', 
                  new_callable=AsyncMock) as mock_face:
            with patch('app.services.cv_generator.run_agent', 
                      new_callable=AsyncMock) as mock_run_agent:
                with patch('app.services.cv_generator.WeasyPrintPDFGenerator.generate_pdf', 
                          new_callable=AsyncMock) as mock_pdf:
                    with patch('app.services.cv_generator.update_index') as mock_index:
                        
                        # Setup mocks
                        mock_face.return_value = '<img src="test.jpg" />'
                        mock_run_agent.return_value = "<html><body><h1>Test CV</h1></body></html>"
                        mock_pdf.return_value = "/path/to/test.pdf"
                        
                        # Create service and generate CV
                        service = CVGeneratorFactory.create(cv_generator_config)
                        result = await service._generate_single_cv(sample_cv_template, 0)
                        
                        # Verify result
                        assert result.filename is not None
                        assert result.file_path == "/path/to/test.pdf"
                        assert result.template == sample_cv_template
                        assert result.generated_at is not None
                        
                        # Verify mocks were called
                        mock_face.assert_called_once()
                        mock_run_agent.assert_called()  # May be called multiple times for different agents
                        mock_pdf.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multiple_cv_generation_integration(self, cv_generator_config):
        """Test generation of multiple CVs."""
        with patch('app.services.cv_generator.ThisPersonDoesNotExistProvider.get_face_image', 
                  new_callable=AsyncMock) as mock_face:
            with patch('app.services.cv_generator.run_agent', 
                      new_callable=AsyncMock) as mock_run_agent:
                with patch('app.services.cv_generator.WeasyPrintPDFGenerator.generate_pdf', 
                          new_callable=AsyncMock) as mock_pdf:
                    with patch('app.services.cv_generator.update_index') as mock_index:
                        
                        # Setup mocks
                        mock_face.return_value = '<img src="test.jpg" />'
                        mock_run_agent.return_value = "<html><body><h1>Test CV</h1></body></html>"
                        mock_pdf.return_value = "/path/to/test.pdf"
                        
                        # Create service and generate multiple CVs
                        service = CVGeneratorFactory.create(cv_generator_config)
                        results = await service.generate_cvs(3)
                        
                        # Verify results
                        assert len(results) == 3
                        assert all(result.filename is not None for result in results)
                        assert all(result.file_path == "/path/to/test.pdf" for result in results)
                        
                        # Verify mocks were called correct number of times
                        assert mock_face.call_count == 3
                        # mock_run_agent may be called more times due to template generation and CV generation
                        assert mock_run_agent.call_count >= 3
                        assert mock_pdf.call_count == 3
                        mock_index.assert_called_once()


class TestImageValidationIntegration:
    """Test image validation integration."""
    
    @pytest.mark.asyncio
    async def test_image_validation_integration(self, cv_generator_config, sample_cv_template):
        """Test image validation integration with CV generation."""
        with patch('app.services.cv_generator.ThisPersonDoesNotExistProvider.get_face_image', 
                  new_callable=AsyncMock) as mock_face:
            with patch('app.services.cv_generator.OpenAICVContentGenerator.generate_cv_content', 
                      new_callable=AsyncMock) as mock_content:
                with patch('app.services.cv_generator.WeasyPrintPDFGenerator.generate_pdf', 
                          new_callable=AsyncMock) as mock_pdf:
                    with patch('app.services.cv_generator.run_agent') as mock_run_agent:
                        
                        # Setup mocks
                        mock_face.return_value = '<img src="test.jpg" />'
                        mock_content.return_value = "<html><body><h1>Test CV</h1></body></html>"
                        mock_pdf.return_value = "/path/to/test.pdf"
                        
                        # Mock agentic response
                        mock_run_agent.return_value = '''
                        {
                            "html_content": "<html><body><h1>Test CV</h1></body></html>",
                            "person_characteristics": {
                                "gender": "male",
                                "estimated_age": "30s",
                                "professional_appearance": true,
                                "overall_impression": "professional"
                            }
                        }
                        '''
                        
                        # Create service with image validation enabled
                        config = cv_generator_config
                        config.enable_image_validation = True
                        service = CVGeneratorFactory.create(config)
                        
                        result = await service._generate_single_cv(sample_cv_template, 0)
                        
                        # Verify agentic generation was called
                        mock_run_agent.assert_called_once()
                        
                        # Verify validation data is included in result
                        assert result.image_validation is not None
                        assert result.image_validation['is_valid'] is True
                        assert result.image_validation['confidence_score'] == 0.9
    
    @pytest.mark.asyncio
    async def test_image_validation_disabled_integration(self, cv_generator_config, sample_cv_template):
        """Test CV generation when image validation is disabled."""
        with patch('app.services.cv_generator.ThisPersonDoesNotExistProvider.get_face_image', 
                  new_callable=AsyncMock) as mock_face:
            with patch('app.services.cv_generator.OpenAICVContentGenerator.generate_cv_content', 
                      new_callable=AsyncMock) as mock_content:
                with patch('app.services.cv_generator.WeasyPrintPDFGenerator.generate_pdf', 
                          new_callable=AsyncMock) as mock_pdf:
                    
                    # Setup mocks
                    mock_face.return_value = '<img src="test.jpg" />'
                    mock_content.return_value = "<html><body><h1>Test CV</h1></body></html>"
                    mock_pdf.return_value = "/path/to/test.pdf"
                    
                    # Create service with image validation disabled
                    config = cv_generator_config
                    config.enable_image_validation = False
                    service = CVGeneratorFactory.create(config)
                    
                    result = await service._generate_single_cv(sample_cv_template, 0)
                    
                    # Verify validation was not called
                    assert result.image_validation is None


class TestErrorHandlingIntegration:
    """Test error handling integration."""
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_on_face_provider_failure(self, cv_generator_config, sample_cv_template):
        """Test graceful degradation when face provider fails."""
        with patch('app.services.cv_generator.ThisPersonDoesNotExistProvider.get_face_image', 
                  side_effect=Exception("Face provider failed")) as mock_face:
            with patch('app.services.cv_generator.OpenAICVContentGenerator.generate_cv_content', 
                      new_callable=AsyncMock) as mock_content:
                with patch('app.services.cv_generator.WeasyPrintPDFGenerator.generate_pdf', 
                          new_callable=AsyncMock) as mock_pdf:
                    
                    # Setup mocks
                    mock_content.return_value = "<html><body><h1>Test CV</h1></body></html>"
                    mock_pdf.return_value = "/path/to/test.pdf"
                    
                    service = CVGeneratorFactory.create(cv_generator_config)
                    
                    # Should raise exception due to face provider failure
                    with pytest.raises(Exception, match="Face provider failed"):
                        await service._generate_single_cv(sample_cv_template, 0)
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_on_pdf_generation_failure(self, cv_generator_config, sample_cv_template):
        """Test graceful degradation when PDF generation fails."""
        with patch('app.services.cv_generator.ThisPersonDoesNotExistProvider.get_face_image', 
                  new_callable=AsyncMock) as mock_face:
            with patch('app.services.cv_generator.OpenAICVContentGenerator.generate_cv_content', 
                      new_callable=AsyncMock) as mock_content:
                with patch('app.services.cv_generator.WeasyPrintPDFGenerator.generate_pdf', 
                          side_effect=Exception("PDF generation failed")) as mock_pdf:
                    
                    # Setup mocks
                    mock_face.return_value = '<img src="test.jpg" />'
                    mock_content.return_value = "<html><body><h1>Test CV</h1></body></html>"
                    
                    service = CVGeneratorFactory.create(cv_generator_config)
                    
                    # Should raise exception due to PDF generation failure
                    with pytest.raises(Exception, match="PDF generation failed"):
                        await service._generate_single_cv(sample_cv_template, 0)
    
    @pytest.mark.asyncio
    async def test_partial_failures_in_batch_generation(self, cv_generator_config):
        """Test handling of partial failures in batch generation."""
        with patch('app.services.cv_generator.ThisPersonDoesNotExistProvider.get_face_image', 
                  new_callable=AsyncMock) as mock_face:
            with patch('app.services.cv_generator.OpenAICVContentGenerator.generate_cv_content', 
                      new_callable=AsyncMock) as mock_content:
                with patch('app.services.cv_generator.WeasyPrintPDFGenerator.generate_pdf', 
                          new_callable=AsyncMock) as mock_pdf:
                    with patch('app.services.cv_generator.update_index') as mock_index:
                        
                        # Setup mocks with mixed success/failure
                        mock_face.return_value = '<img src="test.jpg" />'
                        mock_content.return_value = "<html><body><h1>Test CV</h1></body></html>"
                        mock_pdf.side_effect = [
                            "/path/to/test1.pdf",  # Success
                            Exception("PDF generation failed"),  # Failure
                            "/path/to/test3.pdf"   # Success
                        ]
                        
                        service = CVGeneratorFactory.create(cv_generator_config)
                        results = await service.generate_cvs(3)
                        
                        # Should have 2 successful results
                        assert len(results) == 2
                        assert results[0].file_path == "/path/to/test1.pdf"
                        assert results[1].file_path == "/path/to/test3.pdf"
                        
                        # Index should still be updated
                        mock_index.assert_called_once()
