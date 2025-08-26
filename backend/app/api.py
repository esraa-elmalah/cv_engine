from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, ValidationError
from typing import List, Optional, Dict, Any
import logging

from app.services.cv_service import CVService
from app.services.image_validator import ImageValidationService
from app.services.rag_service import rag_service
from app.services.index_service import index_service
from app.config import settings
from app.models.cv_models import CVTemplate

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize services
cv_service = CVService()
image_validator = ImageValidationService()

# Request/Response models
class GenerateCVsRequest(BaseModel):
    count: int = 1

class GenerateCVsResponse(BaseModel):
    success: bool
    message: str
    count: int
    cvs: List[Dict[str, Any]]

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    success: bool
    query: str
    results: List[Dict[str, Any]]
    search_stats: Dict[str, Any]

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    success: bool
    reply: str
    session_id: Optional[str]
    search_stats: Dict[str, Any]
    documents_used: List[str]

class ValidateImageRequest(BaseModel):
    image_data: str  # Base64 encoded image

class ValidateImageResponse(BaseModel):
    success: bool
    is_valid: bool
    characteristics: Dict[str, Any]
    confidence: float
    message: str

# CV Generation endpoints
@router.post("/generate-cvs", response_model=GenerateCVsResponse)
async def generate_cvs(request: GenerateCVsRequest):
    """Generate CVs with random templates and validated images."""
    try:
        logger.info(f"Generating {request.count} CVs")
        result = await cv_service.generate_cvs(request.count)
        
        # Convert GeneratedCV objects to dictionaries
        cv_dicts = []
        for cv in result:
            cv_dict = {
                "filename": cv.filename,
                "file_path": cv.file_path,
                "template": {
                    "role": cv.template.role,
                    "level": cv.template.level,
                    "industry": cv.template.industry,
                    "skills": cv.template.skills,
                    "experience_years": cv.template.experience_years
                },
                "generated_at": cv.generated_at.isoformat(),
                "image_validation": cv.image_validation
            }
            cv_dicts.append(cv_dict)
        
        return GenerateCVsResponse(
            success=True,
            message=f"Successfully generated {len(result)} CVs",
            count=len(result),
            cvs=cv_dicts
        )
    except Exception as e:
        logger.error(f"Error generating CVs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-cv-with-template", response_model=GenerateCVsResponse)
async def generate_cv_with_template(request: Dict[str, Any]):
    """Generate a single CV with a specific template."""
    try:
        # Create template from request
        template = CVTemplate(**request)
        
        # Generate CV with template
        result = await cv_service.generate_cv_with_template(template)
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to generate CV with template")
        
        # Convert to dictionary format
        cv_dict = {
            "filename": result.filename,
            "file_path": result.file_path,
            "template": {
                "role": result.template.role,
                "level": result.template.level,
                "industry": result.template.industry,
                "skills": result.template.skills,
                "experience_years": result.template.experience_years
            },
            "generated_at": result.generated_at.isoformat(),
            "image_validation": result.image_validation
        }
        
        return GenerateCVsResponse(
            success=True,
            message="Successfully generated CV with template",
            count=1,
            cvs=[cv_dict]
        )
    except ValidationError as e:
        logger.error(f"Validation error generating CV with template: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating CV with template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates")
async def get_templates():
    """Get available CV templates."""
    try:
        templates = await cv_service.get_available_templates()
        return {
            "success": True,
            "templates": templates
        }
    except Exception as e:
        logger.error(f"Error getting templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates/random")
async def get_random_templates(count: int = 5):
    """Get random CV templates generated by LLM."""
    try:
        templates = await cv_service.get_random_templates(count)
        return {
            "success": True,
            "templates": templates
        }
    except Exception as e:
        logger.error(f"Error getting random templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Image Validation endpoints
@router.post("/validate-image", response_model=ValidateImageResponse)
async def validate_image(request: ValidateImageRequest):
    """Validate if an image is appropriate for a CV."""
    try:
        result = await image_validator.validate_image(request.image_data)
        
        return ValidateImageResponse(
            success=True,
            is_valid=result.is_valid,
            characteristics=result.characteristics,
            confidence=result.confidence,
            message=result.message
        )
    except Exception as e:
        logger.error(f"Error validating image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate-image-upload")
async def validate_image_upload(file: UploadFile = File(...)):
    """Validate uploaded image file."""
    try:
        # Read file content
        content = await file.read()
        
        # Convert to base64
        import base64
        image_data = base64.b64encode(content).decode('utf-8')
        
        # Validate image
        result = await image_validator.validate_image(image_data)
        
        return {
            "success": True,
            "is_valid": result.is_valid,
            "characteristics": result.characteristics,
            "confidence": result.confidence,
            "message": result.message
        }
    except Exception as e:
        logger.error(f"Error validating uploaded image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# RAG and Search endpoints
@router.post("/query", response_model=QueryResponse)
async def query_cvs(request: QueryRequest):
    """Search CVs using RAG."""
    try:
        result = await rag_service.search(request.question)
        
        # Convert results to the expected format
        results_data = []
        for result_item in result.results:
            results_data.append({
                "content": result_item.document.text,
                "score": result_item.score,
                "rank": result_item.rank,
                "metadata": result_item.metadata
            })
        
        search_stats = {
            "total_results": result.total_results,
            "search_time": result.search_time,
            "model_used": result.model_used
        }
        
        return QueryResponse(
            success=True,
            query=request.question,
            results=results_data,
            search_stats=search_stats
        )
    except Exception as e:
        logger.error(f"Error querying CVs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat_with_cvs(request: ChatRequest):
    """Chat with CVs using RAG."""
    try:
        result = await rag_service.chat_with_rag(request.message, request.session_id)
        
        return ChatResponse(
            success=True,
            reply=result["reply"],
            session_id=result.get("session_id"),
            search_stats=result.get("search_stats", {}),
            documents_used=result.get("documents_used", [])
        )
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Index Management endpoints
@router.get("/index/stats")
async def get_index_stats():
    """Get index statistics."""
    try:
        stats = index_service.get_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting index stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/index/rebuild")
async def rebuild_index():
    """Rebuild the entire index."""
    try:
        result = await index_service.rebuild_index()
        return {
            "success": True,
            "message": "Index rebuilt successfully",
            "stats": result
        }
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# RAG Management endpoints
@router.get("/rag/stats")
async def get_rag_stats():
    """Get RAG service statistics."""
    try:
        stats = rag_service.get_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag/clear-cache")
async def clear_rag_cache():
    """Clear RAG service cache."""
    try:
        rag_service.clear_cache()
        return {
            "success": True,
            "message": "RAG cache cleared successfully"
        }
    except Exception as e:
        logger.error(f"Error clearing RAG cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System endpoints
@router.get("/stats")
async def get_system_stats():
    """Get overall system statistics."""
    try:
        return {
            "success": True,
            "system": {
                "environment": settings.environment,
                "debug": settings.debug,
                "version": "1.0.0"
            },
            "index": index_service.get_stats(),
            "rag": rag_service.get_stats()
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
