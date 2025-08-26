from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.api import router
from app.config import settings, ensure_directories
from app.utils.logger import get_logger

# Load environment variables from .env file
load_dotenv()

# Setup logging
logger = get_logger(__name__)

# Ensure required directories exist
ensure_directories()

# Create FastAPI app
app = FastAPI(
    title="CV Engine API",
    description="A scalable CV generation and management system",
    version="1.0.0",
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting CV Engine API...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down CV Engine API...")

@app.get("/")
async def root():
    return {
        "message": "CV Engine API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z"
    }