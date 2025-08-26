import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# Helper functions for directory paths
def get_base_dir() -> Path:
    """Get the base directory for the project (backend directory)."""
    return Path(__file__).parent.parent


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_base_dir() / "data"


def get_cvs_dir() -> Path:
    """Get the CVs directory."""
    return get_data_dir() / "cvs"


def get_index_dir() -> Path:
    """Get the index directory."""
    return get_data_dir() / "index"


class DatabaseConfig(BaseModel):
    """Database configuration."""
    url: str = Field(default="sqlite:///./cv_engine.db")
    echo: bool = Field(default=False)


class OpenAIConfig(BaseModel):
    """OpenAI configuration."""
    api_key: str = Field(default="")
    model: str = Field(default="gpt-4.1-mini")
    embedding_model: str = Field(default="text-embedding-3-small")


class CVGeneratorConfig(BaseModel):
    """CV Generator specific configuration."""
    output_dir: str = Field(default_factory=lambda: str(get_cvs_dir()))
    face_image_url: str = Field(default="https://thispersondoesnotexist.com")
    face_image_timeout: int = Field(default=10)
    face_image_style: str = Field(
        default="border-radius:50%; width:120px; height:120px; object-fit:cover; float:left; margin-right:20px;"
    )
    agent_model: str = Field(default="gpt-4.1-mini")
    max_retries: int = Field(default=3)
    batch_size: int = Field(default=5)
    enable_image_validation: bool = Field(default=True, description="Enable image validation")
    max_image_validation_attempts: int = Field(default=3, description="Maximum attempts for image validation")
    min_validation_confidence: float = Field(default=0.7, description="Minimum confidence score for image validation")
    enable_age_filtering: bool = Field(default=True, description="Filter out child/elderly faces")
    max_face_attempts: int = Field(default=5, description="Maximum attempts to find appropriate face")
    enable_agentic_workflow: bool = Field(default=True, description="Use agentic workflow for cost efficiency")
    face_analysis_model: str = Field(default="gpt-4o-mini", description="Model for face analysis (cheaper)")
    cv_generation_model: str = Field(default="gpt-4.1-mini", description="Model for CV generation")


class IndexConfig(BaseModel):
    """Index configuration."""
    index_dir: str = Field(default_factory=lambda: str(get_index_dir()))
    cvs_dir: str = Field(default_factory=lambda: str(get_cvs_dir()))
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    batch_size: int = Field(default=10)
    enable_incremental: bool = Field(default=True)
    processed_file: str = Field(default="processed_files.txt")


class EmbeddingConfig(BaseModel):
    """Embedding configuration."""
    model: str = Field(default="text-embedding-3-small")
    dimensions: int = Field(default=1536)
    batch_size: int = Field(default=10)
    max_retries: int = Field(default=3)
    timeout: int = Field(default=30)


class RAGConfig(BaseModel):
    """RAG configuration."""
    default_k: int = Field(default=3)
    max_k: int = Field(default=10)
    cache_results: bool = Field(default=True)
    cache_ttl: int = Field(default=300)
    enable_reranking: bool = Field(default=False)
    reranking_model: str = Field(default="gpt-4o-mini")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file: Optional[str] = Field(default=None)


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: str = Field(default="development")
    debug: bool = Field(default=True)
    
    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    
    # Database
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    
    # OpenAI
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    
    # Direct environment variable mapping
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    
    # CV Generator
    cv_generator: CVGeneratorConfig = Field(default_factory=CVGeneratorConfig)
    
    # Index
    index: IndexConfig = Field(default_factory=IndexConfig)
    
    # Embedding
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    
    # RAG
    rag: RAGConfig = Field(default_factory=RAGConfig)
    
    # Logging
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False
        env_file_encoding = "utf-8"
        
        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            return (
                init_settings,
                env_settings,
                file_secret_settings,
            )


# Global settings instance
settings = Settings()


# Ensure directories exist
def ensure_directories():
    """Ensure all required directories exist."""
    get_cvs_dir().mkdir(parents=True, exist_ok=True)
    get_index_dir().mkdir(parents=True, exist_ok=True)
