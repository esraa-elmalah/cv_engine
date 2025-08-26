import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import faiss
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingConfig(BaseModel):
    """Configuration for embedding operations."""
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 10
    max_retries: int = 3
    timeout: int = 30


class Document(BaseModel):
    """Document model for embedding."""
    text: str
    metadata: Dict[str, Any] = {}
    source: Optional[str] = None


class EmbeddingResult(BaseModel):
    """Result of embedding operation."""
    embedding: List[float]
    document: Document
    model_used: str
    dimensions: int


class EmbeddingService:
    """Service for managing document embeddings with agentic capabilities."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client with proper error handling."""
        try:
            from openai import OpenAI
            # Get API key from settings
            api_key = settings.openai_api_key or settings.openai.api_key
            self.client = OpenAI(
                api_key=api_key,
                timeout=self.config.timeout
            )
            logger.info(f"Embedding service initialized with model: {self.config.model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise RuntimeError(f"Embedding service initialization failed: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a single text with retry logic and error handling."""
        try:
            if not self.client:
                self._initialize_client()
            
            logger.debug(f"Embedding text with model: {self.config.model}")
            
            # Use asyncio to run the blocking API call
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.embeddings.create(
                    input=text,
                    model=self.config.model
                )
            )
            
            embedding = response.data[0].embedding
            
            result = EmbeddingResult(
                embedding=embedding,
                document=Document(text=text),
                model_used=self.config.model,
                dimensions=len(embedding)
            )
            
            logger.debug(f"Successfully embedded text (dimensions: {len(embedding)})")
            return result
            
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def embed_batch(self, documents: List[Document]) -> List[EmbeddingResult]:
        """Embed a batch of documents efficiently."""
        try:
            if not self.client:
                self._initialize_client()
            
            logger.info(f"Embedding batch of {len(documents)} documents")
            
            # Process in batches for efficiency
            results = []
            for i in range(0, len(documents), self.config.batch_size):
                batch = documents[i:i + self.config.batch_size]
                texts = [doc.text for doc in batch]
                
                # Use asyncio to run the blocking API call
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.embeddings.create(
                        input=texts,
                        model=self.config.model
                    )
                )
                
                # Process results
                for j, embedding_data in enumerate(response.data):
                    result = EmbeddingResult(
                        embedding=embedding_data.embedding,
                        document=batch[j],
                        model_used=self.config.model,
                        dimensions=len(embedding_data.embedding)
                    )
                    results.append(result)
                
                logger.debug(f"Processed batch {i//self.config.batch_size + 1}")
            
            logger.info(f"Successfully embedded {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Failed to embed batch: {e}")
            raise
    
    async def embed_query(self, query: str) -> List[float]:
        """Embed a query for search operations."""
        try:
            result = await self.embed_text(query)
            return result.embedding
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise


class FAISSIndexManager:
    """Manager for FAISS index operations with agentic capabilities."""
    
    def __init__(self, index_path: str, dimensions: int = 1536):
        self.index_path = Path(index_path)
        self.dimensions = dimensions
        self.index = None
        self.documents = []
        self.embedding_service = EmbeddingService()
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize or load FAISS index."""
        try:
            if self.index_path.exists():
                logger.info(f"Loading existing FAISS index from {self.index_path}")
                self.index = faiss.read_index(str(self.index_path))
                self._load_documents()
            else:
                logger.info(f"Creating new FAISS index with {self.dimensions} dimensions")
                self.index = faiss.IndexFlatL2(self.dimensions)
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise RuntimeError(f"Index initialization failed: {e}")
    
    def _load_documents(self):
        """Load document metadata from disk."""
        try:
            docs_path = self.index_path.with_suffix('.docs.npz')
            if docs_path.exists():
                import pickle
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info(f"Loaded {len(self.documents)} documents from disk")
        except Exception as e:
            logger.warning(f"Failed to load documents: {e}")
            self.documents = []
    
    def _save_documents(self):
        """Save document metadata to disk."""
        try:
            docs_path = self.index_path.with_suffix('.docs.npz')
            import pickle
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            logger.debug(f"Saved {len(self.documents)} documents to disk")
        except Exception as e:
            logger.error(f"Failed to save documents: {e}")
    
    async def add_documents(self, documents: List[Document]):
        """Add documents to the index with embeddings."""
        try:
            logger.info(f"Adding {len(documents)} documents to index")
            
            # Generate embeddings
            embedding_results = await self.embedding_service.embed_batch(documents)
            
            # Prepare vectors for FAISS
            vectors = np.array([result.embedding for result in embedding_results]).astype("float32")
            
            # Add to index
            self.index.add(vectors)
            
            # Store documents
            for result in embedding_results:
                self.documents.append(result.document)
            
            logger.info(f"Successfully added {len(documents)} documents to index")
            
        except Exception as e:
            logger.error(f"Failed to add documents to index: {e}")
            raise
    
    async def search(self, query: str, k: int = 3) -> List[Document]:
        """Search the index for similar documents."""
        try:
            # Check if index has any documents
            if len(self.documents) == 0:
                logger.info("Index is empty, returning empty results")
                return []
            
            # Embed the query
            query_embedding = await self.embedding_service.embed_query(query)
            query_vector = np.array([query_embedding]).astype("float32")
            
            # Search the index
            distances, indices = self.index.search(query_vector, k)
            
            # Return matching documents
            results = []
            if len(indices) > 0 and len(indices[0]) > 0:
                for i in indices[0]:
                    if i < len(self.documents):
                        results.append(self.documents[i])
            
            logger.info(f"Found {len(results)} documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search index: {e}")
            raise
    
    def save(self):
        """Save the index to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save documents
            self._save_documents()
            
            logger.info(f"Saved index to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal,
            "dimensions": self.dimensions,
            "index_path": str(self.index_path)
        }


# Global instances for easy access
embedding_service = EmbeddingService()

# Use the config helper to get the correct index path
from app.config import get_index_dir
index_manager = FAISSIndexManager(
    index_path=str(get_index_dir() / "faiss_index.bin")
)
