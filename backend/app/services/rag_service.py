import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.services.index_service import index_service
from app.services.embedder import Document

logger = logging.getLogger(__name__)


class RAGConfig(BaseModel):
    """Configuration for RAG operations."""
    default_k: int = 3
    max_k: int = 10
    cache_results: bool = True
    cache_ttl: int = 300  # 5 minutes
    enable_reranking: bool = False
    reranking_model: str = "gpt-4o-mini"


class SearchResult(BaseModel):
    """Result of a search operation."""
    document: Document
    score: float
    rank: int
    metadata: Dict[str, Any] = {}


class RAGResponse(BaseModel):
    """Response from RAG operations."""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: float
    model_used: str


class RAGService:
    """Service for RAG (Retrieval-Augmented Generation) operations with agentic capabilities."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.cache = {}  # Simple in-memory cache
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialize the cache system."""
        if self.config.cache_results:
            logger.info("RAG service initialized with caching enabled")
        else:
            logger.info("RAG service initialized without caching")
    
    def _get_cache_key(self, query: str, k: int) -> str:
        """Generate cache key for query."""
        return f"{query.lower().strip()}:{k}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[RAGResponse]:
        """Get cached result if available and not expired."""
        if not self.config.cache_results:
            return None
        
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            # Simple TTL check (in production, use Redis or similar)
            if cached_data.get("expires_at", 0) > asyncio.get_event_loop().time():
                logger.debug(f"Cache hit for query: {cache_key}")
                return cached_data["response"]
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, response: RAGResponse):
        """Cache the search result."""
        if not self.config.cache_results:
            return
        
        import time
        self.cache[cache_key] = {
            "response": response,
            "expires_at": time.time() + self.config.cache_ttl
        }
        logger.debug(f"Cached result for query: {cache_key}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def search(self, query: str, k: Optional[int] = None) -> RAGResponse:
        """Search the index for relevant documents with caching and error handling."""
        try:
            import time
            start_time = time.time()
            
            # Validate parameters
            k = min(k or self.config.default_k, self.config.max_k)
            
            # Check cache first
            cache_key = self._get_cache_key(query, k)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            logger.debug(f"Searching index for query: '{query}' with k={k}")
            
            # Perform search
            documents = await index_service.search(query, k)
            
            # Convert to search results
            results = []
            if documents:  # Check if documents exist
                for i, doc in enumerate(documents):
                    result = SearchResult(
                        document=doc,
                        score=1.0 - (i / len(documents)),  # Simple ranking
                        rank=i + 1,
                        metadata=doc.metadata
                    )
                    results.append(result)
            
            # Optional reranking
            if self.config.enable_reranking and results:
                results = await self._rerank_results(query, results)
            
            # Calculate search time
            search_time = time.time() - start_time
            
            # Create response
            response = RAGResponse(
                query=query,
                results=results,
                total_results=len(results),
                search_time=search_time,
                model_used="faiss-index"
            )
            
            # Cache result
            self._cache_result(cache_key, response)
            
            logger.info(f"Search completed: {len(results)} results in {search_time:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"Failed to search index: {e}")
            raise
    
    async def _rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank search results using LLM for better relevance."""
        try:
            logger.debug("Reranking search results")
            
            # Create reranking prompt
            prompt = f"""
Given the query: "{query}"

Rank the following documents by relevance (1 = most relevant):

"""
            
            for i, result in enumerate(results):
                prompt += f"{i+1}. {result.document.text[:200]}...\n"
            
            prompt += "\nReturn ONLY a JSON array with the new ranking (e.g., [3,1,2] means document 3 is most relevant):"
            
            # Use LLM for reranking
            from openai import OpenAI
            # Get API key from settings
            api_key = settings.openai_api_key or settings.openai.api_key
            client = OpenAI(api_key=api_key)
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=self.config.reranking_model,
                    messages=[
                        {"role": "system", "content": "You are an expert at ranking document relevance. Return only JSON arrays."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
            )
            
            # Parse ranking
            import json
            import re
            
            content = response.choices[0].message.content
            ranking_match = re.search(r'\[.*\]', content)
            
            if ranking_match:
                new_ranking = json.loads(ranking_match.group())
                reranked_results = []
                
                for new_rank in new_ranking:
                    if 0 <= new_rank - 1 < len(results):
                        result = results[new_rank - 1]
                        result.rank = len(reranked_results) + 1
                        result.score = 1.0 - (len(reranked_results) / len(results))
                        reranked_results.append(result)
                
                logger.debug(f"Reranked {len(reranked_results)} results")
                return reranked_results
            
            logger.warning("Failed to parse reranking response, using original ranking")
            return results
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, using original ranking")
            return results
    
    async def query_with_context(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Query the index and return results with context for LLM."""
        try:
            # Search for relevant documents
            search_response = await self.search(query, k)
            
            # Prepare context for LLM
            context_parts = []
            for result in search_response.results:
                context_parts.append(f"Document {result.rank} (Score: {result.score:.3f}):\n{result.document.text}")
            
            context = "\n\n".join(context_parts)
            
            return {
                "query": query,
                "context": context,
                "documents": [result.document for result in search_response.results],
                "search_stats": {
                    "total_results": search_response.total_results,
                    "search_time": search_response.search_time,
                    "model_used": search_response.model_used
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to query with context: {e}")
            raise
    
    async def chat_with_rag(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Chat with RAG capabilities - retrieve relevant docs and generate response."""
        try:
            # Get relevant documents
            rag_context = await self.query_with_context(message)
            
            # Generate response using LLM
            context = rag_context['context']
            if not context.strip():
                system_prompt = (
                    "You are a helpful assistant answering questions about candidate CVs. "
                    "Currently there are no CVs in the database. "
                    "Please inform the user that the database is empty and suggest they generate some CVs first."
                )
            else:
                system_prompt = (
                    "You are a helpful assistant answering questions about candidate CVs. "
                    "Use the retrieved CV context to answer precisely. "
                    "If unsure, say you don't know.\n\n"
                    f"Context:\n{context}"
                )
            
            from openai import OpenAI
            # Get API key from settings
            api_key = settings.openai_api_key or settings.openai.api_key
            client = OpenAI(api_key=api_key)
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message},
                    ],
                )
            )
            
            assistant_reply = response.choices[0].message.content
            
            return {
                "reply": assistant_reply,
                "query": message,
                "session_id": session_id,
                "search_stats": rag_context["search_stats"],
                "documents_used": [doc.source for doc in rag_context["documents"] if doc.source]
            }
            
        except Exception as e:
            logger.error(f"Failed to chat with RAG: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG service statistics."""
        return {
            "cache_enabled": self.config.cache_results,
            "cache_size": len(self.cache),
            "default_k": self.config.default_k,
            "max_k": self.config.max_k,
            "reranking_enabled": self.config.enable_reranking
        }
    
    def clear_cache(self):
        """Clear the search cache."""
        self.cache.clear()
        logger.info("RAG service cache cleared")


# Global service instance
rag_service = RAGService()


# Backward compatibility functions
async def query_index(question: str, k: int = 3):
    """Query the index for relevant documents (backward compatibility)."""
    try:
        response = await rag_service.search(question, k)
        return [result.document for result in response.results]
    except Exception as e:
        logger.error(f"Failed to query index: {e}")
        raise
