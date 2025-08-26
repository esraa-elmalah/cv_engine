import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field

from app.config import settings, get_index_dir, get_cvs_dir
from app.services.embedder import Document, FAISSIndexManager

logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    """Result of indexing operation."""
    files_processed: int
    documents_added: int
    errors: List[str]
    total_documents: int


class IndexConfig(BaseModel):
    """Configuration for indexing operations."""
    index_dir: str = Field(default_factory=lambda: str(get_index_dir()))
    cvs_dir: str = Field(default_factory=lambda: str(get_cvs_dir()))
    batch_size: int = 10
    max_retries: int = 3
    enable_incremental: bool = True
    processed_file: str = "processed_files.txt"


class FileManager:
    """Manages file tracking for incremental indexing."""
    
    def __init__(self, processed_file_path: Path):
        self.processed_file_path = processed_file_path
        self.processed_files = self._load_processed_files()
    
    def _load_processed_files(self) -> set:
        """Load list of processed files from disk."""
        try:
            if self.processed_file_path.exists():
                files = set(self.processed_file_path.read_text().splitlines())
                logger.info(f"Loaded {len(files)} processed files from {self.processed_file_path}")
                return files
            else:
                logger.info("No processed files record found, starting fresh")
                return set()
        except Exception as e:
            logger.warning(f"Failed to load processed files: {e}")
            return set()
    
    def _save_processed_files(self):
        """Save list of processed files to disk."""
        try:
            self.processed_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.processed_file_path.write_text("\n".join(sorted(self.processed_files)))
            logger.debug(f"Saved {len(self.processed_files)} processed files")
        except Exception as e:
            logger.error(f"Failed to save processed files: {e}")
            raise
    
    def is_processed(self, file_path: str) -> bool:
        """Check if a file has been processed."""
        return file_path in self.processed_files
    
    def mark_processed(self, file_path: str):
        """Mark a file as processed."""
        self.processed_files.add(file_path)
    
    def get_unprocessed_files(self, directory: Path) -> List[Path]:
        """Get list of unprocessed files in directory."""
        unprocessed = []
        for file_path in directory.glob("*.pdf"):
            if not self.is_processed(str(file_path)):
                unprocessed.append(file_path)
        return unprocessed
    
    def commit_changes(self):
        """Save changes to processed files."""
        self._save_processed_files()


class DocumentProcessor:
    """Processes PDF documents for indexing."""
    
    def __init__(self, config: IndexConfig):
        self.config = config
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def process_pdf(self, pdf_path: Path) -> List[Document]:
        """Process a single PDF file and extract documents."""
        try:
            logger.debug(f"Processing PDF: {pdf_path}")
            
            # Use asyncio to run the blocking PDF processing
            loop = asyncio.get_event_loop()
            documents = await loop.run_in_executor(
                None,
                self._extract_pdf_content,
                pdf_path
            )
            
            # Convert to Document objects
            docs = []
            for i, content in enumerate(documents):
                doc = Document(
                    text=content.page_content,
                    metadata={
                        "source": str(pdf_path),
                        "page": i + 1,
                        "filename": pdf_path.name,
                        **content.metadata
                    },
                    source=str(pdf_path)
                )
                docs.append(doc)
            
            logger.debug(f"Extracted {len(docs)} documents from {pdf_path}")
            return docs
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            raise
    
    def _extract_pdf_content(self, pdf_path: Path) -> List:
        """Extract content from PDF using LangChain."""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(pdf_path))
            return loader.load()
        except Exception as e:
            logger.error(f"Failed to extract PDF content from {pdf_path}: {e}")
            raise
    
    async def process_batch(self, pdf_files: List[Path]) -> List[Document]:
        """Process a batch of PDF files."""
        try:
            logger.info(f"Processing batch of {len(pdf_files)} PDF files")
            
            # Process files concurrently
            tasks = [self.process_pdf(pdf_path) for pdf_path in pdf_files]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect successful results
            all_documents = []
            errors = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    errors.append(f"Failed to process {pdf_files[i]}: {result}")
                    logger.error(f"Batch processing error for {pdf_files[i]}: {result}")
                else:
                    all_documents.extend(result)
            
            if errors:
                logger.warning(f"Batch processing completed with {len(errors)} errors")
            
            logger.info(f"Successfully processed {len(all_documents)} documents from batch")
            return all_documents
            
        except Exception as e:
            logger.error(f"Failed to process batch: {e}")
            raise


class IndexService:
    """Service for managing CV indexing with agentic capabilities."""
    
    def __init__(self, config: Optional[IndexConfig] = None):
        self.config = config or IndexConfig()
        self.index_manager = None
        self.file_manager = None
        self.document_processor = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components."""
        try:
            # Initialize paths
            index_dir = Path(self.config.index_dir)
            cvs_dir = Path(self.config.cvs_dir)
            processed_file = index_dir / self.config.processed_file
            
            # Initialize components
            self.index_manager = FAISSIndexManager(
                index_path=str(index_dir / "faiss_index.bin")
            )
            self.file_manager = FileManager(processed_file)
            self.document_processor = DocumentProcessor(self.config)
            
            logger.info("Index service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize index service: {e}")
            raise RuntimeError(f"Index service initialization failed: {e}")
    
    async def update_index(self) -> IndexingResult:
        """Incrementally update the index with new CVs."""
        try:
            logger.info("Starting incremental index update")
            
            cvs_dir = Path(self.config.cvs_dir)
            if not cvs_dir.exists():
                logger.warning(f"CVs directory does not exist: {cvs_dir}")
                return IndexingResult(
                    files_processed=0,
                    documents_added=0,
                    errors=["CVs directory does not exist"],
                    total_documents=0
                )
            
            # Get unprocessed files
            unprocessed_files = self.file_manager.get_unprocessed_files(cvs_dir)
            
            if not unprocessed_files:
                logger.info("No new CVs to index")
                return IndexingResult(
                    files_processed=0,
                    documents_added=0,
                    errors=[],
                    total_documents=self.index_manager.get_stats()["total_documents"]
                )
            
            logger.info(f"Found {len(unprocessed_files)} new CVs to index")
            
            # Process files in batches
            all_documents = []
            errors = []
            
            for i in range(0, len(unprocessed_files), self.config.batch_size):
                batch = unprocessed_files[i:i + self.config.batch_size]
                
                try:
                    batch_documents = await self.document_processor.process_batch(batch)
                    all_documents.extend(batch_documents)
                    
                    # Mark files as processed
                    for file_path in batch:
                        self.file_manager.mark_processed(str(file_path))
                    
                    logger.debug(f"Processed batch {i//self.config.batch_size + 1}")
                    
                except Exception as e:
                    error_msg = f"Failed to process batch {i//self.config.batch_size + 1}: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            # Add documents to index
            if all_documents:
                try:
                    await self.index_manager.add_documents(all_documents)
                    self.index_manager.save()
                    logger.info(f"Successfully added {len(all_documents)} documents to index")
                except Exception as e:
                    error_msg = f"Failed to add documents to index: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            # Commit file tracking changes
            self.file_manager.commit_changes()
            
            # Get final stats
            stats = self.index_manager.get_stats()
            
            result = IndexingResult(
                files_processed=len(unprocessed_files),
                documents_added=len(all_documents),
                errors=errors,
                total_documents=stats["total_documents"]
            )
            
            logger.info(f"Index update completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to update index: {e}")
            raise
    
    async def rebuild_index(self) -> IndexingResult:
        """Rebuild the entire index from scratch."""
        try:
            logger.info("Starting full index rebuild")
            
            # Reset file tracking
            self.file_manager.processed_files.clear()
            
            # Update index (will process all files)
            result = await self.update_index()
            
            logger.info(f"Index rebuild completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")
            raise
    
    async def search(self, query: str, k: int = 3) -> List[Document]:
        """Search the index for relevant documents."""
        try:
            logger.debug(f"Searching index for query: {query}")
            results = await self.index_manager.search(query, k)
            logger.info(f"Found {len(results)} documents for query")
            return results
        except Exception as e:
            logger.error(f"Failed to search index: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            index_stats = self.index_manager.get_stats()
            file_stats = {
                "processed_files": len(self.file_manager.processed_files),
                "index_dir": self.config.index_dir,
                "cvs_dir": self.config.cvs_dir
            }
            return {**index_stats, **file_stats}
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            raise


# Global service instance
index_service = IndexService()


# Backward compatibility function
async def update_index():
    """Update the index with new CVs (backward compatibility)."""
    try:
        result = await index_service.update_index()
        logger.info(f"Index update completed: {result}")
        return result
    except Exception as e:
        logger.error(f"Index update failed: {e}")
        raise
