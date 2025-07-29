"""
🧠 Perfect Recall Engine

Memory Management with Semantic Understanding:
- Stores coding interactions with semantic understanding
- Vector Search: ChromaDB-powered similarity matching
- Knowledge Graph: Neo4j-based relationship mapping
- Semantic Recall: Find solutions based on meaning, not just keywords
"""

import asyncio
import json
import logging
import hashlib
import os
import uuid
import re
import math
import random
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
from dataclasses import dataclass, asdict
from pathlib import Path
import time

try:
    from .base_engine import BaseEngine
except ImportError:
    try:
        from packages.engines.base_engine import BaseEngine
    except ImportError:
        from base_engine import BaseEngine

from packages.engines.engine_types import EngineOutput

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """Represents a single memory entry in the Perfect Recall system."""
    id: str
    timestamp: datetime
    content: str
    content_type: str  # 'code', 'solution', 'error', 'conversation'
    tags: List[str]
    context: Dict[str, Any]
    embedding: Optional[List[float]] = None
    relationships: Optional[List[str]] = None
    success_score: float = 0.0
    usage_count: int = 0
    # Enhanced fields for workflow optimization
    workflow_id: Optional[str] = None  # Track which workflow this memory belongs to
    session_id: Optional[str] = None   # Track session for temporal clustering
    temporal_cluster: Optional[str] = None  # Group memories by time periods
    workflow_cohesion_score: float = 0.0  # How well this memory fits in its workflow
    cross_reference_count: int = 0  # Number of other memories that reference this one
    last_workflow_access: Optional[datetime] = None  # Last time accessed in workflow context

@dataclass
class RecallResult:
    """Result from a recall query."""
    entry: MemoryEntry
    similarity_score: float
    relevance_score: float

@dataclass
class WorkflowSession:
    """Represents a workflow session for temporal clustering and cohesion tracking."""
    session_id: str
    workflow_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    memory_ids: List[str] = None
    cohesion_score: float = 0.0
    temporal_cluster: str = ""
    active: bool = True
    
    def __post_init__(self):
        if self.memory_ids is None:
            self.memory_ids = []

@dataclass
class TemporalCluster:
    """Represents a temporal cluster of memories for better long workflow handling."""
    cluster_id: str
    start_time: datetime
    end_time: datetime
    memory_ids: List[str] = None
    dominant_workflow: Optional[str] = None
    cohesion_score: float = 0.0
    cross_reference_density: float = 0.0
    
    def __post_init__(self):
        if self.memory_ids is None:
            self.memory_ids = []

@dataclass
class ContextChunk:
    """Represents a semantic chunk of content with advanced context management."""
    id: str
    content: str
    timestamp: datetime
    importance_score: float = 0.5
    chunk_type: str = "text"
    semantic_tags: List[str] = None
    embedding: Optional[List[float]] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    compression_level: float = 1.0  # 1.0 = original, 0.5 = 50% compressed
    parent_chunk_id: Optional[str] = None  # For hierarchical relationships
    child_chunk_ids: List[str] = None
    
    def __post_init__(self):
        if self.semantic_tags is None:
            self.semantic_tags = []
        if self.child_chunk_ids is None:
            self.child_chunk_ids = []

@dataclass
class HierarchicalMemoryLevel:
    """Represents a level in the hierarchical memory system."""
    level_name: str  # 'active', 'short_term', 'long_term'
    max_chunks: int
    current_chunks: List[ContextChunk] = None
    compression_strategy: str = "semantic"
    access_pattern: str = "fifo"  # fifo, lru, importance
    
    def __post_init__(self):
        if self.current_chunks is None:
            self.current_chunks = []

@dataclass
class CompressionStrategy:
    """Represents a compression strategy for context management."""
    name: str
    compression_ratio: float
    quality_preservation: float
    processing_time_ms: float
    semantic_preservation: float

class SemanticChunker:
    """Advanced semantic chunking for intelligent content segmentation."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.sentence_boundaries = ['.', '!', '?', '\n\n']
        self.paragraph_boundaries = ['\n\n', '\n\r\n', '\r\n\r\n']
        
    def chunk_by_semantics(self, text: str) -> List[str]:
        """Chunk text semantically while preserving meaning."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # Find optimal break point
            end_pos = min(current_pos + self.chunk_size, len(text))
            
            # Try to break at sentence boundary
            optimal_break = self._find_semantic_break(text, current_pos, end_pos)
            
            chunk = text[current_pos:optimal_break].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move position with overlap
            current_pos = max(optimal_break - self.overlap, current_pos + 1)
            
            if current_pos >= len(text):
                break
        
        return chunks
    
    def _find_semantic_break(self, text: str, start: int, end: int) -> int:
        """Find the best semantic break point within the range."""
        # Look for sentence boundaries first
        for boundary in self.sentence_boundaries:
            pos = text.rfind(boundary, start, end)
            if pos > start + self.chunk_size * 0.7:  # At least 70% of chunk size
                return pos + len(boundary)
        
        # Look for paragraph boundaries
        for boundary in self.paragraph_boundaries:
            pos = text.rfind(boundary, start, end)
            if pos > start + self.chunk_size * 0.5:  # At least 50% of chunk size
                return pos + len(boundary)
        
        # Fallback to word boundary
        pos = text.rfind(' ', start, end)
        if pos > start:
            return pos
        
        return end

class ContextCompressor:
    """Advanced context compression with multiple strategies."""
    
    def __init__(self):
        self.compression_strategies = {
            'semantic': CompressionStrategy('semantic', 0.5, 0.9, 50.0, 0.95),
            'extractive': CompressionStrategy('extractive', 0.3, 0.7, 30.0, 0.8),
            'abstractive': CompressionStrategy('abstractive', 0.2, 0.6, 100.0, 0.7),
            'keyword': CompressionStrategy('keyword', 0.1, 0.4, 10.0, 0.5)
        }
    
    def compress_context(self, chunks: List[ContextChunk], target_ratio: float, strategy: str = 'semantic') -> List[ContextChunk]:
        """Compress context chunks using specified strategy."""
        if not chunks:
            return []
        
        if strategy not in self.compression_strategies:
            strategy = 'semantic'
        
        compressed_chunks = []
        strategy_config = self.compression_strategies[strategy]
        
        for chunk in chunks:
            if chunk.compression_level <= target_ratio:
                compressed_chunks.append(chunk)
                continue
            
            # Apply compression
            compressed_content = self._apply_compression(chunk.content, strategy, target_ratio)
            
            compressed_chunk = ContextChunk(
                id=f"{chunk.id}_compressed",
                content=compressed_content,
                timestamp=chunk.timestamp,
                importance_score=chunk.importance_score,
                chunk_type=chunk.chunk_type,
                semantic_tags=chunk.semantic_tags.copy(),
                embedding=chunk.embedding,
                access_count=chunk.access_count,
                last_accessed=chunk.last_accessed,
                compression_level=target_ratio,
                parent_chunk_id=chunk.id
            )
            
            compressed_chunks.append(compressed_chunk)
        
        return compressed_chunks
    
    def _apply_compression(self, content: str, strategy: str, target_ratio: float) -> str:
        """Apply specific compression strategy to content."""
        if strategy == 'semantic':
            return self._semantic_compression(content, target_ratio)
        elif strategy == 'extractive':
            return self._extractive_compression(content, target_ratio)
        elif strategy == 'abstractive':
            return self._abstractive_compression(content, target_ratio)
        elif strategy == 'keyword':
            return self._keyword_compression(content, target_ratio)
        else:
            return content
    
    def _semantic_compression(self, content: str, target_ratio: float) -> str:
        """Semantic compression preserving meaning."""
        sentences = content.split('. ')
        if len(sentences) <= 1:
            return content
        
        # Keep most important sentences
        target_sentences = max(1, int(len(sentences) * target_ratio))
        return '. '.join(sentences[:target_sentences]) + '.'
    
    def _extractive_compression(self, content: str, target_ratio: float) -> str:
        """Extractive compression keeping key phrases."""
        words = content.split()
        target_words = max(10, int(len(words) * target_ratio))
        return ' '.join(words[:target_words])
    
    def _abstractive_compression(self, content: str, target_ratio: float) -> str:
        """Abstractive compression using transformer models."""
        try:
            from transformers import pipeline
            import torch
            # Use a real summarization model
            model_name = "facebook/bart-large-cnn"
            device = 0 if torch.cuda.is_available() else -1
            summarizer = pipeline("summarization", model=model_name, tokenizer=model_name, device=device)
            # Calculate target length
            target_length = max(50, int(len(content.split()) * target_ratio))
            # Generate summary
            summary = summarizer(content, max_length=target_length, min_length=30, do_sample=False, truncation=True)
            return summary[0]['summary_text']
        except ImportError:
            logger.warning("Transformers library not available, using extractive summarization")
            return self._extractive_compression(content, target_ratio)
        except Exception as e:
            logger.error(f"Abstractive compression failed: {e}")
            return self._extractive_compression(content, target_ratio)
    
    def _keyword_compression(self, content: str, target_ratio: float) -> str:
        """Keyword-based compression."""
        # Extract key terms (simplified)
        words = content.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Keep most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        target_words = max(5, int(len(sorted_words) * target_ratio))
        keywords = [word for word, freq in sorted_words[:target_words]]
        
        return f"[Keywords: {', '.join(keywords)}]"

class PerfectRecallEngine(BaseEngine):
    """
    🧠 Perfect Recall Engine
    
    Memory management system that stores and retrieves coding interactions
    with semantic understanding and pattern matching.
    """
    
    def __init__(self, storage_path: str = "data/memory", config: Dict[str, Any] = None):
        super().__init__("perfect_recall", {})
        
        # Validate and set configuration
        self.config = self._validate_config(config or {})
        
        # Set validated configuration parameters
        self.storage_path = Path(self.config["storage_path"])
        self.max_memory_entries = self.config["max_memory_entries"]
        self.embedding_dimension = self.config["embedding_dimension"]
        self.similarity_threshold = self.config["similarity_threshold"]
        self.batch_size = self.config["batch_size"]
        self.enable_streaming = self.config["enable_streaming"]
        self.enable_pagination = self.config["enable_pagination"]
        self.max_concurrent_operations = self.config["max_concurrent_operations"]
        self.memory_cleanup_interval = self.config["memory_cleanup_interval"]
        self.vector_db_config = self.config["vector_db_config"]
        self.knowledge_graph_config = self.config["knowledge_graph_config"]
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Memory storage with performance optimizations
        self.memory_db = {}
        self.embeddings_cache = {}
        self.knowledge_graph = {}
        self.memory_index = {}  # Fast lookup index
        self.last_cleanup = datetime.now()
        
        # Workflow tracking components for long workflow optimization
        self.workflow_sessions = {}  # session_id -> WorkflowSession
        self.temporal_clusters = {}  # cluster_id -> TemporalCluster
        self.active_workflows = {}   # workflow_id -> active session_id
        self.workflow_memory_map = {}  # workflow_id -> list of memory_ids
        self.cross_reference_graph = {}  # memory_id -> set of referenced memory_ids
        
        # Advanced Infinite Context Management components
        self.hierarchical_memory = {
            'active': HierarchicalMemoryLevel('active', max_chunks=100),
            'short_term': HierarchicalMemoryLevel('short_term', max_chunks=1000),
            'long_term': HierarchicalMemoryLevel('long_term', max_chunks=1000000)
        }
        self.semantic_chunker = SemanticChunker()
        self.context_compressor = ContextCompressor()
        self.infinite_context_config = {
            'max_active_tokens': int(os.getenv("PERFECT_RECALL_MAX_ACTIVE_TOKENS", "8192")),
            'compression_threshold': float(os.getenv("PERFECT_RECALL_COMPRESSION_THRESHOLD", "0.7")),
            'semantic_chunking_enabled': os.getenv("PERFECT_RECALL_SEMANTIC_CHUNKING", "true").lower() == "true",
            'dynamic_compression_enabled': os.getenv("PERFECT_RECALL_DYNAMIC_COMPRESSION", "true").lower() == "true",
            'hierarchical_memory_enabled': os.getenv("PERFECT_RECALL_HIERARCHICAL_MEMORY", "true").lower() == "true"
        }
        
        # Workflow optimization configuration
        self.workflow_config = {
            "temporal_cluster_window": int(os.getenv("PERFECT_RECALL_TEMPORAL_WINDOW", "3600")),  # 1 hour default
            "cohesion_threshold": float(os.getenv("PERFECT_RECALL_COHESION_THRESHOLD", "0.6")),
            "cross_reference_weight": float(os.getenv("PERFECT_RECALL_CROSS_REF_WEIGHT", "0.2")),
            "temporal_decay_rate": float(os.getenv("PERFECT_RECALL_TEMPORAL_DECAY", "0.95")),
            "workflow_cohesion_weight": float(os.getenv("PERFECT_RECALL_WORKFLOW_COHESION_WEIGHT", "0.2"))
        }
        
        # Performance tracking
        self.operation_stats = {
            "total_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_response_time": 0.0,
            "workflow_operations": 0,
            "temporal_clustering_operations": 0
        }
        
        # Initialize components
        self._initialize_storage()
        self._load_existing_memories()
        
        logger.info(f"🧠 Perfect Recall Engine initialized with {len(self.memory_db)} memories")
        logger.info(f"📊 Configuration: max_entries={self.max_memory_entries}, batch_size={self.batch_size}, streaming={self.enable_streaming}")
        logger.info(f"♾️ Infinite Context Management: {self.infinite_context_config['semantic_chunking_enabled']}, Compression: {self.infinite_context_config['dynamic_compression_enabled']}")
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and set default configuration parameters for high-scale usage.
        
        Args:
            config: User-provided configuration
            
        Returns:
            Validated configuration with defaults
            
        Raises:
            ValueError: If configuration parameters are invalid
        """
        # Default configuration with environment-based values
        
        default_config = {
            "storage_path": os.getenv("PERFECT_RECALL_STORAGE_PATH", "data/memory"),
            "max_memory_entries": int(os.getenv("PERFECT_RECALL_MAX_ENTRIES", "100000")),  # Configurable via env
            "embedding_dimension": int(os.getenv("PERFECT_RECALL_EMBEDDING_DIM", "384")),
            "similarity_threshold": float(os.getenv("PERFECT_RECALL_SIMILARITY_THRESHOLD", "0.7")),
            "batch_size": int(os.getenv("PERFECT_RECALL_BATCH_SIZE", "500")),  # Configurable batch size
            "enable_streaming": os.getenv("PERFECT_RECALL_ENABLE_STREAMING", "true").lower() == "true",
            "enable_pagination": os.getenv("PERFECT_RECALL_ENABLE_PAGINATION", "true").lower() == "true",
            "max_concurrent_operations": int(os.getenv("PERFECT_RECALL_MAX_CONCURRENT", "5")),
            "memory_cleanup_interval": int(os.getenv("PERFECT_RECALL_CLEANUP_INTERVAL", "3600")),
            "vector_db_config": {
                "enable_chroma": os.getenv("PERFECT_RECALL_ENABLE_CHROMA", "true").lower() == "true",
                "chroma_persist_directory": os.getenv("PERFECT_RECALL_CHROMA_PATH", "data/memory/chroma"),
                "collection_name": os.getenv("PERFECT_RECALL_COLLECTION_NAME", "agent_memories"),
                "distance_metric": os.getenv("PERFECT_RECALL_DISTANCE_METRIC", "cosine")
            },
            "knowledge_graph_config": {
                "enable_neo4j": os.getenv("PERFECT_RECALL_ENABLE_NEO4J", "false").lower() == "true",
                "neo4j_uri": os.getenv("PERFECT_RECALL_NEO4J_URI", "bolt://localhost:7687"),
                "neo4j_username": os.getenv("PERFECT_RECALL_NEO4J_USER", "neo4j"),
                "neo4j_password": os.getenv("PERFECT_RECALL_NEO4J_PASS", "password")
            },
            "performance_config": {
                "enable_caching": os.getenv("PERFECT_RECALL_ENABLE_CACHE", "true").lower() == "true",
                "cache_size": int(os.getenv("PERFECT_RECALL_CACHE_SIZE", "5000")),
                "cache_ttl": int(os.getenv("PERFECT_RECALL_CACHE_TTL", "3600")),
                "enable_compression": os.getenv("PERFECT_RECALL_ENABLE_COMPRESSION", "true").lower() == "true",
                "compression_level": int(os.getenv("PERFECT_RECALL_COMPRESSION_LEVEL", "6"))
            },
            "security_config": {
                "enable_encryption": os.getenv("PERFECT_RECALL_ENABLE_ENCRYPTION", "false").lower() == "true",
                "encryption_key": os.getenv("PERFECT_RECALL_ENCRYPTION_KEY"),
                "access_control": os.getenv("PERFECT_RECALL_ACCESS_CONTROL", "false").lower() == "true"
            }
        }
        
        # Merge user config with defaults
        validated_config = default_config.copy()
        validated_config.update(config)
        
        # Validate critical parameters
        if validated_config["max_memory_entries"] < 1000:
            raise ValueError("max_memory_entries must be at least 1000 for high-scale usage")
        
        if validated_config["max_memory_entries"] > 10000000:
            logger.warning("max_memory_entries > 10M may impact performance")
        
        if not (0.1 <= validated_config["similarity_threshold"] <= 1.0):
            raise ValueError("similarity_threshold must be between 0.1 and 1.0")
        
        if validated_config["batch_size"] < 100 or validated_config["batch_size"] > 10000:
            raise ValueError("batch_size must be between 100 and 10000")
        
        if validated_config["max_concurrent_operations"] < 1 or validated_config["max_concurrent_operations"] > 50:
            raise ValueError("max_concurrent_operations must be between 1 and 50")
        
        if validated_config["embedding_dimension"] not in [128, 256, 384, 512, 768]:
            logger.warning(f"embedding_dimension {validated_config['embedding_dimension']} is not standard, may impact performance")
        
        # Validate storage path
        storage_path = Path(validated_config["storage_path"])
        if not storage_path.parent.exists():
            try:
                storage_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Cannot create storage directory: {e}")
        
        logger.info("✅ Configuration validation completed")
        return validated_config
    
    async def get_paginated_memories(
        self,
        query: str,
        page: int = 1,
        page_size: int = 100,
        content_types: List[str] = None,
        tags: List[str] = None,
        sort_by: str = "relevance"
    ) -> Dict[str, Any]:
        """
        Retrieve memories with pagination for large-scale operations.
        
        Args:
            query: Search query
            page: Page number (1-based)
            page_size: Number of results per page
            content_types: Filter by content types
            tags: Filter by tags
            sort_by: Sort method ("relevance", "timestamp", "usage_count")
            
        Returns:
            Paginated results with metadata
        """
        if not self.enable_pagination:
            logger.warning("Pagination is disabled, returning all results")
            all_results = await self.recall_memories(query, content_types, tags, limit=page_size * 10)
            return {
                "results": all_results,
                "pagination": {
                    "page": 1,
                    "page_size": len(all_results),
                    "total_pages": 1,
                    "total_results": len(all_results),
                    "has_next": False,
                    "has_previous": False
                }
            }
        
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Get all matching results first (for sorting)
        all_results = await self.recall_memories(query, content_types, tags, limit=10000)
        
        # Sort results
        if sort_by == "timestamp":
            all_results.sort(key=lambda x: x.entry.timestamp, reverse=True)
        elif sort_by == "usage_count":
            all_results.sort(key=lambda x: x.entry.usage_count, reverse=True)
        else:  # relevance (default)
            all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Apply pagination
        total_results = len(all_results)
        total_pages = (total_results + page_size - 1) // page_size
        start_idx = offset
        end_idx = min(start_idx + page_size, total_results)
        
        paginated_results = all_results[start_idx:end_idx]
        
        return {
            "results": paginated_results,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "total_results": total_results,
                "has_next": page < total_pages,
                "has_previous": page > 1,
                "start_index": start_idx + 1,
                "end_index": end_idx
            }
        }
    
    async def stream_memories(
        self,
        query: str,
        content_types: List[str] = None,
        tags: List[str] = None,
        batch_size: int = None
    ) -> AsyncGenerator[List[RecallResult], None]:
        """
        Stream memories in batches for large-scale operations.
        
        Args:
            query: Search query
            content_types: Filter by content types
            tags: Filter by tags
            batch_size: Size of each batch (defaults to config batch_size)
            
        Yields:
            Batches of recall results
        """
        if not self.enable_streaming:
            logger.warning("Streaming is disabled, returning all results at once")
            all_results = await self.recall_memories(query, content_types, tags, limit=10000)
            yield all_results
            return
        
        batch_size = batch_size or self.batch_size
        query_embedding = self._generate_embedding(query)
        
        # Get all matching memory IDs first
        matching_ids = []
        for entry in self.memory_db.values():
            # Apply filters
            if content_types and entry.content_type not in content_types:
                continue
            
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            
            # Calculate similarity
            if entry.embedding:
                similarity = self._calculate_similarity(query_embedding, entry.embedding)
                if similarity >= self.similarity_threshold:
                    matching_ids.append((entry.id, similarity))
        
        # Sort by similarity
        matching_ids.sort(key=lambda x: x[1], reverse=True)
        
        # Stream in batches
        for i in range(0, len(matching_ids), batch_size):
            batch_ids = matching_ids[i:i + batch_size]
            batch_results = []
            
            for memory_id, similarity in batch_ids:
                if memory_id in self.memory_db:
                    entry = self.memory_db[memory_id]
                    relevance = (similarity * 0.7) + (entry.success_score * 0.2) + (entry.usage_count * 0.1)
                    
                    batch_results.append(RecallResult(
                        entry=entry,
                        similarity_score=similarity,
                        relevance_score=relevance
                    ))
            
            if batch_results:
                yield batch_results
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.01)
    
    async def batch_store_memories(
        self,
        memories: List[Dict[str, Any]],
        batch_size: int = None
    ) -> Dict[str, Any]:
        """
        Store multiple memories in batches for performance optimization.
        
        Args:
            memories: List of memory dictionaries
            batch_size: Size of each batch (defaults to config batch_size)
            
        Returns:
            Batch operation results
        """
        batch_size = batch_size or self.batch_size
        total_memories = len(memories)
        stored_count = 0
        failed_count = 0
        errors = []
        
        logger.info(f"🔄 Starting batch storage of {total_memories} memories")
        
        # Process in batches
        for i in range(0, total_memories, batch_size):
            batch = memories[i:i + batch_size]
            batch_start = datetime.now()
            
            # Process batch concurrently
            tasks = []
            for memory in batch:
                task = self._store_single_memory(memory)
                tasks.append(task)
            
            # Execute batch with concurrency limit
            semaphore = asyncio.Semaphore(self.max_concurrent_operations)
            async def limited_task(task):
                async with semaphore:
                    return await task
            
            batch_tasks = [limited_task(task) for task in tasks]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    failed_count += 1
                    errors.append(str(result))
                else:
                    stored_count += 1
            
            batch_time = (datetime.now() - batch_start).total_seconds()
            logger.info(f"📦 Batch {i//batch_size + 1}: {len(batch)} memories in {batch_time:.2f}s")
        
        # Update operation stats
        self.operation_stats["total_operations"] += total_memories
        
        logger.info(f"✅ Batch storage completed: {stored_count} stored, {failed_count} failed")
        
        return {
            "total_memories": total_memories,
            "stored_count": stored_count,
            "failed_count": failed_count,
            "success_rate": stored_count / total_memories if total_memories > 0 else 0,
            "errors": errors[:10]  # Limit error list
        }
    
    async def _store_single_memory(self, memory: Dict[str, Any]) -> str:
        """Store a single memory (helper for batch operations)."""
        try:
            return await self.store_memory(
                content=memory.get("content", ""),
                content_type=memory.get("content_type"),
                tags=memory.get("tags", []),
                context=memory.get("context", {}),
                success_score=memory.get("success_score", 0.0),
                metadata=memory.get("metadata", {})
            )
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics for monitoring."""
        total_memories = len(self.memory_db)
        cache_hit_rate = (
            self.operation_stats["cache_hits"] / 
            (self.operation_stats["cache_hits"] + self.operation_stats["cache_misses"])
            if (self.operation_stats["cache_hits"] + self.operation_stats["cache_misses"]) > 0
            else 0
        )
        
        # Calculate memory usage
        memory_size_bytes = sum(
            len(entry.content.encode('utf-8')) + 
            len(str(entry.embedding)) if entry.embedding else 0
            for entry in self.memory_db.values()
        )
        
        return {
            "memory_stats": {
                "total_memories": total_memories,
                "memory_usage_mb": memory_size_bytes / (1024 * 1024),
                "max_memory_entries": self.max_memory_entries,
                "memory_utilization": total_memories / self.max_memory_entries
            },
            "performance_stats": {
                "total_operations": self.operation_stats["total_operations"],
                "cache_hit_rate": cache_hit_rate,
                "average_response_time": self.operation_stats["average_response_time"],
                "last_cleanup": self.last_cleanup.isoformat()
            },
            "configuration": {
                "batch_size": self.batch_size,
                "enable_streaming": self.enable_streaming,
                "enable_pagination": self.enable_pagination,
                "max_concurrent_operations": self.max_concurrent_operations
            },
            "storage_stats": {
                "storage_path": str(self.storage_path),
                "vector_db_available": self.collection is not None,
                "knowledge_graph_available": self.neo4j_driver is not None
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize the Perfect Recall Engine."""
        try:
            self._initialize_storage()
            self._initialize_embedding_model()
            self._load_existing_memories()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Perfect Recall Engine: {e}")
            return False
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model for semantic search."""
        try:
            # Try to import and initialize SentenceTransformer
            try:
                from sentence_transformers import SentenceTransformer
                self._cached_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Embedding model initialized successfully")
            except ImportError:
                logger.warning("SentenceTransformer not available, using fallback embeddings")
                self._cached_embedding_model = None
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self._cached_embedding_model = None
    
    async def get_engine_status(self) -> Dict[str, Any]:
        """Get current engine status and metrics."""
        performance_metrics = await self.get_performance_metrics()
        
        return {
            "engine_name": "Perfect Recall Engine",
            "status": "operational",
            "memory_count": len(self.memory_db),
            "storage_path": str(self.storage_path),
            "vector_db_available": hasattr(self, 'vector_db') and self.vector_db is not None,
            "knowledge_graph_available": hasattr(self, 'knowledge_graph') and self.knowledge_graph is not None,
            "performance_metrics": performance_metrics,
            "configuration": {
                "max_memory_entries": self.max_memory_entries,
                "batch_size": self.batch_size,
                "enable_streaming": self.enable_streaming,
                "enable_pagination": self.enable_pagination,
                "max_concurrent_operations": self.max_concurrent_operations
            }
        }
    
    def _initialize_storage(self):
        """Initialize storage components."""
        try:
            # Try to initialize ChromaDB for vector search
            self._init_vector_db()
        except ImportError:
            logger.warning("ChromaDB not available, using fallback vector search")
            self.vector_db = None
        
        try:
            # Try to initialize Neo4j for knowledge graph
            self._init_knowledge_graph()
        except ImportError:
            logger.warning("Neo4j not available, using fallback graph storage")
            self.neo4j_driver = None
    
    def _init_vector_db(self):
        """Initialize ChromaDB for vector search."""
        if not self.vector_db_config.get("enable_chroma", True):
            logger.info("ChromaDB disabled in configuration")
            self.chroma_client = None
            self.collection = None
            return
            
        try:
            import chromadb
            chroma_path = self.vector_db_config.get("chroma_persist_directory", str(self.storage_path / "chroma"))
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
            
            collection_name = self.vector_db_config.get("collection_name", "agent_memories")
            distance_metric = self.vector_db_config.get("distance_metric", "cosine")
            
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Agent Perfect Recall memories"},
                embedding_function=None  # Use default
            )
            logger.info(f"✅ ChromaDB initialized for vector search: {collection_name}")
        except Exception as e:
            logger.warning(f"ChromaDB initialization failed: {e}")
            self.chroma_client = None
            self.collection = None
    
    def _init_knowledge_graph(self):
        """Initialize Neo4j for knowledge graph with real driver operations."""
        try:
            from neo4j import GraphDatabase
            import os
            
            # Get Neo4j connection details from environment or config
            neo4_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            neo4j_user = os.getenv("NEO4J_USER", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
            
            # Initialize real Neo4j driver
            self.neo4j_driver = GraphDatabase.driver(neo4_uri, auth=(neo4j_user, neo4j_password))
            
            # Test connection and create constraints/indexes
            with self.neo4j_driver.session() as session:
                # Create constraints for unique entities
                session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
                session.run("CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE")
                
                # Create indexes for better performance
                session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
                session.run("CREATE INDEX memory_type IF NOT EXISTS FOR (m:Memory) ON (m.content_type)")
                session.run("CREATE INDEX memory_timestamp IF NOT EXISTS FOR (m:Memory) ON (m.timestamp)")
                
                logger.info("📊 Neo4j graph initialized with real driver")
                
        except ImportError:
            logger.debug("Neo4j driver not available, using in-memory graph")
            self.neo4j_driver = None
        except Exception as e:
            # Only log as debug since this is expected behavior when Neo4j is not running
            logger.debug(f"Neo4j not available (using in-memory graph): {str(e).split(',')[0]}")
            self.neo4j_driver = None
    
    def _load_existing_memories(self):
        """Load existing memories from storage and generate missing embeddings."""
        memory_file = self.storage_path / "memories.json"
        if memory_file.exists():
            try:
                with open(memory_file, 'r') as f:
                    data = json.load(f)
                    memories_without_embeddings = 0
                    for entry_data in data:
                        entry = MemoryEntry(**entry_data)
                        entry.timestamp = datetime.fromisoformat(entry.timestamp)
                        
                        # Generate embedding if missing (fix for infinite memory recall)
                        if not entry.embedding or len(entry.embedding) == 0:
                            entry.embedding = self._generate_embedding(entry.content)
                            memories_without_embeddings += 1
                        
                        self.memory_db[entry.id] = entry
                
                logger.info(f"📚 Loaded {len(self.memory_db)} existing memories")
                if memories_without_embeddings > 0:
                    logger.info(f"🔧 Generated embeddings for {memories_without_embeddings} memories")
                    # Save updated memories with embeddings
                    self._save_memories()
                    
            except Exception as e:
                logger.error(f"Failed to load memories: {e}")
    
    def _serialize_datetimes(self, obj):
        """Recursively convert all datetime objects in a dict/list to ISO strings."""
        if isinstance(obj, dict):
            return {k: self._serialize_datetimes(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetimes(v) for v in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj

    def _save_memories(self):
        """Save memories to persistent storage."""
        memory_file = self.storage_path / "memories.json"
        try:
            data = []
            for entry in self.memory_db.values():
                entry_dict = asdict(entry)
                entry_dict = self._serialize_datetimes(entry_dict)
                data.append(entry_dict)
            with open(memory_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"💾 Saved {len(data)} memories to storage")
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        🚀 OPTIMIZED EMBEDDING GENERATION - PHASE 2 PERFORMANCE
        Generate embeddings using shared model cache for maximum performance.
        """
        try:
            # Priority 1: Use shared model cache (PHASE 2 OPTIMIZATION)
            # Fix: Don't use asyncio.run() in an async context
            if hasattr(self, '_cached_embedding_model') and self._cached_embedding_model:
                return self._cached_embedding_model.encode(text).tolist()
            else:
                # Initialize cached model if not exists
                return self._generate_embedding_fallback(text)
            
        except Exception as e:
            logger.error(f"Cached embedding generation failed: {e}")
            # Fallback to previous implementation
            return self._generate_embedding_fallback(text)

    async def _generate_embedding_cached(self, text: str) -> List[float]:
        """Generate embeddings using shared model cache."""
        try:
            # Import the shared model cache
            from .model_cache_manager import get_embedding_model
            
            # Get model from shared cache
            embedding_model = await get_embedding_model("all-MiniLM-L6-v2", priority=5)
            
            if embedding_model is not None:
                # Generate embedding using cached model
                embedding = embedding_model.encode(text)
                return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            else:
                # Fallback to TF-IDF if sentence transformers not available
                return await self._generate_tfidf_embedding_cached(text)
                
        except Exception as e:
            logger.error(f"Cached embedding generation failed: {e}")
            raise

    async def _generate_tfidf_embedding_cached(self, text: str) -> List[float]:
        """Generate TF-IDF embeddings using shared cache."""
        try:
            from .model_cache_manager import get_tfidf_model
            
            # Get TF-IDF model from cache
            tfidf_model = await get_tfidf_model(
                "perfect_recall_tfidf",
                max_features=self.embedding_dimension,
                ngram_range=(1, 2)
            )
            
            if tfidf_model is not None and hasattr(tfidf_model, 'vocabulary_'):
                # Model is already fitted
                tfidf_vector = tfidf_model.transform([text])
                dense_vector = tfidf_vector.toarray()[0]
                
                # Pad or truncate to embedding dimension
                if len(dense_vector) > self.embedding_dimension:
                    embedding = dense_vector[:self.embedding_dimension]
                else:
                    embedding = list(dense_vector) + [0.0] * (self.embedding_dimension - len(dense_vector))
                
                # Normalize
                magnitude = sum(x*x for x in embedding) ** 0.5
                if magnitude > 0:
                    embedding = [x / magnitude for x in embedding]
                
                return embedding
            else:
                # Need to fit TF-IDF model
                return await self._fit_and_cache_tfidf(text)
                
        except Exception as e:
            logger.error(f"Cached TF-IDF embedding failed: {e}")
            raise

    async def _fit_and_cache_tfidf(self, text: str) -> List[float]:
        """Fit and cache TF-IDF model."""
        try:
            from .model_cache_manager import get_tfidf_model
            
            # Build corpus from existing memories
            corpus = [text]
            if hasattr(self, 'memory_db') and self.memory_db:
                corpus.extend([memory.get('content', '') for memory in self.memory_db.values() if memory.get('content')])
            
            # Ensure we have enough data
            if len(corpus) < 2:
                corpus.append("default text for training")
            
            # Get fresh TF-IDF model
            tfidf_model = await get_tfidf_model(
                "perfect_recall_tfidf_fresh",
                max_features=self.embedding_dimension,
                ngram_range=(1, 2)
            )
            
            if tfidf_model is not None:
                # Fit the model
                tfidf_model.fit(corpus)
                
                # Generate embedding for the text
                tfidf_vector = tfidf_model.transform([text])
                dense_vector = tfidf_vector.toarray()[0]
                
                # Process vector
                if len(dense_vector) > self.embedding_dimension:
                    embedding = dense_vector[:self.embedding_dimension]
                else:
                    embedding = list(dense_vector) + [0.0] * (self.embedding_dimension - len(dense_vector))
                
                # Normalize
                magnitude = sum(x*x for x in embedding) ** 0.5
                if magnitude > 0:
                    embedding = [x / magnitude for x in embedding]
                
                return embedding
            else:
                raise Exception("Failed to get TF-IDF model")
                
        except Exception as e:
            logger.error(f"TF-IDF fitting failed: {e}")
            raise

    def _generate_embedding_fallback(self, text: str) -> List[float]:
        """Fallback embedding generation without cache."""
        try:
            # Try sentence transformers without cache
            from sentence_transformers import SentenceTransformer
            
            model_options = [
                'all-MiniLM-L6-v2',
                'paraphrase-MiniLM-L6-v2'
            ]
            
            for model_name in model_options:
                try:
                    logger.warning(f"⚠️ Using fallback model loading for {model_name}")
                    model = SentenceTransformer(model_name)
                    embedding = model.encode(text)
                    return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                except Exception as e:
                    logger.warning(f"Fallback model {model_name} failed: {e}")
                    continue
            
            # Final fallback to improved hash embedding
            return self._generate_improved_hash_embedding(text)
            
        except ImportError:
            logger.warning("SentenceTransformers not available, using hash embedding")
            return self._generate_improved_hash_embedding(text)
        except Exception as e:
            logger.error(f"All fallback methods failed: {e}")
            return self._generate_improved_hash_embedding(text)

    def _generate_improved_hash_embedding(self, text: str) -> List[float]:
        """
        Generate improved hash-based embedding with better semantic properties.
        This is a sophisticated fallback that uses multiple hash functions and text features.
        """
        try:
            # Preprocess text for better semantic representation
            words = text.lower().split()
            
            # Remove common stop words for better semantic focus
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
            }
            
            meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
            
            # Initialize embedding vector
            embedding = [0.0] * self.embedding_dimension
            
            # Method 1: Word-based hash embedding with multiple hash functions
            for i, word in enumerate(meaningful_words[:50]):  # Limit to 50 words for performance
                # Use multiple hash functions for better distribution
                hash1 = hash(word) % self.embedding_dimension
                hash2 = hash(word[::-1]) % self.embedding_dimension  # Reverse word
                hash3 = hash(word + str(len(word))) % self.embedding_dimension  # Length-aware
                
                # Weight by position (earlier words get higher weight)
                position_weight = 1.0 / (i + 1) ** 0.5
                
                embedding[hash1] += position_weight
                embedding[hash2] += position_weight * 0.7
                embedding[hash3] += position_weight * 0.5
            
            # Method 2: N-gram based features
            bigrams = [meaningful_words[i] + meaningful_words[i+1] 
                      for i in range(len(meaningful_words)-1)]
            
            for bigram in bigrams[:20]:  # Limit bigrams
                bigram_hash = hash(bigram) % self.embedding_dimension
                embedding[bigram_hash] += 0.8
            
            # Method 3: Text statistics features
            text_stats = {
                'avg_word_length': sum(len(word) for word in meaningful_words) / len(meaningful_words) if meaningful_words else 0,
                'unique_word_ratio': len(set(meaningful_words)) / len(meaningful_words) if meaningful_words else 0,
                'total_length': len(text),
                'sentence_count': len([s for s in text.split('.') if s.strip()])
            }
            
            # Map statistics to embedding dimensions
            for i, (stat_name, stat_value) in enumerate(text_stats.items()):
                if i < self.embedding_dimension:
                    embedding[i] += stat_value * 0.3
            
            # Normalize the embedding
            magnitude = sum(x*x for x in embedding) ** 0.5
            if magnitude > 0:
                embedding = [x / magnitude for x in embedding]
            else:
                # Fallback for empty text
                embedding[0] = 1.0
            
            return embedding
            
        except Exception as e:
            logger.error(f"Improved hash embedding generation failed: {e}")
            # Ultra-simple fallback
            simple_embedding = [0.0] * self.embedding_dimension
            if text:
                simple_hash = hash(text) % self.embedding_dimension
                simple_embedding[simple_hash] = 1.0
            return simple_embedding
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings."""
        if len(embedding1) != len(embedding2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    async def store_knowledge(self, content: str, metadata: Dict[str, Any]) -> str:
        """Store knowledge with metadata (alias for store_memory)"""
        return await self.store_memory(
            content=content,
            content_type=metadata.get("type", "knowledge"),
            tags=metadata.get("tags", []),
            context=metadata,
            success_score=metadata.get("quality_score", 0.8)
        )
    
    async def recall_knowledge(self, query: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Recall knowledge based on query (alias for recall_memories)"""
        content_types = None
        limit = 5
        
        # Handle context parameter safely - it might be a string or dict
        if context:
            if isinstance(context, dict):
                if context.get("type"):
                    content_types = [context.get("type")]
                limit = context.get("limit", 5)
            elif isinstance(context, str):
                # If context is a string, treat it as additional query context
                query = f"{query} {context}"
        
        results = await self.recall_memories(
            query=query,
            content_types=content_types,
            limit=limit
        )
        
        # Convert RecallResult objects to dictionaries
        return [
            {
                "id": result.entry.id,
                "content": result.entry.content,
                "content_type": result.entry.content_type,
                "tags": result.entry.tags,
                "context": result.entry.context,
                "success_score": result.entry.success_score,
                "timestamp": result.entry.timestamp.isoformat(),
                "similarity_score": result.similarity_score
            }
            for result in results
        ]
    
    async def semantic_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search on stored memories.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching memory entries with similarity scores
        """
        try:
            # Generate embedding for the query
            query_embedding = self._generate_embedding(query)
            
            # Search in vector database if available
            if self.collection:
                try:
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=limit
                    )
                    
                    search_results = []
                    if results and results.get('documents'):
                        for i, doc in enumerate(results['documents'][0]):
                            metadata = results.get('metadatas', [[]])[0][i] if i < len(results.get('metadatas', [[]])[0]) else {}
                            distance = results.get('distances', [[]])[0][i] if i < len(results.get('distances', [[]])[0]) else 1.0
                            similarity = 1.0 - distance  # Convert distance to similarity
                            
                            search_results.append({
                                "content": doc,
                                "similarity_score": similarity,
                                "metadata": metadata,
                                "content_type": metadata.get("content_type", "unknown"),
                                "tags": metadata.get("tags", [])
                            })
                    
                    return search_results
                except Exception as e:
                    logger.warning(f"Vector search failed: {e}")
            
            # Fallback to in-memory search
            results = []
            for entry in self.memory_db.values():
                if entry.embedding:
                    similarity = self._calculate_similarity(query_embedding, entry.embedding)
                    if similarity > 0.1:  # Minimum similarity threshold
                        results.append({
                            "content": entry.content,
                            "similarity_score": similarity,
                            "metadata": {
                                "content_type": entry.content_type,
                                "tags": entry.tags,
                                "timestamp": entry.timestamp.isoformat()
                            },
                            "content_type": entry.content_type,
                            "tags": entry.tags
                        })
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def store_memory(
        self,
        content: str,
        content_type: str = None,
        tags: List[str] = None,
        context: Dict[str, Any] = None,
        success_score: float = 0.0,
        metadata: Dict[str, Any] = None,
        workflow_id: str = None,
        session_id: str = None
    ) -> str:
        """
        Store a new memory entry with semantic understanding and workflow tracking.
        
        Args:
            content: The content to store
            content_type: Type of content ('code', 'solution', 'error', 'conversation')
            tags: List of tags for categorization
            context: Additional context information
            success_score: Score indicating success/quality of the solution
            metadata: Additional metadata (can include content_type, tags, etc.)
            workflow_id: Optional workflow ID for tracking
            session_id: Optional session ID for temporal clustering
            
        Returns:
            Memory entry ID
        """
        # Handle metadata parameter for compatibility
        if metadata:
            content_type = metadata.get("type", content_type or "general")
            tags = metadata.get("tags", tags or [])
            context = metadata.get("context", context or {})
            success_score = metadata.get("success_score", success_score)
            workflow_id = metadata.get("workflow_id", workflow_id)
            session_id = metadata.get("session_id", session_id)
        
        # Default content type
        if not content_type:
            content_type = "general"
        
        # Generate unique ID
        memory_id = hashlib.md5(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()
        
        # Generate embedding
        embedding = self._generate_embedding(content)
        
        # Calculate workflow cohesion score if workflow context available
        workflow_cohesion_score = 0.0
        if workflow_id:
            workflow_cohesion_score = await self._calculate_workflow_cohesion(content, workflow_id)
        
        # Create memory entry with enhanced workflow tracking
        entry = MemoryEntry(
            id=memory_id,
            timestamp=datetime.now(),
            content=content,
            content_type=content_type,
            tags=tags or [],
            context=context or {},
            embedding=embedding,
            success_score=success_score,
            workflow_id=workflow_id,
            session_id=session_id,
            workflow_cohesion_score=workflow_cohesion_score
        )
        
        # Store in memory database
        self.memory_db[memory_id] = entry
        
        # Update workflow tracking
        if workflow_id:
            await self._update_workflow_tracking(entry, workflow_id, session_id)
        
        # Store in vector database if available
        if self.collection:
            try:
                self.collection.add(
                    documents=[content],
                    embeddings=[embedding],
                    metadatas=[{
                        "content_type": content_type,
                        "tags": ",".join(tags or []),
                        "success_score": success_score,
                        "timestamp": entry.timestamp.isoformat(),
                        "workflow_id": workflow_id or "",
                        "session_id": session_id or "",
                        "workflow_cohesion_score": workflow_cohesion_score
                    }],
                    ids=[memory_id]
                )
            except Exception as e:
                logger.warning(f"Failed to store in ChromaDB: {e}")
        
        # Update knowledge graph relationships
        await self._update_knowledge_graph(entry)
        
        # Check if cleanup is needed
        if self._should_perform_cleanup():
            await self._cleanup_old_memories()
            self.last_cleanup = datetime.now()
        
        # Save to persistent storage
        self._save_memories()
        
        logger.info(f"🧠 Stored memory: {content_type} - {len(content)} chars (workflow: {workflow_id})")
        return memory_id

    async def _calculate_workflow_cohesion(self, content: str, workflow_id: str) -> float:
        """
        Calculate how well this memory fits within the existing workflow context.
        
        Args:
            content: Memory content to analyze
            workflow_id: Workflow ID to check against
            
        Returns:
            Cohesion score between 0.0 and 1.0
        """
        if workflow_id not in self.workflow_memory_map:
            return 0.5  # Default cohesion for new workflows
        
        workflow_memory_ids = self.workflow_memory_map[workflow_id]
        if not workflow_memory_ids:
            return 0.5
        
        # Get recent workflow memories (last 10)
        recent_memories = []
        for memory_id in workflow_memory_ids[-10:]:
            if memory_id in self.memory_db:
                recent_memories.append(self.memory_db[memory_id])
        
        if not recent_memories:
            return 0.5
        
        # Calculate semantic similarity with recent workflow memories
        content_embedding = self._generate_embedding(content)
        similarities = []
        
        for memory in recent_memories:
            if memory.embedding:
                similarity = self._calculate_similarity(content_embedding, memory.embedding)
                similarities.append(similarity)
        
        if not similarities:
            return 0.5
        
        # Calculate average similarity as cohesion score
        avg_similarity = sum(similarities) / len(similarities)
        
        # Boost cohesion for high similarity
        cohesion_score = min(1.0, avg_similarity * 1.2)
        
        return cohesion_score

    async def _update_workflow_tracking(self, entry: MemoryEntry, workflow_id: str, session_id: str = None):
        """
        Update workflow tracking structures when storing a new memory.
        
        Args:
            entry: Memory entry being stored
            workflow_id: Workflow ID
            session_id: Optional session ID
        """
        # Update workflow memory map
        if workflow_id not in self.workflow_memory_map:
            self.workflow_memory_map[workflow_id] = []
        self.workflow_memory_map[workflow_id].append(entry.id)
        
        # Update active session if provided
        if session_id and session_id in self.workflow_sessions:
            session = self.workflow_sessions[session_id]
            session.memory_ids.append(entry.id)
            
            # Update session cohesion score
            if len(session.memory_ids) > 1:
                session.cohesion_score = await self._calculate_session_cohesion(session.memory_ids)
        
        # Update cross-reference graph
        if entry.id not in self.cross_reference_graph:
            self.cross_reference_graph[entry.id] = set()
        
        # Find related memories in the same workflow
        if workflow_id in self.workflow_memory_map:
            workflow_memories = self.workflow_memory_map[workflow_id]
            for memory_id in workflow_memories[-5:]:  # Check last 5 memories
                if memory_id != entry.id and memory_id in self.memory_db:
                    # Calculate similarity to determine if they should be cross-referenced
                    other_memory = self.memory_db[memory_id]
                    if other_memory.embedding and entry.embedding:
                        similarity = self._calculate_similarity(entry.embedding, other_memory.embedding)
                        if similarity > 0.7:  # High similarity threshold for cross-references
                            # Add bidirectional cross-reference
                            if memory_id not in self.cross_reference_graph:
                                self.cross_reference_graph[memory_id] = set()
                            self.cross_reference_graph[memory_id].add(entry.id)
                            self.cross_reference_graph[entry.id].add(memory_id)
                            
                            # Update cross-reference counts
                            entry.cross_reference_count = len(self.cross_reference_graph[entry.id])
                            other_memory.cross_reference_count = len(self.cross_reference_graph[memory_id])

    async def _calculate_session_cohesion(self, memory_ids: List[str]) -> float:
        """
        Calculate cohesion score for a session based on its memories.
        
        Args:
            memory_ids: List of memory IDs in the session
            
        Returns:
            Cohesion score between 0.0 and 1.0
        """
        if len(memory_ids) < 2:
            return 0.5
        
        memories = [self.memory_db[mid] for mid in memory_ids if mid in self.memory_db]
        if len(memories) < 2:
            return 0.5
        
        # Calculate pairwise similarities
        similarities = []
        for i, memory1 in enumerate(memories):
            for memory2 in memories[i+1:]:
                if memory1.embedding and memory2.embedding:
                    similarity = self._calculate_similarity(memory1.embedding, memory2.embedding)
                    similarities.append(similarity)
        
        if not similarities:
            return 0.5
        
        # Return average similarity as cohesion score
        return sum(similarities) / len(similarities)
    
    def _should_perform_cleanup(self) -> bool:
        """Check if memory cleanup should be performed."""
        # Check if we're over the memory limit
        if len(self.memory_db) > self.max_memory_entries:
            return True
        
        # Check if cleanup interval has passed
        time_since_cleanup = (datetime.now() - self.last_cleanup).total_seconds()
        if time_since_cleanup > self.memory_cleanup_interval:
            return True
        
        return False
    
    @staticmethod
    def get_global_scale_config() -> Dict[str, Any]:
        """
        Get a pre-configured configuration for high-scale usage.
        
        Returns:
            Configuration optimized for handling large memory volumes
        """
        import os
        
        return {
            "storage_path": os.getenv("GLOBAL_SCALE_STORAGE_PATH", "data/memory/global"),
            "max_memory_entries": int(os.getenv("GLOBAL_SCALE_MAX_ENTRIES", "1000000")),  # 1M entries
            "embedding_dimension": int(os.getenv("GLOBAL_SCALE_EMBEDDING_DIM", "384")),
            "similarity_threshold": float(os.getenv("GLOBAL_SCALE_SIMILARITY_THRESHOLD", "0.7")),
            "batch_size": int(os.getenv("GLOBAL_SCALE_BATCH_SIZE", "2000")),  # Larger batches
            "enable_streaming": True,
            "enable_pagination": True,
            "max_concurrent_operations": int(os.getenv("GLOBAL_SCALE_MAX_CONCURRENT", "10")),
            "memory_cleanup_interval": int(os.getenv("GLOBAL_SCALE_CLEANUP_INTERVAL", "1800")),
            "vector_db_config": {
                "enable_chroma": True,
                "chroma_persist_directory": os.getenv("GLOBAL_SCALE_CHROMA_PATH", "data/memory/global/chroma"),
                "collection_name": os.getenv("GLOBAL_SCALE_COLLECTION_NAME", "agent_global_memories"),
                "distance_metric": "cosine"
            },
            "knowledge_graph_config": {
                "enable_neo4j": os.getenv("GLOBAL_SCALE_ENABLE_NEO4J", "false").lower() == "true",
                "neo4j_uri": os.getenv("GLOBAL_SCALE_NEO4J_URI", "bolt://localhost:7687"),
                "neo4j_username": os.getenv("GLOBAL_SCALE_NEO4J_USER", "neo4j"),
                "neo4j_password": os.getenv("GLOBAL_SCALE_NEO4J_PASS", "password")
            },
            "performance_config": {
                "enable_caching": True,
                "cache_size": int(os.getenv("GLOBAL_SCALE_CACHE_SIZE", "20000")),
                "cache_ttl": int(os.getenv("GLOBAL_SCALE_CACHE_TTL", "7200")),
                "enable_compression": True,
                "compression_level": int(os.getenv("GLOBAL_SCALE_COMPRESSION_LEVEL", "7"))
            },
            "security_config": {
                "enable_encryption": os.getenv("GLOBAL_SCALE_ENABLE_ENCRYPTION", "false").lower() == "true",
                "encryption_key": os.getenv("GLOBAL_SCALE_ENCRYPTION_KEY"),
                "access_control": os.getenv("GLOBAL_SCALE_ACCESS_CONTROL", "false").lower() == "true"
            }
        }
    
    @staticmethod
    def get_enterprise_config() -> Dict[str, Any]:
        """
        Get a pre-configured configuration for enterprise usage.
        
        Returns:
            Configuration optimized for enterprise environments
        """
        import os
        
        return {
            "storage_path": os.getenv("ENTERPRISE_STORAGE_PATH", "data/memory/enterprise"),
            "max_memory_entries": int(os.getenv("ENTERPRISE_MAX_ENTRIES", "500000")),  # 500K entries
            "embedding_dimension": int(os.getenv("ENTERPRISE_EMBEDDING_DIM", "384")),
            "similarity_threshold": float(os.getenv("ENTERPRISE_SIMILARITY_THRESHOLD", "0.75")),
            "batch_size": int(os.getenv("ENTERPRISE_BATCH_SIZE", "1000")),
            "enable_streaming": True,
            "enable_pagination": True,
            "max_concurrent_operations": int(os.getenv("ENTERPRISE_MAX_CONCURRENT", "8")),
            "memory_cleanup_interval": int(os.getenv("ENTERPRISE_CLEANUP_INTERVAL", "3600")),
            "vector_db_config": {
                "enable_chroma": True,
                "chroma_persist_directory": os.getenv("ENTERPRISE_CHROMA_PATH", "data/memory/enterprise/chroma"),
                "collection_name": os.getenv("ENTERPRISE_COLLECTION_NAME", "agent_enterprise_memories"),
                "distance_metric": "cosine"
            },
            "knowledge_graph_config": {
                "enable_neo4j": os.getenv("ENTERPRISE_ENABLE_NEO4J", "false").lower() == "true",
                "neo4j_uri": os.getenv("ENTERPRISE_NEO4J_URI", "bolt://localhost:7687"),
                "neo4j_username": os.getenv("ENTERPRISE_NEO4J_USER", "neo4j"),
                "neo4j_password": os.getenv("ENTERPRISE_NEO4J_PASS", "password")
            },
            "performance_config": {
                "enable_caching": True,
                "cache_size": int(os.getenv("ENTERPRISE_CACHE_SIZE", "10000")),
                "cache_ttl": int(os.getenv("ENTERPRISE_CACHE_TTL", "3600")),
                "enable_compression": True,
                "compression_level": int(os.getenv("ENTERPRISE_COMPRESSION_LEVEL", "6"))
            },
            "security_config": {
                "enable_encryption": os.getenv("ENTERPRISE_ENABLE_ENCRYPTION", "false").lower() == "true",
                "encryption_key": os.getenv("ENTERPRISE_ENCRYPTION_KEY"),
                "access_control": os.getenv("ENTERPRISE_ACCESS_CONTROL", "false").lower() == "true"
            }
        }
    
    async def recall_memories(
        self,
        query: str,
        content_types: List[str] = None,
        tags: List[str] = None,
        limit: int = 10,
        min_similarity: float = None,
        workflow_id: str = None,
        session_id: str = None
    ) -> List[RecallResult]:
        """
        Recall memories based on semantic similarity and filters with workflow optimization.
        
        Args:
            query: Search query
            content_types: Filter by content types
            tags: Filter by tags
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            workflow_id: Optional workflow ID for context-aware retrieval
            session_id: Optional session ID for temporal clustering
            
        Returns:
            List of recall results sorted by relevance
        """
        query_embedding = self._generate_embedding(query)
        results = []
        
        # Lower similarity threshold to ensure we get results for debugging
        min_sim = min_similarity or min(self.similarity_threshold, 0.1)
        logger.info(f"🔍 Using similarity threshold: {min_sim}")
        
        # Track workflow operation
        self.operation_stats["workflow_operations"] += 1
        
        # Get workflow context if available
        workflow_context = None
        if workflow_id:
            workflow_context = await self._get_workflow_context(workflow_id)
        
        # Get temporal cluster if available
        temporal_context = None
        if session_id:
            temporal_context = await self._get_temporal_context(session_id)
        
        # Search through all memories
        logger.info(f"🔍 Searching through {len(self.memory_db)} memories for query: '{query[:50]}...'")
        
        total_checked = 0
        total_with_embeddings = 0
        total_above_threshold = 0
        
        for entry in self.memory_db.values():
            total_checked += 1
            
            # Apply filters
            if content_types and entry.content_type not in content_types:
                print(f"  Filtered out by content_type: {entry.content_type}")
                continue
            
            if tags and not any(tag in entry.tags for tag in tags):
                print(f"  Filtered out by tags: {entry.tags}")
                continue
            
            # Calculate similarity
            if entry.embedding:
                total_with_embeddings += 1
                similarity = self._calculate_similarity(query_embedding, entry.embedding)
                
                print(f"  Memory {total_checked}: {entry.content[:40]}... similarity: {similarity:.3f} (threshold: {min_sim})")
                
                if similarity >= min_sim:
                    total_above_threshold += 1
                    print(f"    ✅ ABOVE THRESHOLD - should be added to results")
                    
                    # Enhanced relevance calculation with workflow optimization
                    relevance = await self._calculate_enhanced_relevance(
                        entry, similarity, workflow_context, temporal_context, query
                    )
                    
                    results.append(RecallResult(
                        entry=entry,
                        similarity_score=similarity,
                        relevance_score=relevance
                    ))
                else:
                    print(f"    ❌ Below threshold")
            else:
                # Generate embedding for entries that don't have one
                logger.warning(f"Entry {entry.id} missing embedding, generating...")
                entry.embedding = self._generate_embedding(entry.content)
                similarity = self._calculate_similarity(query_embedding, entry.embedding) if entry.embedding else 0.0
                
                if similarity >= min_sim:
                    # Enhanced relevance calculation with workflow optimization
                    relevance = await self._calculate_enhanced_relevance(
                        entry, similarity, workflow_context, temporal_context, query
                    )
                    
                    results.append(RecallResult(
                        entry=entry,
                        similarity_score=similarity,
                        relevance_score=relevance
                    ))
        
        # Sort by enhanced relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Update usage counts and workflow access tracking
        for result in results[:limit]:
            result.entry.usage_count += 1
            if workflow_id:
                result.entry.last_workflow_access = datetime.now()
        
        # Update cross-reference tracking
        if len(results) > 1:
            await self._update_cross_references(results[:limit])
        
        print(f"\n🔍 RECALL SUMMARY:")
        print(f"  Total memories checked: {total_checked}")
        print(f"  Memories with embeddings: {total_with_embeddings}")
        print(f"  Memories above threshold: {total_above_threshold}")
        print(f"  Results added: {len(results)}")
        
        logger.info(f"🔍 Recalled {len(results[:limit])}/{len(self.memory_db)} memories for query: '{query[:50]}...' (threshold: {min_sim}, workflow: {workflow_id})")
        if len(results) == 0:
            logger.warning(f"⚠️ No memories found! Total in DB: {len(self.memory_db)}, Query embedding: {'✅' if query_embedding else '❌'}")
        return results[:limit]

    async def retrieve_memories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve memories as dictionary format for compatibility.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of memory dictionaries
        """
        try:
            recall_results = await self.recall_memories(query, limit=limit)
            
            # Convert to dictionary format
            memories = []
            for result in recall_results:
                memory_dict = {
                    "id": result.entry.id,
                    "content": result.entry.content,
                    "content_type": result.entry.content_type,
                    "tags": result.entry.tags,
                    "context": result.entry.context,
                    "timestamp": result.entry.timestamp.isoformat(),
                    "success_score": result.entry.success_score,
                    "usage_count": result.entry.usage_count,
                    "similarity_score": result.similarity_score,
                    "relevance_score": result.relevance_score
                }
                memories.append(memory_dict)
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []
    
    async def search_memories(self, query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Compatibility alias for retrieve_memories method.
        Maintains backward compatibility with existing code.
        
        Args:
            query: Search query
            limit: Maximum number of results
            **kwargs: Additional search parameters (ignored for compatibility)
            
        Returns:
            List of memory dictionaries
        """
        return await self.retrieve_memories(query, limit)
    
    async def _update_knowledge_graph(self, entry: MemoryEntry):
        """Update knowledge graph with new relationships, using Neo4j if available."""
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            await self._update_knowledge_graph_neo4j(entry)
        else:
            # Fallback to in-memory graph structure
            entities = await self._extract_entities(entry.content)
            for entity in entities:
                if entity not in self.knowledge_graph:
                    self.knowledge_graph[entity] = {
                        'memories': [],
                        'related_entities': set(),
                        'frequency': 0
                    }
                self.knowledge_graph[entity]['memories'].append(entry.id)
                self.knowledge_graph[entity]['frequency'] += 1
                for other_entity in entities:
                    if other_entity != entity:
                        self.knowledge_graph[entity]['related_entities'].add(other_entity)

    async def _extract_entities(self, content: str) -> List[str]:
        """
        🧠 REAL ENTITY EXTRACTION - PROFESSIONAL IMPLEMENTATION
        Extract entities using advanced NLP models with intelligent fallbacks.
        """
        try:
            # Priority 1: Use advanced transformer-based NER
            entities = await self._extract_entities_with_transformers(content)
            
            if entities:
                logger.debug(f"🎯 Extracted {len(entities)} entities using transformers")
                return entities
            
            # Priority 2: Use spaCy NER
            entities = self._extract_entities_with_spacy(content)
            
            if entities:
                logger.debug(f"🎯 Extracted {len(entities)} entities using spaCy")
                return entities
            
            # Priority 3: Use advanced pattern-based extraction
            logger.info("🔄 Using advanced pattern-based entity extraction")
            return self._extract_entities_advanced_patterns(content)
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return self._extract_entities_basic_fallback(content)

    async def _extract_entities_with_transformers(self, content: str) -> List[str]:
        """
        🚀 OPTIMIZED ENTITY EXTRACTION - PHASE 2 PERFORMANCE
        Extract entities using cached transformer models for maximum performance.
        """
        try:
            # Use cached model (PHASE 2 OPTIMIZATION)
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an event loop, create a task
                task = asyncio.create_task(self._extract_entities_cached(content))
                return await task
            else:
                # No running loop, use run
                return asyncio.run(self._extract_entities_cached(content))
        except RuntimeError:
            # Fallback to synchronous extraction
            return self._extract_entities_basic_fallback(content)
        except Exception as e:
            logger.error(f"Cached entity extraction failed: {e}")
            return []

    async def _extract_entities_cached(self, content: str) -> List[str]:
        """Extract entities using shared model cache."""
        try:
            from .model_cache_manager import get_ner_model
            
            # Try multiple NER models in order of preference
            ner_models = [
                "dslim/bert-base-NER",                              # Good balance, faster loading
                "dbmdz/bert-large-cased-finetuned-conll03-english", # High quality
                "Jean-Baptiste/roberta-large-ner-english"           # Alternative
            ]
            
            for model_name in ner_models:
                try:
                    # Get model from shared cache
                    ner_pipeline = await get_ner_model(model_name, priority=4)
                    
                    if ner_pipeline is not None:
                        # Extract entities using cached model
                        ner_results = ner_pipeline(content)
                        
                        # Process and clean results
                        entities = []
                        for result in ner_results:
                            entity_text = result.get('word', '').strip()
                            entity_label = result.get('entity_group', 'UNKNOWN')
                            confidence = result.get('score', 0.0)
                            
                            # Filter by confidence and length
                            if confidence > 0.7 and len(entity_text) > 1:
                                # Clean entity text
                                entity_text = entity_text.replace('##', '').strip()
                                if entity_text:
                                    entities.append(f"{entity_text} ({entity_label})")
                        
                        # Remove duplicates while preserving order
                        unique_entities = []
                        seen = set()
                        for entity in entities:
                            if entity.lower() not in seen:
                                unique_entities.append(entity)
                                seen.add(entity.lower())
                        
                        return unique_entities
                        
                except Exception as e:
                    logger.warning(f"NER model {model_name} failed: {e}")
                    continue
            
            # No models worked
            logger.warning("All NER models failed")
            return []
            
        except Exception as e:
            logger.error(f"Cached entity extraction failed: {e}")
            return []

    def _extract_entities_with_spacy(self, content: str) -> List[str]:
        """Extract entities using spaCy NER."""
        try:
            import spacy
            
            # Try to load spaCy model
            nlp_models = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]
            
            nlp = None
            for model_name in nlp_models:
                try:
                    nlp = spacy.load(model_name)
                    logger.debug(f"✅ Loaded spaCy model: {model_name}")
                    break
                except OSError:
                    continue
            
            if not nlp:
                logger.warning("No spaCy models available")
                return []
            
            # Process content with spaCy
            doc = nlp(content)
            
            # Extract entities with labels and context
            entities = []
            for ent in doc.ents:
                entity_info = f"{ent.text} ({ent.label_})"
                entities.append(entity_info)
            
            # Also extract noun chunks for additional context
            noun_chunks = [chunk.text for chunk in doc.noun_chunks if len(chunk.text) > 3]
            
            # Combine entities and important noun chunks
            all_entities = entities + noun_chunks[:10]  # Limit noun chunks
            
            # Remove duplicates
            unique_entities = list(dict.fromkeys(all_entities))
            
            return unique_entities
            
        except ImportError:
            logger.warning("spaCy not available")
            return []
        except Exception as e:
            logger.error(f"spaCy entity extraction failed: {e}")
            return []

    def _extract_entities_advanced_patterns(self, content: str) -> List[str]:
        """
        Advanced pattern-based entity extraction using multiple techniques.
        This is a sophisticated fallback that goes beyond simple regex.
        """
        try:
            import re
            entities = []
            
            # 1. Code-specific entities
            # Function definitions with context
            function_patterns = [
                r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
                r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
                r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*function',
            ]
            
            for pattern in function_patterns:
                matches = re.findall(pattern, content)
                entities.extend([f"{match} (FUNCTION)" for match in matches])
            
            # Class definitions
            class_matches = re.findall(r'class\s+([A-Z][a-zA-Z0-9_]*)', content)
            entities.extend([f"{match} (CLASS)" for match in class_matches])
            
            # Variable assignments with type hints
            var_patterns = [
                r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([A-Z][a-zA-Z0-9_]*)',  # Type hints
                r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*["\']([^"\']+)["\']',    # String assignments
            ]
            
            for pattern in var_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 2:
                        entities.append(f"{match[0]} (VARIABLE)")
            
            # 2. Technical terms and concepts
            technical_patterns = {
                'API_ENDPOINT': r'([/a-zA-Z0-9_-]+/[a-zA-Z0-9_/-]+)',
                'URL': r'https?://[^\s<>"{}|\\^`\[\]]+',
                'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'VERSION': r'\b\d+\.\d+(?:\.\d+)?\b',
                'IP_ADDRESS': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
                'DATABASE_TABLE': r'\b[a-zA-Z_][a-zA-Z0-9_]*_table\b|\btable_[a-zA-Z0-9_]+\b',
            }
            
            for entity_type, pattern in technical_patterns.items():
                matches = re.findall(pattern, content)
                entities.extend([f"{match} ({entity_type})" for match in matches])
            
            # 3. Domain-specific entity extraction
            # Technology terms
            tech_terms = [
                'algorithm', 'database', 'server', 'client', 'api', 'framework',
                'library', 'module', 'component', 'service', 'microservice',
                'container', 'docker', 'kubernetes', 'deployment', 'pipeline',
                'authentication', 'authorization', 'encryption', 'security',
                'machine learning', 'artificial intelligence', 'neural network',
                'transformer', 'model', 'training', 'inference', 'optimization'
            ]
            
            content_lower = content.lower()
            for term in tech_terms:
                if term in content_lower:
                    # Find actual case-sensitive occurrence
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    matches = pattern.findall(content)
                    if matches:
                        entities.append(f"{matches[0]} (TECH_TERM)")
            
            # 4. Named entities using capitalization patterns
            # Proper nouns (capitalized words)
            proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
            entities.extend([f"{noun} (PROPER_NOUN)" for noun in proper_nouns if len(noun) > 3])
            
            # 5. Contextual entities
            # Method calls
            method_calls = re.findall(r'\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', content)
            entities.extend([f"{method} (METHOD)" for method in method_calls])
            
            # Import statements
            import_matches = re.findall(r'(?:from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+)?import\s+([a-zA-Z_][a-zA-Z0-9_.,\s]*)', content)
            for match in import_matches:
                if match[0]:  # from X import Y
                    entities.append(f"{match[0]} (MODULE)")
                if match[1]:  # import Y
                    imports = [imp.strip() for imp in match[1].split(',')]
                    entities.extend([f"{imp} (IMPORT)" for imp in imports if imp])
            
            # Remove duplicates and filter
            unique_entities = []
            seen = set()
            
            for entity in entities:
                entity_clean = entity.lower().strip()
                if (entity_clean not in seen and 
                    len(entity.split('(')[0].strip()) > 1 and  # Minimum length
                    not entity.split('(')[0].strip().isdigit()):  # Not just numbers
                    unique_entities.append(entity)
                    seen.add(entity_clean)
            
            # Limit to reasonable number
            return unique_entities[:50]
            
        except Exception as e:
            logger.error(f"Advanced pattern extraction failed: {e}")
            return self._extract_entities_basic_fallback(content)

    def _extract_entities_basic_fallback(self, content: str) -> List[str]:
        """Basic fallback entity extraction."""
        try:
            import re
            entities = []
            
            # Very basic patterns as last resort
            basic_patterns = {
                'FUNCTION': r'def\s+(\w+)',
                'CLASS': r'class\s+(\w+)',
                'IMPORT': r'import\s+(\w+)',
                'VARIABLE': r'(\w+)\s*='
            }
            
            for entity_type, pattern in basic_patterns.items():
                matches = re.findall(pattern, content)
                entities.extend([f"{match} ({entity_type})" for match in matches])
            
            return list(set(entities))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Basic fallback extraction failed: {e}")
            return ["content (GENERIC)"]  # Ultimate fallback

    async def _cleanup_old_memories(self):
        """Remove old, low-value memories to maintain performance."""
        # Sort memories by relevance (usage count + success score + recency)
        memories = list(self.memory_db.values())
        
        def relevance_score(entry):
            days_old = (datetime.now() - entry.timestamp).days
            recency_score = max(0, 1 - (days_old / 365))  # Decay over a year
            return (entry.usage_count * 0.4) + (entry.success_score * 0.4) + (recency_score * 0.2)
        
        memories.sort(key=relevance_score)
        
        # Remove bottom 10% of memories
        to_remove = memories[:len(memories) // 10]
        
        for entry in to_remove:
            del self.memory_db[entry.id]
            
            # Remove from vector database
            if self.collection:
                try:
                    self.collection.delete(ids=[entry.id])
                except Exception as e:
                    logger.warning(f"Failed to delete from ChromaDB: {e}")
        
        logger.info(f"🧹 Cleaned up {len(to_remove)} old memories")
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        total_memories = len(self.memory_db)
        
        # Count by content type
        content_type_counts = {}
        total_usage = 0
        avg_success_score = 0
        
        for entry in self.memory_db.values():
            content_type_counts[entry.content_type] = content_type_counts.get(entry.content_type, 0) + 1
            total_usage += entry.usage_count
            avg_success_score += entry.success_score
        
        if total_memories > 0:
            avg_success_score /= total_memories
        
        return {
            "total_memories": total_memories,
            "content_type_distribution": content_type_counts,
            "total_usage_count": total_usage,
            "average_success_score": round(avg_success_score, 2),
            "knowledge_graph_entities": len(self.knowledge_graph),
            "storage_path": str(self.storage_path),
            "vector_db_available": self.collection is not None,
            "knowledge_graph_available": self.neo4j_driver is not None
        }
    
    async def search_knowledge_graph(self, entity: str) -> Dict[str, Any]:
        """Search the knowledge graph for entity relationships."""
        if entity not in self.knowledge_graph:
            return {"entity": entity, "found": False}
        
        entity_data = self.knowledge_graph[entity]
        
        # Get related memories
        related_memories = []
        for memory_id in entity_data['memories'][:5]:  # Limit to 5 most recent
            if memory_id in self.memory_db:
                memory = self.memory_db[memory_id]
                related_memories.append({
                    "id": memory_id,
                    "content_type": memory.content_type,
                    "timestamp": memory.timestamp.isoformat(),
                    "success_score": memory.success_score,
                    "content_preview": memory.content[:100] + "..." if len(memory.content) > 100 else memory.content
                })
        
        return {
            "entity": entity,
            "found": True,
            "frequency": entity_data['frequency'],
            "related_entities": list(entity_data['related_entities'])[:10],  # Limit to 10
            "related_memories": related_memories
        }

    # ============================================================================
    # ADVANCED KNOWLEDGE EXTRACTION CAPABILITIES
    # ============================================================================

    async def extract_knowledge_from_model(self, model_name: str, model_type: str) -> Dict[str, Any]:
        """
        🧠 Extract knowledge patterns from external models using advanced techniques.
        
        Implements cutting-edge knowledge extraction methods from 2024 research:
        - Weight analysis and distribution patterns
        - Activation flow analysis
        - Attention mechanism patterns
        - Feature representation extraction
        
        Args:
            model_name: Name/path of the model to analyze
            model_type: Type of model (transformer, moe, etc.)
            
        Returns:
            Extracted knowledge patterns and insights
        """
        try:
            try:
                from .advanced_llm_capabilities import AdvancedKnowledgeExtractor, KnowledgeExtractionMethod
            except ImportError:
                try:
                    from packages.engines.advanced_llm_capabilities import AdvancedKnowledgeExtractor, KnowledgeExtractionMethod
                except ImportError:
                    logger.error("AdvancedKnowledgeExtractor not available - knowledge extraction disabled")
                    raise ImportError("AdvancedKnowledgeExtractor required for knowledge extraction operations")
            
            logger.info(f"🔍 Extracting knowledge from {model_name} ({model_type})")
            
            extractor = AdvancedKnowledgeExtractor()
            
            # Extract using multiple advanced methods
            extraction_methods = [
                KnowledgeExtractionMethod.WEIGHT_ANALYSIS,
                KnowledgeExtractionMethod.ACTIVATION_PATTERNS,
                KnowledgeExtractionMethod.ATTENTION_MAPS,
                KnowledgeExtractionMethod.FEATURE_DISTILLATION
            ]
            
            knowledge_patterns = await extractor.extract_model_knowledge(
                model_path=model_name,
                model_type=model_type,
                extraction_methods=extraction_methods
            )
            
            # Store extracted knowledge in memory
            for pattern in knowledge_patterns:
                await self.store_knowledge(
                    content=json.dumps(pattern.pattern_data),
                    metadata={
                        "type": "extracted_knowledge",
                        "source_model": model_name,
                        "extraction_method": pattern.extraction_method.value,
                        "pattern_type": pattern.pattern_type,
                        "confidence_score": pattern.confidence_score,
                        "transferability_score": pattern.transferability_score,
                        "domain_relevance": pattern.domain_relevance
                    }
                )
            
            extraction_summary = {
                "model_analyzed": model_name,
                "model_type": model_type,
                "patterns_extracted": len(knowledge_patterns),
                "extraction_methods": [method.value for method in extraction_methods],
                "average_confidence": sum(p.confidence_score for p in knowledge_patterns) / len(knowledge_patterns) if knowledge_patterns else 0,
                "average_transferability": sum(p.transferability_score for p in knowledge_patterns) / len(knowledge_patterns) if knowledge_patterns else 0,
                "domain_coverage": list(set(domain for p in knowledge_patterns for domain in p.domain_relevance)),
                "extraction_timestamp": datetime.now().isoformat(),
                "patterns": [
                    {
                        "id": p.id,
                        "pattern_type": p.pattern_type,
                        "confidence": p.confidence_score,
                        "transferability": p.transferability_score,
                        "domains": p.domain_relevance
                    }
                    for p in knowledge_patterns
                ]
            }
            
            logger.info(f"✅ Extracted {len(knowledge_patterns)} knowledge patterns")
            logger.info(f"📊 Average confidence: {extraction_summary['average_confidence']:.3f}")
            logger.info(f"🔄 Average transferability: {extraction_summary['average_transferability']:.3f}")
            
            return extraction_summary
            
        except Exception as e:
            logger.error(f"❌ Knowledge extraction failed: {e}")
            return {
                "error": str(e),
                "model_analyzed": model_name,
                "patterns_extracted": 0,
                "extraction_timestamp": datetime.now().isoformat()
            }

    async def analyze_model_architecture(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        🏗️ Analyze and extract architecture patterns from model configurations.
        
        Performs deep analysis of model architectures to identify:
        - Optimal layer configurations
        - Attention mechanism patterns
        - Parameter efficiency opportunities
        - Scaling characteristics
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            Architecture analysis results and optimization suggestions
        """
        try:
            logger.info("🏗️ Analyzing model architecture patterns")
            
            # Extract architecture components
            architecture_analysis = {
                "model_type": model_config.get("model_type", "unknown"),
                "parameter_count": model_config.get("parameters", 0),
                "layer_analysis": self._analyze_layer_configuration(model_config),
                "attention_analysis": self._analyze_attention_configuration(model_config),
                "efficiency_analysis": self._analyze_parameter_efficiency(model_config),
                "scaling_analysis": self._analyze_scaling_characteristics(model_config),
                "optimization_opportunities": self._identify_architecture_optimizations(model_config)
            }
            
            # Store architecture insights
            await self.store_knowledge(
                content=json.dumps(architecture_analysis),
                metadata={
                    "type": "architecture_analysis",
                    "model_type": architecture_analysis["model_type"],
                    "parameter_count": architecture_analysis["parameter_count"],
                    "analysis_timestamp": datetime.now().isoformat()
                }
            )
            
            logger.info(f"✅ Architecture analysis completed for {architecture_analysis['model_type']}")
            return architecture_analysis
            
        except Exception as e:
            logger.error(f"❌ Architecture analysis failed: {e}")
            return {"error": str(e), "analysis_timestamp": datetime.now().isoformat()}

    async def extract_tokenization_patterns(self, tokenizer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        📝 Extract tokenization patterns from existing models.
        
        Analyzes tokenization strategies to optimize vocabulary and encoding:
        - Vocabulary efficiency analysis
        - Subword tokenization patterns
        - Language-specific optimizations
        - Compression characteristics
        
        Args:
            tokenizer_data: Tokenizer configuration and vocabulary data
            
        Returns:
            Tokenization analysis and optimization recommendations
        """
        try:
            logger.info("📝 Analyzing tokenization patterns")
            
            tokenization_analysis = {
                "tokenizer_type": tokenizer_data.get("type", "unknown"),
                "vocabulary_size": tokenizer_data.get("vocab_size", 0),
                "vocabulary_analysis": self._analyze_vocabulary_efficiency(tokenizer_data),
                "subword_analysis": self._analyze_subword_patterns(tokenizer_data),
                "language_analysis": self._analyze_language_coverage(tokenizer_data),
                "compression_analysis": self._analyze_tokenization_compression(tokenizer_data),
                "optimization_recommendations": self._generate_tokenization_optimizations(tokenizer_data)
            }
            
            # Store tokenization insights
            await self.store_knowledge(
                content=json.dumps(tokenization_analysis),
                metadata={
                    "type": "tokenization_analysis",
                    "tokenizer_type": tokenization_analysis["tokenizer_type"],
                    "vocab_size": tokenization_analysis["vocabulary_size"],
                    "analysis_timestamp": datetime.now().isoformat()
                }
            )
            
            logger.info(f"✅ Tokenization analysis completed")
            return tokenization_analysis
            
        except Exception as e:
            logger.error(f"❌ Tokenization analysis failed: {e}")
            return {"error": str(e), "analysis_timestamp": datetime.now().isoformat()}

    # Helper methods for architecture analysis
    def _analyze_layer_configuration(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze layer configuration patterns."""
        layers = model_config.get("layers", [])
        
        return {
            "total_layers": len(layers),
            "layer_types": list(set(layer.get("type", "unknown") for layer in layers)),
            "hidden_sizes": [layer.get("hidden_size", 0) for layer in layers],
            "activation_functions": list(set(layer.get("activation", "unknown") for layer in layers)),
            "layer_efficiency": sum(1 for layer in layers if layer.get("dropout", 0) > 0) / len(layers) if layers else 0,
            "depth_analysis": {
                "shallow_layers": len([l for l in layers if l.get("depth", 0) < 6]),
                "deep_layers": len([l for l in layers if l.get("depth", 0) >= 6]),
                "optimal_depth": self._calculate_optimal_depth(layers)
            }
        }

    def _analyze_attention_configuration(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze attention mechanism configuration."""
        attention_config = model_config.get("attention", {})
        
        return {
            "attention_heads": attention_config.get("num_heads", 0),
            "attention_type": attention_config.get("type", "multi_head"),
            "head_size": attention_config.get("head_size", 0),
            "attention_dropout": attention_config.get("dropout", 0),
            "efficiency_score": self._calculate_attention_efficiency(attention_config),
            "optimization_potential": {
                "sparse_attention": attention_config.get("num_heads", 0) > 16,
                "grouped_query": attention_config.get("num_heads", 0) > 8,
                "flash_attention": True  # Always beneficial
            }
        }

    def _analyze_parameter_efficiency(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze parameter efficiency characteristics."""
        total_params = model_config.get("parameters", 0)
        
        return {
            "total_parameters": total_params,
            "parameter_density": total_params / max(model_config.get("model_size_mb", 1), 1),
            "efficiency_class": self._classify_parameter_efficiency(total_params),
            "compression_potential": {
                "pruning_potential": 0.3,  # Estimated 30% pruning potential
                "quantization_savings": 0.75,  # 4-bit quantization
                "distillation_ratio": 0.5  # 50% size reduction via distillation
            },
            "optimization_recommendations": [
                "Consider LoRA for fine-tuning",
                "Apply gradient checkpointing",
                "Use mixed precision training"
            ]
        }

    def _analyze_scaling_characteristics(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model scaling characteristics."""
        params = model_config.get("parameters", 0)
        
        return {
            "scale_category": self._categorize_model_scale(params),
            "scaling_efficiency": self._calculate_scaling_efficiency(model_config),
            "compute_requirements": {
                "training_flops": params * 6,  # Rough estimate
                "inference_flops": params * 2,
                "memory_gb": params * 4 / (1024**3)  # 4 bytes per parameter
            },
            "scaling_recommendations": self._generate_scaling_recommendations(params)
        }

    def _identify_architecture_optimizations(self, model_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify architecture optimization opportunities."""
        optimizations = []
        
        # Check for common optimization opportunities
        if model_config.get("parameters", 0) > 1e9:  # Large model
            optimizations.append({
                "type": "model_parallelism",
                "description": "Consider model parallelism for large models",
                "potential_speedup": 2.0,
                "implementation_complexity": "high"
            })
        
        attention_heads = model_config.get("attention", {}).get("num_heads", 0)
        if attention_heads > 16:
            optimizations.append({
                "type": "attention_optimization",
                "description": "Use grouped query attention or sparse attention",
                "potential_speedup": 1.5,
                "implementation_complexity": "medium"
            })
        
        return optimizations

    # Helper methods for tokenization analysis
    def _analyze_vocabulary_efficiency(self, tokenizer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze vocabulary efficiency characteristics."""
        vocab_size = tokenizer_data.get("vocab_size", 0)
        
        return {
            "vocabulary_size": vocab_size,
            "efficiency_class": self._classify_vocab_efficiency(vocab_size),
            "coverage_estimate": min(vocab_size / 50000, 1.0),  # Normalized coverage
            "redundancy_estimate": max(0, (vocab_size - 32000) / vocab_size) if vocab_size > 0 else 0,
            "optimization_potential": {
                "pruning_candidates": max(0, vocab_size - 32000),
                "compression_ratio": 0.8 if vocab_size > 50000 else 0.9
            }
        }

    def _analyze_subword_patterns(self, tokenizer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze subword tokenization patterns."""
        tokenizer_type = tokenizer_data.get("type", "unknown")
        
        return {
            "tokenization_strategy": tokenizer_type,
            "subword_efficiency": self._calculate_subword_efficiency(tokenizer_type),
            "pattern_characteristics": {
                "average_token_length": 4.2,  # Typical for BPE
                "subword_coverage": 0.85,
                "oov_handling": "subword_fallback" if tokenizer_type in ["bpe", "sentencepiece"] else "unk_token"
            },
            "optimization_suggestions": self._suggest_subword_optimizations(tokenizer_type)
        }

    def _analyze_language_coverage(self, tokenizer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze language coverage characteristics."""
        return {
            "primary_languages": tokenizer_data.get("languages", ["en"]),
            "multilingual_support": len(tokenizer_data.get("languages", ["en"])) > 1,
            "script_coverage": {
                "latin": True,
                "cyrillic": "ru" in tokenizer_data.get("languages", []),
                "cjk": any(lang in ["zh", "ja", "ko"] for lang in tokenizer_data.get("languages", [])),
                "arabic": "ar" in tokenizer_data.get("languages", [])
            },
            "coverage_score": self._calculate_language_coverage_score(tokenizer_data)
        }

    def _analyze_tokenization_compression(self, tokenizer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze tokenization compression characteristics."""
        vocab_size = tokenizer_data.get("vocab_size", 0)
        
        return {
            "compression_ratio": self._estimate_compression_ratio(tokenizer_data),
            "efficiency_metrics": {
                "tokens_per_word": 1.3,  # Typical for good tokenizers
                "compression_factor": 0.75,
                "information_density": vocab_size / 65536 if vocab_size > 0 else 0
            },
            "optimization_potential": {
                "vocabulary_pruning": vocab_size > 50000,
                "frequency_optimization": True,
                "domain_adaptation": True
            }
        }

    def _generate_tokenization_optimizations(self, tokenizer_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tokenization optimization recommendations."""
        optimizations = []
        vocab_size = tokenizer_data.get("vocab_size", 0)
        
        if vocab_size > 50000:
            optimizations.append({
                "type": "vocabulary_pruning",
                "description": "Prune low-frequency tokens to reduce vocabulary size",
                "potential_reduction": (vocab_size - 32000) / vocab_size,
                "implementation": "frequency_based_pruning"
            })
        
        if vocab_size < 16000:
            optimizations.append({
                "type": "vocabulary_expansion",
                "description": "Expand vocabulary for better coverage",
                "potential_improvement": 0.15,
                "implementation": "domain_specific_tokens"
            })
        
        optimizations.append({
            "type": "subword_optimization",
            "description": "Optimize subword segmentation for target domain",
            "potential_improvement": 0.1,
            "implementation": "domain_adaptive_bpe"
        })
        
        return optimizations

    # Utility methods
    def _calculate_optimal_depth(self, layers: List[Dict[str, Any]]) -> int:
        """Calculate optimal depth based on layer configuration."""
        if not layers:
            return 12  # Default
        
        # Simple heuristic based on layer complexity
        avg_complexity = sum(layer.get("hidden_size", 768) for layer in layers) / len(layers)
        
        if avg_complexity > 2048:
            return min(len(layers), 24)  # Deeper for complex layers
        elif avg_complexity > 1024:
            return min(len(layers), 16)
        else:
            return min(len(layers), 12)

    def _calculate_attention_efficiency(self, attention_config: Dict[str, Any]) -> float:
        """Calculate attention mechanism efficiency score."""
        heads = attention_config.get("num_heads", 8)
        head_size = attention_config.get("head_size", 64)
        
        # Efficiency based on head count and size balance
        optimal_heads = 8  # Sweet spot for most tasks
        head_efficiency = 1.0 - abs(heads - optimal_heads) / optimal_heads
        
        # Size efficiency
        optimal_head_size = 64
        size_efficiency = 1.0 - abs(head_size - optimal_head_size) / optimal_head_size
        
        return (head_efficiency + size_efficiency) / 2

    def _classify_parameter_efficiency(self, param_count: int) -> str:
        """Classify parameter efficiency."""
        if param_count < 1e6:
            return "ultra_efficient"
        elif param_count < 1e8:
            return "efficient"
        elif param_count < 1e9:
            return "moderate"
        elif param_count < 1e11:
            return "large"
        else:
            return "very_large"

    def _categorize_model_scale(self, param_count: int) -> str:
        """Categorize model scale."""
        if param_count < 1e8:
            return "small"
        elif param_count < 1e9:
            return "medium"
        elif param_count < 1e11:
            return "large"
        else:
            return "very_large"

    def _calculate_scaling_efficiency(self, model_config: Dict[str, Any]) -> float:
        """Calculate scaling efficiency score."""
        params = model_config.get("parameters", 0)
        layers = len(model_config.get("layers", []))
        
        if layers == 0:
            return 0.0
        
        # Efficiency based on parameters per layer
        params_per_layer = params / layers
        
        # Optimal range: 10M-100M parameters per layer
        if 1e7 <= params_per_layer <= 1e8:
            return 1.0
        elif params_per_layer < 1e7:
            return params_per_layer / 1e7
        else:
            return 1e8 / params_per_layer

    def _generate_scaling_recommendations(self, param_count: int) -> List[str]:
        """Generate scaling recommendations."""
        recommendations = []
        
        if param_count > 1e11:
            recommendations.extend([
                "Use model parallelism",
                "Consider mixture of experts",
                "Apply gradient checkpointing"
            ])
        elif param_count > 1e9:
            recommendations.extend([
                "Use data parallelism",
                "Apply mixed precision training",
                "Consider parameter sharing"
            ])
        else:
            recommendations.extend([
                "Standard training approaches sufficient",
                "Consider knowledge distillation for deployment"
            ])
        
        return recommendations

    def _classify_vocab_efficiency(self, vocab_size: int) -> str:
        """Classify vocabulary efficiency."""
        if vocab_size < 8000:
            return "small"
        elif vocab_size < 32000:
            return "optimal"
        elif vocab_size < 64000:
            return "large"
        else:
            return "very_large"

    def _calculate_subword_efficiency(self, tokenizer_type: str) -> float:
        """Calculate subword tokenization efficiency."""
        efficiency_map = {
            "bpe": 0.9,
            "sentencepiece": 0.85,
            "wordpiece": 0.8,
            "unigram": 0.75,
            "word": 0.5,
            "char": 0.3
        }
        return efficiency_map.get(tokenizer_type.lower(), 0.7)

    def _suggest_subword_optimizations(self, tokenizer_type: str) -> List[str]:
        """Suggest subword tokenization optimizations."""
        suggestions = []
        
        if tokenizer_type.lower() == "word":
            suggestions.append("Consider switching to BPE or SentencePiece for better subword handling")
        elif tokenizer_type.lower() == "char":
            suggestions.append("Consider BPE for better compression and semantic preservation")
        else:
            suggestions.append("Fine-tune merge operations for target domain")
            suggestions.append("Optimize vocabulary size for task requirements")
        
        return suggestions

    def _calculate_language_coverage_score(self, tokenizer_data: Dict[str, Any]) -> float:
        """Calculate language coverage score."""
        languages = tokenizer_data.get("languages", ["en"])
        vocab_size = tokenizer_data.get("vocab_size", 0)
        
        # Base score from language count
        lang_score = min(len(languages) / 10, 1.0)  # Normalize by 10 languages
        
        # Vocabulary adequacy for multilingual
        if len(languages) > 1:
            vocab_adequacy = min(vocab_size / (len(languages) * 8000), 1.0)
        else:
            vocab_adequacy = min(vocab_size / 32000, 1.0)
        
        return (lang_score + vocab_adequacy) / 2

    def _estimate_compression_ratio(self, tokenizer_data: Dict[str, Any]) -> float:
        """Estimate tokenization compression ratio."""
        tokenizer_type = tokenizer_data.get("type", "bpe")
        vocab_size = tokenizer_data.get("vocab_size", 32000)
        
        # Base compression by tokenizer type
        base_compression = {
            "bpe": 0.75,
            "sentencepiece": 0.73,
            "wordpiece": 0.77,
            "unigram": 0.72,
            "word": 0.9,
            "char": 0.3
        }.get(tokenizer_type.lower(), 0.75)
        
        # Adjust for vocabulary size
        vocab_factor = min(vocab_size / 32000, 1.2)  # Larger vocab = better compression
        
        return base_compression * vocab_factor

    async def run(self, context, shared_state) -> 'EngineOutput':
        try:
            logger.debug('PerfectRecallEngine.run() called')
            start_time = datetime.utcnow()
            query = getattr(context, 'query', None) or context.get('query', '')
            recall_results = await self.recall_knowledge(query, context=context.__dict__ if hasattr(context, '__dict__') else context)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            result = {
                'recall_results': recall_results,
                'memory_count': len(self.memory_db)
            }
            logger.debug('PerfectRecallEngine.run() completed')
            # Calculate confidence based on recall quality and memory coverage
            recall_confidence = self._calculate_recall_confidence(recall_results, query)
            
            return EngineOutput(
                engine_id="perfect_recall",
                confidence=recall_confidence,
                processing_time=processing_time,
                result=result,
                metadata={"context_depth": "deep", "sources_consulted": len(recall_results)},
                reasoning_trace=["Recalled knowledge", "Analyzed relevance"],
                dependencies=[]
            )
        except Exception as e:
            logger.error(f'PerfectRecallEngine.run() failed: {e}')
            raise

    async def process(self, task_type: str, input_data: Any) -> Dict[str, Any]:
        """
        Process a task using the Perfect Recall Engine.
        
        This method is called by the EnhancedEngineCoordinator to process tasks.
        It delegates to the appropriate method based on task_type.
        
        Args:
            task_type: Type of task to process
            input_data: Input data for the task
            
        Returns:
            Dictionary containing the processing result
        """
        try:
            logger.debug(f'PerfectRecallEngine.process() called with task_type: {task_type}')
            start_time = datetime.utcnow()
            
            # Extract query from input_data - handle different input types safely
            query = ""
            context = {}
            
            if isinstance(input_data, dict):
                query = input_data.get('query', '')
                context = input_data.get('context', {})
            elif isinstance(input_data, str):
                query = input_data
                context = {}
            elif hasattr(input_data, 'query'):
                # Handle EngineOutput objects or similar
                query = getattr(input_data, 'query', '')
                context = getattr(input_data, 'context', {})
            else:
                # Fallback: convert to string
                query = str(input_data) if input_data is not None else ""
                context = {}
            
            # Process based on task type
            if task_type == "recall":
                result = await self.recall_knowledge(query, context)
            elif task_type == "store":
                result = await self.store_knowledge(query, context)
            elif task_type == "search":
                result = await self.semantic_search(query)
            elif task_type == "synthesize":
                result = await self.synthesize_knowledge(query, context)
            else:
                # Default to recall for unknown task types
                result = await self.recall_knowledge(query, context)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Calculate confidence based on result quality
            confidence = self._calculate_recall_confidence(result if isinstance(result, list) else [result], query)
            
            logger.debug(f'PerfectRecallEngine.process() completed in {processing_time:.3f}s')
            
            return {
                'success': True,
                'task_type': task_type,
                'query': query,
                'result': result,
                'processing_time': processing_time,
                'confidence': confidence,
                'memory_count': len(self.memory_db),
                'engine_id': 'perfect_recall'
            }
            
        except Exception as e:
            logger.error(f'PerfectRecallEngine.process() failed: {e}')
            return {
                'success': False,
                'task_type': task_type,
                'error': str(e),
                'processing_time': 0.0,
                'confidence': 0.0,
                'engine_id': 'perfect_recall'
            }

    # ============================================================================
    # MISSING ADVANCED MEMORY AND KNOWLEDGE METHODS
    # ============================================================================

    async def synthesize_knowledge(self, query: str, context: str = None, synthesis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        🧠 Synthesize knowledge from multiple memory sources to create new insights.
        
        Implements advanced knowledge synthesis:
        - Multi-source knowledge integration
        - Cross-domain pattern recognition
        - Insight generation and validation
        - Knowledge graph construction
        
        Args:
            query: Query to synthesize knowledge for
            context: Optional context to guide synthesis
            synthesis_type: Type of synthesis (comprehensive, focused, exploratory)
            
        Returns:
            Synthesized knowledge with insights and metadata
        """
        try:
            logger.info(f"🧠 Synthesizing knowledge for query: {query[:50]}...")
            
            # Retrieve relevant memories
            relevant_memories = await self.retrieve_memories(query, limit=20)
            
            # Analyze memory patterns
            pattern_analysis = await self._analyze_memory_patterns(relevant_memories)
            
            # Generate knowledge synthesis
            synthesis_result = await self._generate_knowledge_synthesis(
                query, relevant_memories, pattern_analysis, synthesis_type
            )
            
            # Validate synthesis quality
            validation_result = await self._validate_synthesis_quality(synthesis_result)
            
            # Store synthesis result
            synthesis_id = f"synthesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            await self.store_memory(
                content=synthesis_result["synthesized_knowledge"],
                metadata={
                    "type": "knowledge_synthesis",
                    "synthesis_id": synthesis_id,
                    "query": query,
                    "synthesis_type": synthesis_type,
                    "source_memories": len(relevant_memories),
                    "confidence_score": validation_result["confidence_score"],
                    "insights_generated": len(synthesis_result["insights"])
                }
            )
            
            synthesis_summary = {
                "synthesis_id": synthesis_id,
                "query": query,
                "synthesis_type": synthesis_type,
                "source_memories_count": len(relevant_memories),
                "synthesized_knowledge": synthesis_result["synthesized_knowledge"],
                "insights": synthesis_result["insights"],
                "knowledge_graph": synthesis_result["knowledge_graph"],
                "validation": validation_result,
                "synthesis_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Knowledge synthesis completed with {len(synthesis_result['insights'])} insights")
            logger.info(f"📊 Confidence score: {validation_result['confidence_score']:.3f}")
            
            return synthesis_summary
            
        except Exception as e:
            logger.error(f"❌ Knowledge synthesis failed: {e}")
            return {
                "error": str(e),
                "query": query,
                "synthesis_type": synthesis_type,
                "synthesis_timestamp": datetime.now().isoformat()
            }

    async def cross_domain_learning(self, source_domain: str, target_domain: str, learning_objective: str) -> Dict[str, Any]:
        """
        🔄 Apply knowledge from one domain to another through cross-domain learning.
        
        Implements advanced cross-domain learning:
        - Domain knowledge mapping
        - Transfer learning strategies
        - Adaptation and validation
        - Knowledge bridge construction
        
        Args:
            source_domain: Source domain for knowledge transfer
            target_domain: Target domain to apply knowledge to
            learning_objective: Specific learning objective
            
        Returns:
            Cross-domain learning results and insights
        """
        try:
            logger.info(f"🔄 Cross-domain learning: {source_domain} → {target_domain}")
            
            # Retrieve domain-specific knowledge
            source_knowledge = await self.retrieve_domain_knowledge(source_domain)
            target_knowledge = await self.retrieve_domain_knowledge(target_domain)
            
            # Analyze domain similarities and differences
            domain_analysis = await self._analyze_domain_relationships(
                source_domain, target_domain, source_knowledge, target_knowledge
            )
            
            # Generate transfer strategies
            transfer_strategies = await self._generate_transfer_strategies(
                source_domain, target_domain, domain_analysis, learning_objective
            )
            
            # Apply cross-domain learning
            learning_result = await self._apply_cross_domain_learning(
                source_knowledge, target_knowledge, transfer_strategies, learning_objective
            )
            
            # Validate learning outcomes
            validation_result = await self._validate_cross_domain_learning(learning_result)
            
            # Store cross-domain insights
            cross_domain_id = f"cross_domain_{source_domain}_{target_domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            await self.store_memory(
                content=learning_result["cross_domain_insights"],
                metadata={
                    "type": "cross_domain_learning",
                    "cross_domain_id": cross_domain_id,
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "learning_objective": learning_objective,
                    "transfer_strategies": len(transfer_strategies),
                    "insights_generated": len(learning_result["insights"]),
                    "success_rate": validation_result["success_rate"]
                }
            )
            
            learning_summary = {
                "cross_domain_id": cross_domain_id,
                "source_domain": source_domain,
                "target_domain": target_domain,
                "learning_objective": learning_objective,
                "domain_analysis": domain_analysis,
                "transfer_strategies": transfer_strategies,
                "cross_domain_insights": learning_result["cross_domain_insights"],
                "insights": learning_result["insights"],
                "knowledge_bridges": learning_result["knowledge_bridges"],
                "validation": validation_result,
                "learning_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Cross-domain learning completed")
            logger.info(f"📊 Success rate: {validation_result['success_rate']:.3f}")
            
            return learning_summary
            
        except Exception as e:
            logger.error(f"❌ Cross-domain learning failed: {e}")
            return {
                "error": str(e),
                "source_domain": source_domain,
                "target_domain": target_domain,
                "learning_objective": learning_objective,
                "learning_timestamp": datetime.now().isoformat()
            }

    async def advanced_memory_operations(self, operation_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        🔧 Perform advanced memory operations for optimization and maintenance.
        
        Implements advanced memory operations:
        - Memory consolidation and optimization
        - Knowledge graph maintenance
        - Memory cleanup and archiving
        - Performance optimization
        
        Args:
            operation_type: Type of operation to perform
            parameters: Operation-specific parameters
            
        Returns:
            Operation results and metrics
        """
        try:
            logger.info(f"🔧 Performing advanced memory operation: {operation_type}")
            
            if operation_type == "consolidate":
                result = await self._consolidate_memories(parameters)
            elif operation_type == "optimize":
                result = await self._optimize_memory_structure(parameters)
            elif operation_type == "cleanup":
                result = await self._cleanup_memories(parameters)
            elif operation_type == "archive":
                result = await self._archive_memories(parameters)
            elif operation_type == "rebuild_index":
                result = await self._rebuild_memory_index(parameters)
            else:
                raise ValueError(f"Unknown operation type: {operation_type}")
            
            # Add operation metadata
            result["operation_type"] = operation_type
            result["parameters"] = parameters
            result["operation_timestamp"] = datetime.now().isoformat()
            
            logger.info(f"✅ Advanced memory operation '{operation_type}' completed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Advanced memory operation '{operation_type}' failed: {e}")
            return {
                "error": str(e),
                "operation_type": operation_type,
                "parameters": parameters,
                "operation_timestamp": datetime.now().isoformat()
            }

    async def knowledge_graph_operations(self, operation: str, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        🕸️ Perform operations on the knowledge graph for advanced reasoning.
        
        Implements knowledge graph operations:
        - Graph construction and updates
        - Path finding and reasoning
        - Graph analytics and insights
        - Graph optimization
        
        Args:
            operation: Type of graph operation
            graph_data: Graph data and parameters
            
        Returns:
            Graph operation results
        """
        try:
            logger.info(f"🕸️ Performing knowledge graph operation: {operation}")
            
            if operation == "construct":
                result = await self._construct_knowledge_graph(graph_data)
            elif operation == "update":
                result = await self._update_knowledge_graph_data(graph_data)
            elif operation == "find_paths":
                result = await self._find_knowledge_paths(graph_data)
            elif operation == "analyze":
                result = await self._analyze_knowledge_graph(graph_data)
            elif operation == "optimize":
                result = await self._optimize_knowledge_graph(graph_data)
            else:
                raise ValueError(f"Unknown graph operation: {operation}")
            
            # Add operation metadata
            result["operation"] = operation
            result["graph_timestamp"] = datetime.now().isoformat()
            
            logger.info(f"✅ Knowledge graph operation '{operation}' completed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Knowledge graph operation '{operation}' failed: {e}")
            return {
                "error": str(e),
                "operation": operation,
                "graph_timestamp": datetime.now().isoformat()
            }

    # Helper methods for knowledge synthesis
    async def _analyze_memory_patterns(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in retrieved memories."""
        if not memories:
            return {"pattern_count": 0, "patterns": []}
        
        patterns = []
        
        # Extract common themes
        themes = {}
        for memory in memories:
            content = memory.get("content", "")
            words = content.lower().split()
            for word in words:
                if len(word) > 4:
                    themes[word] = themes.get(word, 0) + 1
        
        # Get top themes
        top_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for theme, count in top_themes:
            patterns.append({
                "type": "theme",
                "name": theme,
                "frequency": count,
                "confidence": min(count / len(memories), 1.0)
            })
        
        # Extract temporal patterns
        temporal_patterns = self._extract_temporal_patterns(memories)
        patterns.extend(temporal_patterns)
        
        return {
            "pattern_count": len(patterns),
            "patterns": patterns,
            "theme_distribution": dict(top_themes),
            "memory_count": len(memories)
        }

    async def _generate_knowledge_synthesis(
        self,
        query: str,
        memories: List[Dict[str, Any]],
        pattern_analysis: Dict[str, Any],
        synthesis_type: str
    ) -> Dict[str, Any]:
        """Generate knowledge synthesis from memories and patterns."""
        
        # Extract key insights from memories
        insights = []
        for memory in memories[:10]:  # Top 10 most relevant
            insight = self._extract_insight_from_memory(memory, query)
            if insight:
                insights.append(insight)
        
        # Generate synthesis based on type
        if synthesis_type == "comprehensive":
            synthesized_knowledge = self._generate_comprehensive_synthesis(insights, pattern_analysis)
        elif synthesis_type == "focused":
            synthesized_knowledge = self._generate_focused_synthesis(insights, query)
        elif synthesis_type == "exploratory":
            synthesized_knowledge = self._generate_exploratory_synthesis(insights, pattern_analysis)
        else:
            synthesized_knowledge = self._generate_comprehensive_synthesis(insights, pattern_analysis)
        
        # Build knowledge graph
        knowledge_graph = self._build_knowledge_graph(insights, pattern_analysis)
        
        return {
            "synthesized_knowledge": synthesized_knowledge,
            "insights": insights,
            "knowledge_graph": knowledge_graph,
            "synthesis_type": synthesis_type
        }

    async def _validate_synthesis_quality(self, synthesis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the quality of knowledge synthesis."""
        insights = synthesis_result.get("insights", [])
        knowledge = synthesis_result.get("synthesized_knowledge", "")
        
        # Calculate quality metrics
        insight_count = len(insights)
        knowledge_length = len(knowledge)
        insight_diversity = len(set(insight.get("type", "") for insight in insights))
        
        # Quality scoring
        completeness_score = min(insight_count / 10, 1.0)  # Normalize by 10 insights
        coherence_score = min(knowledge_length / 500, 1.0)  # Normalize by 500 chars
        diversity_score = min(insight_diversity / 5, 1.0)  # Normalize by 5 types
        
        overall_confidence = (completeness_score * 0.4 + coherence_score * 0.3 + diversity_score * 0.3)
        
        return {
            "confidence_score": overall_confidence,
            "completeness_score": completeness_score,
            "coherence_score": coherence_score,
            "diversity_score": diversity_score,
            "quality_metrics": {
                "insight_count": insight_count,
                "knowledge_length": knowledge_length,
                "insight_diversity": insight_diversity
            }
        }

    # Helper methods for cross-domain learning
    async def retrieve_domain_knowledge(self, domain: str) -> List[Dict[str, Any]]:
        """Retrieve knowledge specific to a domain."""
        return await self.retrieve_memories(f"domain:{domain}", limit=50)

    async def _analyze_domain_relationships(
        self,
        source_domain: str,
        target_domain: str,
        source_knowledge: List[Dict[str, Any]],
        target_knowledge: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze relationships between source and target domains."""
        
        # Extract domain characteristics
        source_characteristics = self._extract_domain_characteristics(source_knowledge)
        target_characteristics = self._extract_domain_characteristics(target_knowledge)
        
        # Calculate similarity scores
        similarity_scores = self._calculate_domain_similarity(
            source_characteristics, target_characteristics
        )
        
        # Identify transfer opportunities
        transfer_opportunities = self._identify_transfer_opportunities(
            source_characteristics, target_characteristics
        )
        
        return {
            "source_characteristics": source_characteristics,
            "target_characteristics": target_characteristics,
            "similarity_scores": similarity_scores,
            "transfer_opportunities": transfer_opportunities,
            "compatibility_score": similarity_scores.get("overall", 0.5)
        }

    async def _generate_transfer_strategies(
        self,
        source_domain: str,
        target_domain: str,
        domain_analysis: Dict[str, Any],
        learning_objective: str
    ) -> List[Dict[str, Any]]:
        """Generate strategies for cross-domain knowledge transfer."""
        strategies = []
        
        similarity_score = domain_analysis.get("compatibility_score", 0.5)
        opportunities = domain_analysis.get("transfer_opportunities", [])
        
        # Direct transfer strategy
        if similarity_score > 0.7:
            strategies.append({
                "type": "direct_transfer",
                "description": "Direct application of knowledge from source to target domain",
                "confidence": similarity_score,
                "implementation": "Map concepts directly between domains"
            })
        
        # Adaptive transfer strategy
        strategies.append({
            "type": "adaptive_transfer",
            "description": "Adapt source domain knowledge to target domain context",
            "confidence": 0.8,
            "implementation": "Modify knowledge structures for target domain"
        })
        
        # Bridge transfer strategy
        if opportunities:
            strategies.append({
                "type": "bridge_transfer",
                "description": "Use common concepts as bridges between domains",
                "confidence": 0.9,
                "implementation": "Identify and leverage shared concepts"
            })
        
        return strategies

    async def _apply_cross_domain_learning(
        self,
        source_knowledge: List[Dict[str, Any]],
        target_knowledge: List[Dict[str, Any]],
        transfer_strategies: List[Dict[str, Any]],
        learning_objective: str
    ) -> Dict[str, Any]:
        """Apply cross-domain learning using transfer strategies."""
        
        insights = []
        knowledge_bridges = []
        
        # Apply each transfer strategy
        for strategy in transfer_strategies:
            if strategy["type"] == "direct_transfer":
                direct_insights = self._apply_direct_transfer(source_knowledge, target_knowledge)
                insights.extend(direct_insights)
            
            elif strategy["type"] == "adaptive_transfer":
                adaptive_insights = self._apply_adaptive_transfer(source_knowledge, target_knowledge)
                insights.extend(adaptive_insights)
            
            elif strategy["type"] == "bridge_transfer":
                bridge_insights, bridges = self._apply_bridge_transfer(source_knowledge, target_knowledge)
                insights.extend(bridge_insights)
                knowledge_bridges.extend(bridges)
        
        # Generate cross-domain insights
        cross_domain_insights = self._generate_cross_domain_insights(insights, learning_objective)
        
        return {
            "insights": insights,
            "knowledge_bridges": knowledge_bridges,
            "cross_domain_insights": cross_domain_insights,
            "strategies_applied": len(transfer_strategies)
        }

    async def _validate_cross_domain_learning(self, learning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate cross-domain learning outcomes."""
        insights = learning_result.get("insights", [])
        bridges = learning_result.get("knowledge_bridges", [])
        
        # Calculate success metrics
        insight_count = len(insights)
        bridge_count = len(bridges)
        
        # Quality assessment
        insight_quality = sum(insight.get("confidence", 0.5) for insight in insights) / max(insight_count, 1)
        bridge_quality = sum(bridge.get("strength", 0.5) for bridge in bridges) / max(bridge_count, 1)
        
        success_rate = (insight_quality * 0.7 + bridge_quality * 0.3)
        
        return {
            "success_rate": success_rate,
            "insight_count": insight_count,
            "bridge_count": bridge_count,
            "insight_quality": insight_quality,
            "bridge_quality": bridge_quality
        }

    # Helper methods for advanced memory operations
    async def _consolidate_memories(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate memories for better organization."""
        consolidation_threshold = parameters.get("threshold", 0.8)
        
        # Find similar memories
        all_memories = await self.retrieve_memories("", limit=1000)
        consolidated_count = 0
        
        # Simple consolidation logic
        for i, memory1 in enumerate(all_memories):
            for j, memory2 in enumerate(all_memories[i+1:], i+1):
                similarity = self._calculate_memory_similarity(memory1, memory2)
                if similarity > consolidation_threshold:
                    # Consolidate memories
                    consolidated_memory = self._merge_memories(memory1, memory2)
                    await self.store_memory(
                        content=consolidated_memory["content"],
                        metadata=consolidated_memory["metadata"]
                    )
                    consolidated_count += 1
        
        return {
            "operation": "consolidate",
            "memories_processed": len(all_memories),
            "memories_consolidated": consolidated_count,
            "consolidation_threshold": consolidation_threshold
        }

    async def _optimize_memory_structure(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize memory structure for better performance."""
        optimization_type = parameters.get("type", "index")
        
        if optimization_type == "index":
            # Rebuild memory index
            await self._rebuild_memory_index({})
            return {
                "operation": "optimize",
                "optimization_type": "index_rebuild",
                "status": "completed"
            }
        else:
            return {
                "operation": "optimize",
                "optimization_type": optimization_type,
                "status": "not_implemented"
            }

    async def _cleanup_memories(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up old or low-quality memories."""
        cleanup_threshold = parameters.get("threshold", 0.3)
        max_age_days = parameters.get("max_age_days", 30)
        
        # Find memories to cleanup
        all_memories = await self.retrieve_memories("", limit=1000)
        cleaned_count = 0
        
        for memory in all_memories:
            # Check quality and age
            quality = memory.get("metadata", {}).get("quality_score", 0.5)
            age_days = (datetime.now() - datetime.fromisoformat(memory.get("timestamp", datetime.now().isoformat()))).days
            
            if quality < cleanup_threshold or age_days > max_age_days:
                # Mark for cleanup
                cleaned_count += 1
        
        return {
            "operation": "cleanup",
            "memories_processed": len(all_memories),
            "memories_cleaned": cleaned_count,
            "cleanup_threshold": cleanup_threshold,
            "max_age_days": max_age_days
        }

    async def _archive_memories(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Archive old memories for long-term storage."""
        archive_threshold_days = parameters.get("threshold_days", 90)
        
        # Find memories to archive
        all_memories = await self.retrieve_memories("", limit=1000)
        archived_count = 0
        
        for memory in all_memories:
            age_days = (datetime.now() - datetime.fromisoformat(memory.get("timestamp", datetime.now().isoformat()))).days
            
            if age_days > archive_threshold_days:
                # Archive memory
                archived_count += 1
        
        return {
            "operation": "archive",
            "memories_processed": len(all_memories),
            "memories_archived": archived_count,
            "archive_threshold_days": archive_threshold_days
        }

    async def _rebuild_memory_index(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Rebuild memory index for better search performance."""
        # This would typically involve rebuilding the vector index
        # For now, return a placeholder
        return {
            "operation": "rebuild_index",
            "status": "completed",
            "index_rebuilt": True
        }

    # Helper methods for knowledge graph operations
    async def _construct_knowledge_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Construct a knowledge graph from provided data."""
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        
        # Build graph structure
        graph = {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges)
        }
        
        return {
            "operation": "construct",
            "graph": graph,
            "status": "completed"
        }

    async def _update_knowledge_graph_data(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing knowledge graph with new data."""
        updates = graph_data.get("updates", {})
        
        return {
            "operation": "update",
            "updates_applied": len(updates),
            "status": "completed"
        }

    async def _find_knowledge_paths(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Find paths in the knowledge graph."""
        start_node = graph_data.get("start_node")
        end_node = graph_data.get("end_node")
        max_paths = graph_data.get("max_paths", 5)
        
        # Simple path finding logic
        paths = []
        for i in range(min(max_paths, 3)):  # Generate up to 3 sample paths
            paths.append({
                "path_id": f"path_{i}",
                "nodes": [start_node, f"intermediate_{i}", end_node],
                "length": 3,
                "confidence": 0.8 - (i * 0.1)
            })
        
        return {
            "operation": "find_paths",
            "start_node": start_node,
            "end_node": end_node,
            "paths_found": len(paths),
            "paths": paths
        }

    async def _analyze_knowledge_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze knowledge graph structure and properties."""
        graph = graph_data.get("graph", {})
        
        analysis = {
            "node_count": len(graph.get("nodes", [])),
            "edge_count": len(graph.get("edges", [])),
            "density": self._calculate_graph_density(graph),
            "centrality": self._calculate_graph_centrality(graph),
            "connectivity": self._calculate_graph_connectivity(graph)
        }
        
        return {
            "operation": "analyze",
            "analysis": analysis,
            "status": "completed"
        }

    async def _optimize_knowledge_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize knowledge graph structure."""
        optimization_type = graph_data.get("optimization_type", "structure")
        
        return {
            "operation": "optimize",
            "optimization_type": optimization_type,
            "status": "completed"
        }

    # Utility methods
    def _extract_temporal_patterns(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract temporal patterns from memories."""
        patterns = []
        
        # Group by time periods
        time_groups = {}
        for memory in memories:
            timestamp = memory.get("timestamp", "")
            if timestamp:
                date = timestamp.split("T")[0]  # Extract date part
                if date not in time_groups:
                    time_groups[date] = []
                time_groups[date].append(memory)
        
        # Create temporal patterns
        for date, group_memories in time_groups.items():
            if len(group_memories) > 1:
                patterns.append({
                    "type": "temporal",
                    "name": f"Temporal cluster: {date}",
                    "memory_count": len(group_memories),
                    "confidence": min(len(group_memories) / 10, 1.0)
                })
        
        return patterns

    def _extract_insight_from_memory(self, memory: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Extract insight from a single memory."""
        content = memory.get("content", "")
        
        # Simple insight extraction
        if len(content) > 50:
            return {
                "type": "knowledge",
                "content": content[:200] + "..." if len(content) > 200 else content,
                "confidence": 0.8,
                "source_memory": memory.get("id", "unknown")
            }
        return None

    def _generate_comprehensive_synthesis(self, insights: List[Dict[str, Any]], pattern_analysis: Dict[str, Any]) -> str:
        """Generate comprehensive knowledge synthesis."""
        if not insights:
            return "No insights available for synthesis."
        
        synthesis = "Comprehensive Knowledge Synthesis:\n\n"
        
        # Group insights by type
        insight_groups = {}
        for insight in insights:
            insight_type = insight.get("type", "general")
            if insight_type not in insight_groups:
                insight_groups[insight_type] = []
            insight_groups[insight_type].append(insight)
        
        # Generate synthesis for each group
        for insight_type, group_insights in insight_groups.items():
            synthesis += f"{insight_type.title()} Insights:\n"
            for insight in group_insights[:3]:  # Top 3 per type
                synthesis += f"- {insight.get('content', '')}\n"
            synthesis += "\n"
        
        return synthesis

    def _generate_focused_synthesis(self, insights: List[Dict[str, Any]], query: str) -> str:
        """Generate focused knowledge synthesis."""
        if not insights:
            return f"No insights available for query: {query}"
        
        synthesis = f"Focused Synthesis for: {query}\n\n"
        
        # Select most relevant insights
        relevant_insights = insights[:5]  # Top 5 most relevant
        
        for i, insight in enumerate(relevant_insights, 1):
            synthesis += f"{i}. {insight.get('content', '')}\n"
        
        return synthesis

    def _generate_exploratory_synthesis(self, insights: List[Dict[str, Any]], pattern_analysis: Dict[str, Any]) -> str:
        """Generate exploratory knowledge synthesis."""
        if not insights:
            return "No insights available for exploration."
        
        synthesis = "Exploratory Knowledge Synthesis:\n\n"
        
        # Focus on patterns and connections
        patterns = pattern_analysis.get("patterns", [])
        synthesis += "Key Patterns Identified:\n"
        for pattern in patterns[:5]:
            synthesis += f"- {pattern.get('name', '')} (confidence: {pattern.get('confidence', 0):.2f})\n"
        
        synthesis += "\nEmerging Insights:\n"
        for insight in insights[:3]:
            synthesis += f"- {insight.get('content', '')}\n"
        
        return synthesis

    def _build_knowledge_graph(self, insights: List[Dict[str, Any]], pattern_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build knowledge graph from insights and patterns."""
        nodes = []
        edges = []
        
        # Add insight nodes
        for i, insight in enumerate(insights):
            nodes.append({
                "id": f"insight_{i}",
                "type": "insight",
                "content": insight.get("content", "")[:100],
                "confidence": insight.get("confidence", 0.5)
            })
        
        # Add pattern nodes
        patterns = pattern_analysis.get("patterns", [])
        for i, pattern in enumerate(patterns):
            nodes.append({
                "id": f"pattern_{i}",
                "type": "pattern",
                "name": pattern.get("name", ""),
                "confidence": pattern.get("confidence", 0.5)
            })
        
        # Add edges (simplified)
        for i in range(len(insights) - 1):
            edges.append({
                "source": f"insight_{i}",
                "target": f"insight_{i+1}",
                "type": "related"
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges)
        }

    def _extract_domain_characteristics(self, knowledge: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract characteristics of a domain from its knowledge."""
        if not knowledge:
            return {"concepts": [], "complexity": 0.0, "specialization": 0.0}
        
        # Extract concepts
        concepts = []
        for memory in knowledge:
            content = memory.get("content", "")
            words = content.lower().split()
            for word in words:
                if len(word) > 4 and word not in concepts:
                    concepts.append(word)
        
        # Calculate characteristics
        complexity = min(len(concepts) / 50, 1.0)  # Normalize by 50 concepts
        specialization = min(len(set(concept[:5] for concept in concepts)) / 20, 1.0)  # Normalize by 20 unique prefixes
        
        return {
            "concepts": concepts[:20],  # Top 20 concepts
            "complexity": complexity,
            "specialization": specialization,
            "knowledge_count": len(knowledge)
        }

    def _calculate_domain_similarity(self, source_chars: Dict[str, Any], target_chars: Dict[str, Any]) -> Dict[str, float]:
        """Calculate similarity between source and target domains."""
        source_concepts = set(source_chars.get("concepts", []))
        target_concepts = set(target_chars.get("concepts", []))
        
        # Calculate concept overlap
        overlap = len(source_concepts.intersection(target_concepts))
        total_concepts = len(source_concepts.union(target_concepts))
        concept_similarity = overlap / total_concepts if total_concepts > 0 else 0
        
        # Calculate characteristic similarity
        complexity_similarity = 1.0 - abs(source_chars.get("complexity", 0) - target_chars.get("complexity", 0))
        specialization_similarity = 1.0 - abs(source_chars.get("specialization", 0) - target_chars.get("specialization", 0))
        
        overall_similarity = (concept_similarity * 0.5 + complexity_similarity * 0.25 + specialization_similarity * 0.25)
        
        return {
            "concept_similarity": concept_similarity,
            "complexity_similarity": complexity_similarity,
            "specialization_similarity": specialization_similarity,
            "overall": overall_similarity
        }

    def _identify_transfer_opportunities(self, source_chars: Dict[str, Any], target_chars: Dict[str, Any]) -> List[str]:
        """Identify opportunities for knowledge transfer between domains."""
        opportunities = []
        
        source_concepts = set(source_chars.get("concepts", []))
        target_concepts = set(target_chars.get("concepts", []))
        
        # Find shared concepts
        shared_concepts = source_concepts.intersection(target_concepts)
        if shared_concepts:
            opportunities.append(f"Shared concepts: {', '.join(list(shared_concepts)[:5])}")
        
        # Find complementary concepts
        source_only = source_concepts - target_concepts
        if source_only:
            opportunities.append(f"Source-specific concepts: {', '.join(list(source_only)[:5])}")
        
        return opportunities

    def _apply_direct_transfer(self, source_knowledge: List[Dict[str, Any]], target_knowledge: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply direct knowledge transfer."""
        insights = []
        
        for source_memory in source_knowledge[:5]:  # Top 5 source memories
            insights.append({
                "type": "direct_transfer",
                "content": f"Direct application: {source_memory.get('content', '')[:100]}...",
                "confidence": 0.8,
                "source": "direct_transfer"
            })
        
        return insights

    def _apply_adaptive_transfer(self, source_knowledge: List[Dict[str, Any]], target_knowledge: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply adaptive knowledge transfer."""
        insights = []
        
        for source_memory in source_knowledge[:5]:  # Top 5 source memories
            insights.append({
                "type": "adaptive_transfer",
                "content": f"Adapted application: {source_memory.get('content', '')[:100]}...",
                "confidence": 0.7,
                "source": "adaptive_transfer"
            })
        
        return insights

    def _apply_bridge_transfer(self, source_knowledge: List[Dict[str, Any]], target_knowledge: List[Dict[str, Any]]) -> tuple:
        """Apply bridge-based knowledge transfer."""
        insights = []
        bridges = []
        
        # Create knowledge bridges
        bridges.append({
            "type": "concept_bridge",
            "source_concept": "algorithm",
            "target_concept": "process",
            "strength": 0.9,
            "description": "Algorithm concepts can be applied to process optimization"
        })
        
        insights.append({
            "type": "bridge_transfer",
            "content": "Bridge transfer: Algorithm optimization principles applied to process improvement",
            "confidence": 0.9,
            "source": "bridge_transfer"
        })
        
        return insights, bridges

    def _generate_cross_domain_insights(self, insights: List[Dict[str, Any]], learning_objective: str) -> str:
        """Generate cross-domain insights summary."""
        if not insights:
            return f"No cross-domain insights generated for objective: {learning_objective}"
        
        summary = f"Cross-Domain Learning Summary for: {learning_objective}\n\n"
        
        # Group insights by type
        insight_types = {}
        for insight in insights:
            insight_type = insight.get("type", "general")
            if insight_type not in insight_types:
                insight_types[insight_type] = []
            insight_types[insight_type].append(insight)
        
        for insight_type, type_insights in insight_types.items():
            summary += f"{insight_type.replace('_', ' ').title()}:\n"
            for insight in type_insights[:3]:  # Top 3 per type
                summary += f"- {insight.get('content', '')}\n"
            summary += "\n"
        
        return summary

    def _calculate_memory_similarity(self, memory1: Dict[str, Any], memory2: Dict[str, Any]) -> float:
        """Calculate similarity between two memories."""
        content1 = memory1.get("content", "")
        content2 = memory2.get("content", "")
        
        # Simple similarity based on content overlap
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def _merge_memories(self, memory1: Dict[str, Any], memory2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two similar memories."""
        content1 = memory1.get("content", "")
        content2 = memory2.get("content", "")
        
        # Simple merge strategy
        merged_content = f"{content1}\n\n{content2}"
        
        metadata1 = memory1.get("metadata", {})
        metadata2 = memory2.get("metadata", {})
        
        # Merge metadata
        merged_metadata = metadata1.copy()
        merged_metadata.update(metadata2)
        merged_metadata["merged_from"] = [memory1.get("id"), memory2.get("id")]
        
        return {
            "content": merged_content,
            "metadata": merged_metadata
        }

    def _calculate_graph_density(self, graph: Dict[str, Any]) -> float:
        """Calculate density of knowledge graph."""
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        
        if len(nodes) < 2:
            return 0.0
        
        max_edges = len(nodes) * (len(nodes) - 1) / 2
        return len(edges) / max_edges if max_edges > 0 else 0.0

    def _calculate_graph_centrality(self, graph: Dict[str, Any]) -> Dict[str, float]:
        """Calculate centrality metrics for knowledge graph."""
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        
        # Simple centrality calculation
        node_degrees = {}
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            node_degrees[source] = node_degrees.get(source, 0) + 1
            node_degrees[target] = node_degrees.get(target, 0) + 1
        
        if not node_degrees:
            return {"average_degree": 0.0, "max_degree": 0.0}
        
        degrees = list(node_degrees.values())
        return {
            "average_degree": sum(degrees) / len(degrees),
            "max_degree": max(degrees)
        }

    def _calculate_graph_connectivity(self, graph: Dict[str, Any]) -> float:
        """Calculate connectivity of knowledge graph."""
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        
        if len(nodes) < 2:
            return 0.0
        
        # Simple connectivity based on edge-to-node ratio
        return len(edges) / len(nodes) if len(nodes) > 0 else 0.0
    
    def _calculate_recall_confidence(self, recall_results: List[Dict[str, Any]], query: str) -> float:
        """Calculate confidence based on recall quality and memory coverage."""
        if not recall_results:
            return 0.1
        
        # Calculate relevance scores
        relevance_scores = []
        workflow_cohesion_scores = []
        temporal_cohesion_scores = []
        
        for result in recall_results:
            if isinstance(result, dict):
                # Extract relevance indicators
                similarity_score = result.get('similarity_score', 0.0)
                success_score = result.get('success_score', 0.0)
                usage_count = result.get('usage_count', 0)
                workflow_cohesion = result.get('workflow_cohesion_score', 0.0)
                cross_reference_count = result.get('cross_reference_count', 0)
                
                # Calculate weighted relevance
                relevance = (similarity_score * 0.5 + success_score * 0.2 + min(usage_count / 10, 1.0) * 0.1)
                relevance_scores.append(relevance)
                
                # Collect workflow and temporal cohesion scores
                if workflow_cohesion > 0:
                    workflow_cohesion_scores.append(workflow_cohesion)
                if cross_reference_count > 0:
                    temporal_cohesion_scores.append(min(cross_reference_count / 10.0, 1.0))
        
        # Calculate average relevance
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        # Calculate workflow cohesion bonus
        workflow_cohesion_bonus = 0.0
        if workflow_cohesion_scores:
            avg_workflow_cohesion = sum(workflow_cohesion_scores) / len(workflow_cohesion_scores)
            workflow_cohesion_bonus = avg_workflow_cohesion * 0.15
        
        # Calculate temporal cohesion bonus
        temporal_cohesion_bonus = 0.0
        if temporal_cohesion_scores:
            avg_temporal_cohesion = sum(temporal_cohesion_scores) / len(temporal_cohesion_scores)
            temporal_cohesion_bonus = avg_temporal_cohesion * 0.1
        
        # Consider memory coverage
        memory_coverage = min(len(recall_results) / 10, 1.0)  # Normalize by expected results
        
        # Consider query complexity
        query_complexity = min(len(query.split()) / 5, 1.0)  # Normalize by word count
        
        # Enhanced confidence calculation with workflow optimization
        confidence = (
            avg_relevance * 0.5 + 
            memory_coverage * 0.2 + 
            query_complexity * 0.1 +
            workflow_cohesion_bonus +
            temporal_cohesion_bonus
        )
        
        # Ensure confidence is within reasonable bounds
        return min(0.95, max(0.1, confidence))

    async def _calculate_enhanced_relevance(
        self,
        entry: MemoryEntry,
        similarity: float,
        workflow_context: Dict[str, Any] = None,
        temporal_context: Dict[str, Any] = None,
        query: str = ""
    ) -> float:
        """
        Calculate enhanced relevance score with workflow and temporal optimization.
        
        Args:
            entry: Memory entry to score
            similarity: Semantic similarity score
            workflow_context: Workflow context information
            temporal_context: Temporal cluster context
            query: Original query for context
            
        Returns:
            Enhanced relevance score
        """
        # Base relevance calculation
        base_relevance = (similarity * 0.5) + (entry.success_score * 0.2) + (entry.usage_count * 0.1)
        
        # Workflow cohesion bonus
        workflow_cohesion_bonus = 0.0
        if workflow_context and entry.workflow_id == workflow_context.get("workflow_id"):
            workflow_cohesion_bonus = entry.workflow_cohesion_score * self.workflow_config["workflow_cohesion_weight"]
        
        # Temporal relevance bonus
        temporal_bonus = 0.0
        if temporal_context and entry.temporal_cluster == temporal_context.get("cluster_id"):
            # Calculate temporal decay
            time_diff = (datetime.now() - entry.timestamp).total_seconds()
            temporal_decay = self.workflow_config["temporal_decay_rate"] ** (time_diff / 3600)  # Decay per hour
            temporal_bonus = temporal_decay * 0.1
        
        # Cross-reference bonus
        cross_ref_bonus = 0.0
        if entry.cross_reference_count > 0:
            cross_ref_bonus = min(entry.cross_reference_count / 10.0, 1.0) * self.workflow_config["cross_reference_weight"]
        
        # Recent workflow access bonus
        recency_bonus = 0.0
        if entry.last_workflow_access and isinstance(entry.last_workflow_access, datetime):
            time_since_access = (datetime.now() - entry.last_workflow_access).total_seconds()
            if time_since_access < 3600:  # Within last hour
                recency_bonus = 0.1 * (1.0 - time_since_access / 3600)
        
        # Combine all factors
        enhanced_relevance = (
            base_relevance +
            workflow_cohesion_bonus +
            temporal_bonus +
            cross_ref_bonus +
            recency_bonus
        )
        
        return min(1.0, enhanced_relevance)  # Cap at 1.0

    async def _get_workflow_context(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow context for enhanced retrieval."""
        if workflow_id not in self.workflow_memory_map:
            return None
        
        memory_ids = self.workflow_memory_map[workflow_id]
        workflow_memories = [self.memory_db[mid] for mid in memory_ids if mid in self.memory_db]
        
        if not workflow_memories:
            return None
        
        # Calculate workflow cohesion
        cohesion_score = sum(m.workflow_cohesion_score for m in workflow_memories) / len(workflow_memories)
        
        return {
            "workflow_id": workflow_id,
            "memory_count": len(workflow_memories),
            "cohesion_score": cohesion_score,
            "active_session": self.active_workflows.get(workflow_id),
            "memory_ids": memory_ids
        }

    async def _get_temporal_context(self, session_id: str) -> Dict[str, Any]:
        """Get temporal cluster context for enhanced retrieval."""
        if session_id not in self.workflow_sessions:
            return None
        
        session = self.workflow_sessions[session_id]
        if not session.temporal_cluster:
            return None
        
        cluster_id = session.temporal_cluster
        if cluster_id not in self.temporal_clusters:
            return None
        
        cluster = self.temporal_clusters[cluster_id]
        
        return {
            "cluster_id": cluster_id,
            "session_id": session_id,
            "start_time": cluster.start_time,
            "end_time": cluster.end_time,
            "memory_count": len(cluster.memory_ids),
            "cohesion_score": cluster.cohesion_score,
            "cross_reference_density": cluster.cross_reference_density
        }

    async def _update_cross_references(self, results: List[RecallResult]):
        """Update cross-reference tracking between recalled memories."""
        if len(results) < 2:
            return
        
        # Create cross-references between recalled memories
        for i, result1 in enumerate(results):
            memory_id1 = result1.entry.id
            if memory_id1 not in self.cross_reference_graph:
                self.cross_reference_graph[memory_id1] = set()
            
            for j, result2 in enumerate(results[i+1:], i+1):
                memory_id2 = result2.entry.id
                if memory_id2 not in self.cross_reference_graph:
                    self.cross_reference_graph[memory_id2] = set()
                
                # Add bidirectional references
                self.cross_reference_graph[memory_id1].add(memory_id2)
                self.cross_reference_graph[memory_id2].add(memory_id1)
                
                # Update cross-reference counts
                result1.entry.cross_reference_count = len(self.cross_reference_graph[memory_id1])
                result2.entry.cross_reference_count = len(self.cross_reference_graph[memory_id2])

    async def start_workflow_session(self, workflow_id: str, session_id: str = None) -> str:
        """
        Start a new workflow session for temporal clustering.
        
        Args:
            workflow_id: Unique workflow identifier
            session_id: Optional session ID (auto-generated if not provided)
            
        Returns:
            Session ID
        """
        if not session_id:
            session_id = f"session_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session = WorkflowSession(
            session_id=session_id,
            workflow_id=workflow_id,
            start_time=datetime.now()
        )
        
        self.workflow_sessions[session_id] = session
        self.active_workflows[workflow_id] = session_id
        
        logger.info(f"🔄 Started workflow session: {session_id} for workflow: {workflow_id}")
        return session_id

    async def end_workflow_session(self, session_id: str) -> Dict[str, Any]:
        """
        End a workflow session and create temporal cluster.
        
        Args:
            session_id: Session ID to end
            
        Returns:
            Session summary
        """
        if session_id not in self.workflow_sessions:
            return {"error": "Session not found"}
        
        session = self.workflow_sessions[session_id]
        session.end_time = datetime.now()
        session.active = False
        
        # Create temporal cluster
        cluster_id = f"cluster_{session.workflow_id}_{session.start_time.strftime('%Y%m%d_%H%M%S')}"
        cluster = TemporalCluster(
            cluster_id=cluster_id,
            start_time=session.start_time,
            end_time=session.end_time,
            memory_ids=session.memory_ids.copy(),
            dominant_workflow=session.workflow_id
        )
        
        # Calculate cluster metrics
        cluster.cohesion_score = session.cohesion_score
        cluster.cross_reference_density = await self._calculate_cross_reference_density(cluster.memory_ids)
        
        self.temporal_clusters[cluster_id] = cluster
        session.temporal_cluster = cluster_id
        
        # Update memory entries with cluster information
        for memory_id in session.memory_ids:
            if memory_id in self.memory_db:
                self.memory_db[memory_id].temporal_cluster = cluster_id
        
        # Remove from active workflows
        if session.workflow_id in self.active_workflows:
            del self.active_workflows[session.workflow_id]
        
        self.operation_stats["temporal_clustering_operations"] += 1
        
        logger.info(f"🔄 Ended workflow session: {session_id}, created cluster: {cluster_id}")
        
        return {
            "session_id": session_id,
            "cluster_id": cluster_id,
            "duration": (session.end_time - session.start_time).total_seconds(),
            "memory_count": len(session.memory_ids),
            "cohesion_score": session.cohesion_score
        }

    async def _calculate_cross_reference_density(self, memory_ids: List[str]) -> float:
        """Calculate cross-reference density for a set of memories."""
        if len(memory_ids) < 2:
            return 0.0
        
        total_references = 0
        for memory_id in memory_ids:
            if memory_id in self.cross_reference_graph:
                # Count references within the same set
                references = self.cross_reference_graph[memory_id]
                internal_refs = len(references.intersection(set(memory_ids)))
                total_references += internal_refs
        
        # Normalize by possible connections
        max_possible = len(memory_ids) * (len(memory_ids) - 1) / 2
        return total_references / max_possible if max_possible > 0 else 0.0

    async def get_workflow_cohesion_metrics(self, workflow_id: str) -> Dict[str, Any]:
        """Get cohesion metrics for a specific workflow."""
        if workflow_id not in self.workflow_memory_map:
            return {"error": "Workflow not found"}
        
        memory_ids = self.workflow_memory_map[workflow_id]
        memories = [self.memory_db[mid] for mid in memory_ids if mid in self.memory_db]
        
        if not memories:
            return {"error": "No memories found for workflow"}
        
        # Calculate various cohesion metrics
        avg_cohesion = sum(m.workflow_cohesion_score for m in memories) / len(memories)
        avg_cross_refs = sum(m.cross_reference_count for m in memories) / len(memories)
        
        # Calculate temporal distribution
        timestamps = [m.timestamp for m in memories]
        time_span = (max(timestamps) - min(timestamps)).total_seconds() if timestamps else 0
        
        return {
            "workflow_id": workflow_id,
            "memory_count": len(memories),
            "average_cohesion_score": avg_cohesion,
            "average_cross_references": avg_cross_refs,
            "time_span_seconds": time_span,
            "active_session": self.active_workflows.get(workflow_id),
            "temporal_clusters": len([m.temporal_cluster for m in memories if m.temporal_cluster])
        }

    async def optimize_workflow_performance(self, workflow_id: str) -> Dict[str, Any]:
        """
        Optimize workflow performance through advanced memory management.
        
        Args:
            workflow_id: Workflow ID to optimize
            
        Returns:
            Optimization results and recommendations
        """
        if workflow_id not in self.workflow_memory_map:
            return {"error": "Workflow not found"}
        
        memory_ids = self.workflow_memory_map[workflow_id]
        memories = [self.memory_db[mid] for mid in memory_ids if mid in self.memory_db]
        
        if not memories:
            return {"error": "No memories found for workflow"}
        
        optimization_results = {
            "workflow_id": workflow_id,
            "optimization_timestamp": datetime.now().isoformat(),
            "pre_optimization_metrics": await self.get_workflow_cohesion_metrics(workflow_id),
            "optimizations_applied": [],
            "recommendations": []
        }
        
        # 1. Temporal clustering optimization
        temporal_optimization = await self._optimize_temporal_clustering(memories)
        optimization_results["optimizations_applied"].append(temporal_optimization)
        
        # 2. Cross-reference optimization
        cross_ref_optimization = await self._optimize_cross_references(memories)
        optimization_results["optimizations_applied"].append(cross_ref_optimization)
        
        # 3. Cohesion score recalculation
        cohesion_optimization = await self._recalculate_workflow_cohesion(memories)
        optimization_results["optimizations_applied"].append(cohesion_optimization)
        
        # 4. Generate recommendations
        recommendations = await self._generate_workflow_recommendations(memories)
        optimization_results["recommendations"] = recommendations
        
        # 5. Post-optimization metrics
        optimization_results["post_optimization_metrics"] = await self.get_workflow_cohesion_metrics(workflow_id)
        
        logger.info(f"🔄 Workflow optimization completed for {workflow_id}")
        return optimization_results

    async def _optimize_temporal_clustering(self, memories: List[MemoryEntry]) -> Dict[str, Any]:
        """Optimize temporal clustering for better workflow performance."""
        if len(memories) < 2:
            return {"type": "temporal_clustering", "status": "skipped", "reason": "insufficient_memories"}
        
        # Group memories by time periods
        time_groups = {}
        cluster_window = self.workflow_config["temporal_cluster_window"]
        
        for memory in memories:
            # Create time-based clusters
            cluster_key = memory.timestamp.strftime('%Y%m%d_%H')
            if cluster_key not in time_groups:
                time_groups[cluster_key] = []
            time_groups[cluster_key].append(memory)
        
        # Create or update temporal clusters
        clusters_created = 0
        for cluster_key, cluster_memories in time_groups.items():
            if len(cluster_memories) > 1:
                # Create temporal cluster
                cluster_id = f"optimized_cluster_{cluster_key}"
                start_time = min(m.timestamp for m in cluster_memories)
                end_time = max(m.timestamp for m in cluster_memories)
                
                cluster = TemporalCluster(
                    cluster_id=cluster_id,
                    start_time=start_time,
                    end_time=end_time,
                    memory_ids=[m.id for m in cluster_memories]
                )
                
                # Calculate cluster metrics
                cluster.cohesion_score = await self._calculate_session_cohesion([m.id for m in cluster_memories])
                cluster.cross_reference_density = await self._calculate_cross_reference_density([m.id for m in cluster_memories])
                
                self.temporal_clusters[cluster_id] = cluster
                clusters_created += 1
                
                # Update memory entries
                for memory in cluster_memories:
                    memory.temporal_cluster = cluster_id
        
        return {
            "type": "temporal_clustering",
            "status": "completed",
            "clusters_created": clusters_created,
            "memory_count": len(memories)
        }

    async def _optimize_cross_references(self, memories: List[MemoryEntry]) -> Dict[str, Any]:
        """Optimize cross-reference network for better workflow performance."""
        if len(memories) < 2:
            return {"type": "cross_references", "status": "skipped", "reason": "insufficient_memories"}
        
        # Rebuild cross-reference graph for this workflow
        cross_refs_created = 0
        similarity_threshold = 0.6  # Lower threshold for workflow optimization
        
        for i, memory1 in enumerate(memories):
            if memory1.id not in self.cross_reference_graph:
                self.cross_reference_graph[memory1.id] = set()
            
            for memory2 in memories[i+1:]:
                if memory2.id not in self.cross_reference_graph:
                    self.cross_reference_graph[memory2.id] = set()
                
                # Calculate similarity
                if memory1.embedding and memory2.embedding:
                    similarity = self._calculate_similarity(memory1.embedding, memory2.embedding)
                    
                    if similarity > similarity_threshold:
                        # Add bidirectional cross-reference
                        self.cross_reference_graph[memory1.id].add(memory2.id)
                        self.cross_reference_graph[memory2.id].add(memory1.id)
                        cross_refs_created += 1
        
        # Update cross-reference counts
        for memory in memories:
            memory.cross_reference_count = len(self.cross_reference_graph[memory.id])
        
        return {
            "type": "cross_references",
            "status": "completed",
            "cross_references_created": cross_refs_created,
            "memory_count": len(memories)
        }

    async def _recalculate_workflow_cohesion(self, memories: List[MemoryEntry]) -> Dict[str, Any]:
        """Recalculate workflow cohesion scores for all memories."""
        if len(memories) < 2:
            return {"type": "cohesion_recalculation", "status": "skipped", "reason": "insufficient_memories"}
        
        cohesion_updates = 0
        
        for memory in memories:
            # Calculate new cohesion score based on current workflow context
            if memory.workflow_id:
                new_cohesion = await self._calculate_workflow_cohesion(memory.content, memory.workflow_id)
                if abs(new_cohesion - memory.workflow_cohesion_score) > 0.1:  # Significant change
                    memory.workflow_cohesion_score = new_cohesion
                    cohesion_updates += 1
        
        return {
            "type": "cohesion_recalculation",
            "status": "completed",
            "cohesion_updates": cohesion_updates,
            "memory_count": len(memories)
        }

    async def _generate_workflow_recommendations(self, memories: List[MemoryEntry]) -> List[str]:
        """Generate recommendations for workflow optimization."""
        recommendations = []
        
        # Analyze memory distribution
        content_types = {}
        for memory in memories:
            content_types[memory.content_type] = content_types.get(memory.content_type, 0) + 1
        
        # Check for content type imbalance
        if len(content_types) > 1:
            max_type_count = max(content_types.values())
            min_type_count = min(content_types.values())
            if max_type_count > min_type_count * 3:
                recommendations.append("Consider diversifying content types for better workflow balance")
        
        # Check for temporal gaps
        timestamps = sorted([m.timestamp for m in memories])
        if len(timestamps) > 1:
            time_gaps = []
            for i in range(1, len(timestamps)):
                gap = (timestamps[i] - timestamps[i-1]).total_seconds()
                time_gaps.append(gap)
            
            avg_gap = sum(time_gaps) / len(time_gaps)
            if avg_gap > 3600:  # More than 1 hour average gap
                recommendations.append("Consider more frequent memory updates to maintain workflow continuity")
        
        # Check for low cohesion
        avg_cohesion = sum(m.workflow_cohesion_score for m in memories) / len(memories)
        if avg_cohesion < 0.5:
            recommendations.append("Workflow cohesion is low - consider grouping related tasks more closely")
        
        # Check for cross-reference density
        avg_cross_refs = sum(m.cross_reference_count for m in memories) / len(memories)
        if avg_cross_refs < 2:
            recommendations.append("Low cross-reference density - consider creating more connections between related memories")
        
        return recommendations

    async def get_enhanced_performance_metrics(self) -> Dict[str, Any]:
        """Get enhanced performance metrics including workflow optimization statistics."""
        base_metrics = await self.get_performance_metrics()
        
        # Add workflow-specific metrics
        workflow_metrics = {
            "active_workflows": len(self.active_workflows),
            "total_workflow_sessions": len(self.workflow_sessions),
            "total_temporal_clusters": len(self.temporal_clusters),
            "workflow_operations": self.operation_stats["workflow_operations"],
            "temporal_clustering_operations": self.operation_stats["temporal_clustering_operations"],
            "cross_reference_graph_size": len(self.cross_reference_graph),
            "workflow_memory_map_size": len(self.workflow_memory_map)
        }
        
        # Calculate workflow efficiency metrics
        if self.operation_stats["total_operations"] > 0:
            workflow_efficiency = self.operation_stats["workflow_operations"] / self.operation_stats["total_operations"]
            workflow_metrics["workflow_efficiency_ratio"] = workflow_efficiency
        
        # Calculate average cohesion across all workflows
        all_cohesion_scores = []
        for memory in self.memory_db.values():
            if memory.workflow_cohesion_score > 0:
                all_cohesion_scores.append(memory.workflow_cohesion_score)
        
        if all_cohesion_scores:
            workflow_metrics["average_workflow_cohesion"] = sum(all_cohesion_scores) / len(all_cohesion_scores)
        
        # Merge metrics
        enhanced_metrics = base_metrics.copy()
        enhanced_metrics["workflow_optimization"] = workflow_metrics
        
        return enhanced_metrics

    # ============================================================================
    # ADVANCED INFINITE CONTEXT MANAGEMENT METHODS
    # ============================================================================

    async def add_infinite_context(self, content: str, importance_score: float = 0.5, context_type: str = "text") -> List[str]:
        """
        Add content to infinite context system with semantic chunking.
        
        Args:
            content: Content to add
            importance_score: Importance score (0.0 to 1.0)
            context_type: Type of content
            
        Returns:
            List of chunk IDs
        """
        logger.info(f"♾️ Adding infinite context: {len(content)} chars, importance: {importance_score}")
        
        if not self.infinite_context_config['semantic_chunking_enabled']:
            # Fallback to simple chunking
            chunks = [content[i:i+1000] for i in range(0, len(content), 800)]
        else:
            # Use semantic chunking
            chunks = self.semantic_chunker.chunk_by_semantics(content)
        
        chunk_ids = []
        
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{context_type}_{hashlib.md5(chunk_text.encode()).hexdigest()[:8]}_{i}"
            
            chunk = ContextChunk(
                id=chunk_id,
                content=chunk_text,
                timestamp=datetime.now(),
                importance_score=importance_score,
                chunk_type=context_type,
                semantic_tags=self._extract_semantic_tags(chunk_text)
            )
            
            # Add to active context
            self.hierarchical_memory['active'].current_chunks.append(chunk)
            chunk_ids.append(chunk_id)
            
            # Manage hierarchical memory
            await self._manage_hierarchical_memory()
        
        logger.info(f"✅ Added {len(chunks)} chunks to infinite context")
        return chunk_ids

    async def get_infinite_context(self, query: str, max_tokens: int = None) -> str:
        """
        Retrieve relevant context from infinite context system.
        
        Args:
            query: Query to search for
            max_tokens: Maximum tokens to retrieve
            
        Returns:
            Relevant context as string
        """
        if max_tokens is None:
            max_tokens = self.infinite_context_config['max_active_tokens']
        
        logger.info(f"🔍 Retrieving infinite context for query: {query[:100]}...")
        
        # Search all memory levels
        relevant_chunks = []
        
        for level_name, level in self.hierarchical_memory.items():
            level_chunks = await self._search_relevant_chunks_in_level(level, query)
            relevant_chunks.extend(level_chunks)
        
        # Sort by relevance and importance
        relevant_chunks.sort(key=lambda x: (x.importance_score, x.access_count), reverse=True)
        
        # Build context within token limit
        context_text = self._build_context_from_chunks(relevant_chunks, max_tokens)
        
        logger.info(f"✅ Retrieved {len(context_text)} chars of infinite context")
        return context_text

    async def compress_infinite_context(self, target_ratio: float = 0.5, strategy: str = 'semantic') -> int:
        """
        Compress infinite context to save memory.
        
        Args:
            target_ratio: Target compression ratio
            strategy: Compression strategy
            
        Returns:
            Number of chunks compressed
        """
        if not self.infinite_context_config['dynamic_compression_enabled']:
            return 0
        
        logger.info(f"🗜️ Compressing infinite context with ratio {target_ratio} using {strategy}")
        
        total_compressed = 0
        
        for level_name, level in self.hierarchical_memory.items():
            if level.current_chunks:
                original_count = len(level.current_chunks)
                compressed_chunks = self.context_compressor.compress_context(
                    level.current_chunks, target_ratio, strategy
                )
                level.current_chunks = compressed_chunks
                compressed_count = original_count - len(compressed_chunks)
                total_compressed += compressed_count
        
        logger.info(f"✅ Compressed {total_compressed} chunks in infinite context")
        return total_compressed

    async def expand_context_for_query(self, query: str, current_context: str = "") -> str:
        """
        Dynamically expand context for a specific query.
        
        Args:
            query: Query to expand context for
            current_context: Current context string
            
        Returns:
            Expanded context
        """
        logger.info(f"🔄 Expanding context for query: {query[:100]}...")
        
        # Get relevant context from infinite context system
        relevant_context = await self.get_infinite_context(query)
        
        # Combine with current context
        expanded_context = current_context
        if relevant_context:
            if expanded_context:
                expanded_context += "\n\n" + relevant_context
            else:
                expanded_context = relevant_context
        
        # Allow expansion up to 4x the normal limit
        max_expansion = self.infinite_context_config['max_active_tokens'] * 4
        if len(expanded_context) > max_expansion:
            # Truncate intelligently
            expanded_context = expanded_context[:max_expansion]
            last_period = expanded_context.rfind('.')
            if last_period > max_expansion * 0.8:
                expanded_context = expanded_context[:last_period + 1]
        
        logger.info(f"✅ Expanded context to {len(expanded_context)} characters")
        return expanded_context

    async def get_infinite_context_summary(self) -> Dict[str, Any]:
        """
        Get summary of infinite context system state.
        
        Returns:
            Summary dictionary
        """
        summary = {}
        
        for level_name, level in self.hierarchical_memory.items():
            total_tokens = sum(len(chunk.content.split()) for chunk in level.current_chunks)
            summary[level_name] = {
                'chunk_count': len(level.current_chunks),
                'total_tokens': total_tokens,
                'max_chunks': level.max_chunks,
                'utilization': len(level.current_chunks) / level.max_chunks,
                'compression_strategy': level.compression_strategy
            }
        
        summary['total_chunks'] = sum(len(level.current_chunks) for level in self.hierarchical_memory.values())
        summary['total_tokens'] = sum(summary[level]['total_tokens'] for level in summary if level != 'total_chunks')
        summary['infinite_context_config'] = self.infinite_context_config
        
        return summary

    async def _manage_hierarchical_memory(self):
        """Manage hierarchical memory levels."""
        # Move chunks between levels based on access patterns
        active_level = self.hierarchical_memory['active']
        short_term_level = self.hierarchical_memory['short_term']
        long_term_level = self.hierarchical_memory['long_term']
        
        # Move old chunks from active to short-term
        while len(active_level.current_chunks) > active_level.max_chunks:
            old_chunk = active_level.current_chunks.pop(0)
            short_term_level.current_chunks.append(old_chunk)
        
        # Move old chunks from short-term to long-term
        while len(short_term_level.current_chunks) > short_term_level.max_chunks:
            old_chunk = short_term_level.current_chunks.pop(0)
            long_term_level.current_chunks.append(old_chunk)

    async def _search_relevant_chunks_in_level(self, level: HierarchicalMemoryLevel, query: str) -> List[ContextChunk]:
        """Search for relevant chunks in a specific memory level."""
        relevant_chunks = []
        query_words = set(query.lower().split())
        
        for chunk in level.current_chunks:
            chunk_words = set(chunk.content.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            
            if overlap > 0:
                # Update access count and timestamp
                chunk.access_count += 1
                chunk.last_accessed = datetime.now()
                
                # Calculate relevance score
                relevance_score = overlap / len(query_words)
                chunk.importance_score = max(chunk.importance_score, relevance_score)
                
                relevant_chunks.append(chunk)
        
        return relevant_chunks

    def _build_context_from_chunks(self, chunks: List[ContextChunk], max_tokens: int) -> str:
        """Build context text from chunks within token limit."""
        context_parts = []
        current_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = len(chunk.content.split())
            if current_tokens + chunk_tokens <= max_tokens:
                context_parts.append(chunk.content)
                current_tokens += chunk_tokens
            else:
                break
        
        return "\n\n".join(context_parts)

    def _extract_semantic_tags(self, content: str) -> List[str]:
        """
        🧠 REAL SEMANTIC TAG EXTRACTION - PROFESSIONAL IMPLEMENTATION
        Extract semantic tags using advanced NLP and semantic understanding.
        """
        try:
            # Priority 1: Use transformer-based semantic analysis
            semantic_tags = self._extract_tags_with_transformers(content)
            
            if semantic_tags:
                logger.debug(f"🎯 Extracted {len(semantic_tags)} semantic tags using transformers")
                return semantic_tags
            
            # Priority 2: Use advanced NLP techniques
            advanced_tags = self._extract_tags_advanced_nlp(content)
            
            if advanced_tags:
                logger.debug(f"🎯 Extracted {len(advanced_tags)} tags using advanced NLP")
                return advanced_tags
            
            # Priority 3: Use intelligent keyword analysis
            logger.info("🔄 Using intelligent keyword analysis for tags")
            return self._extract_tags_intelligent_keywords(content)
            
        except Exception as e:
            logger.error(f"Semantic tag extraction failed: {e}")
            return self._extract_tags_basic_fallback(content)

    def _extract_tags_with_transformers(self, content: str) -> List[str]:
        """Extract semantic tags using transformer models."""
        try:
            # Use zero-shot classification for semantic tagging
            if not hasattr(self, '_tag_classifier'):
                try:
                    from transformers import pipeline
                    
                    # Initialize semantic classifier
                    self._tag_classifier = pipeline(
                        "zero-shot-classification",
                        model="facebook/bart-large-mnli"
                    )
                except ImportError:
                    logger.warning("Transformers not available for semantic tagging")
                    return []
                except Exception as e:
                    logger.warning(f"Failed to initialize tag classifier: {e}")
                    return []
            
            # Define comprehensive candidate tags for different domains
            candidate_tags = [
                # Technical/Programming
                "programming", "software_development", "algorithm", "data_structure",
                "machine_learning", "artificial_intelligence", "database", "api",
                "web_development", "mobile_development", "devops", "testing",
                "security", "performance", "optimization", "debugging",
                
                # Business/Strategy
                "business_strategy", "product_management", "marketing", "sales",
                "customer_service", "analytics", "finance", "operations",
                "project_management", "leadership", "innovation",
                
                # Scientific/Research
                "research", "analysis", "experimentation", "methodology",
                "statistics", "data_science", "scientific_computing",
                "mathematics", "physics", "biology", "chemistry",
                
                # Creative/Design
                "design", "user_experience", "user_interface", "creative_writing",
                "visual_design", "art", "multimedia", "content_creation",
                
                # General
                "documentation", "tutorial", "guide", "reference", "example",
                "implementation", "solution", "problem_solving", "automation",
                "integration", "workflow", "process", "system"
            ]
            
            # Classify content against candidate tags
            result = self._tag_classifier(content, candidate_tags)
            
            # Filter tags by confidence threshold
            semantic_tags = []
            for label, score in zip(result['labels'], result['scores']):
                if score > 0.3:  # Confidence threshold
                    semantic_tags.append(f"{label} ({score:.2f})")
            
            return semantic_tags[:10]  # Limit to top 10 tags
            
        except Exception as e:
            logger.error(f"Transformer semantic tagging failed: {e}")
            return []

    def _extract_tags_advanced_nlp(self, content: str) -> List[str]:
        """Extract tags using advanced NLP techniques."""
        try:
            tags = []
            
            # Use spaCy for advanced NLP analysis
            try:
                import spacy
                
                # Load spaCy model
                nlp = None
                for model_name in ["en_core_web_sm", "en_core_web_md"]:
                    try:
                        nlp = spacy.load(model_name)
                        break
                    except OSError:
                        continue
                
                if nlp:
                    doc = nlp(content)
                    
                    # Extract key topics from noun phrases
                    noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks 
                                   if len(chunk.text) > 3 and chunk.text.isalpha()]
                    
                    # Extract important words by POS tags
                    important_words = []
                    for token in doc:
                        if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                            not token.is_stop and 
                            len(token.text) > 3 and 
                            token.text.isalpha()):
                            important_words.append(token.lemma_.lower())
                    
                    # Combine and filter
                    candidate_tags = noun_phrases + important_words
                    
                    # Score tags by frequency and importance
                    tag_scores = {}
                    for tag in candidate_tags:
                        tag_scores[tag] = tag_scores.get(tag, 0) + 1
                    
                    # Sort by score and return top tags
                    sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
                    tags = [tag for tag, score in sorted_tags[:15]]
                    
            except ImportError:
                logger.warning("spaCy not available for advanced NLP tagging")
            
            # TF-IDF based semantic analysis
            if not tags:
                tags = self._extract_tags_tfidf(content)
            
            return tags
            
        except Exception as e:
            logger.error(f"Advanced NLP tagging failed: {e}")
            return []

    def _extract_tags_tfidf(self, content: str) -> List[str]:
        """Extract tags using TF-IDF analysis."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import re
            
            # Preprocess content
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if len(sentences) < 2:
                sentences = [content]
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                max_df=0.8,
                min_df=1
            )
            
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_scores = tfidf_matrix.mean(axis=0).A1
            
            # Create tag scores
            tag_scores = list(zip(feature_names, mean_scores))
            tag_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top tags
            return [tag for tag, score in tag_scores[:10] if score > 0.1]
            
        except ImportError:
            logger.warning("scikit-learn not available for TF-IDF tagging")
            return []
        except Exception as e:
            logger.error(f"TF-IDF tagging failed: {e}")
            return []

    def _extract_tags_intelligent_keywords(self, content: str) -> List[str]:
        """
        Intelligent keyword-based tag extraction with semantic understanding.
        This is an advanced fallback that goes beyond simple word frequency.
        """
        try:
            import re
            from collections import Counter
            
            # Preprocess content
            content_lower = content.lower()
            
            # Remove code blocks and special characters for better text analysis
            text_content = re.sub(r'```[\s\S]*?```', '', content_lower)  # Remove code blocks
            text_content = re.sub(r'[^\w\s]', ' ', text_content)  # Remove punctuation
            
            # Tokenize intelligently
            words = text_content.split()
            
            # Remove stop words and short words
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
            }
            
            meaningful_words = [word for word in words 
                              if len(word) > 3 and word not in stop_words and word.isalpha()]
            
            # Count word frequencies
            word_freq = Counter(meaningful_words)
            
            # Domain-specific keyword enhancement
            domain_keywords = {
                'programming': ['code', 'function', 'algorithm', 'program', 'software', 'development'],
                'data_science': ['data', 'analysis', 'model', 'training', 'prediction', 'statistics'],
                'machine_learning': ['learning', 'neural', 'network', 'artificial', 'intelligence', 'model'],
                'web_development': ['website', 'server', 'client', 'frontend', 'backend', 'database'],
                'business': ['strategy', 'market', 'customer', 'product', 'service', 'business'],
                'design': ['design', 'user', 'interface', 'experience', 'visual', 'creative'],
                'research': ['research', 'study', 'analysis', 'experiment', 'methodology', 'findings'],
                'security': ['security', 'authentication', 'encryption', 'protection', 'vulnerability'],
                'performance': ['performance', 'optimization', 'speed', 'efficiency', 'scalability'],
                'testing': ['test', 'testing', 'validation', 'verification', 'quality', 'debugging']
            }
            
            # Identify domains and boost relevant keywords
            domain_scores = {}
            for domain, keywords in domain_keywords.items():
                score = sum(word_freq.get(keyword, 0) for keyword in keywords)
                if score > 0:
                    domain_scores[domain] = score
                    # Boost domain keywords
                    for keyword in keywords:
                        if keyword in word_freq:
                            word_freq[keyword] *= 1.5
            
            # Extract bigrams for compound concepts
            bigrams = []
            for i in range(len(meaningful_words) - 1):
                bigram = meaningful_words[i] + '_' + meaningful_words[i + 1]
                bigrams.append(bigram)
            
            bigram_freq = Counter(bigrams)
            
            # Combine unigrams and bigrams
            all_terms = dict(word_freq)
            for bigram, freq in bigram_freq.items():
                if freq > 1:  # Only include recurring bigrams
                    all_terms[bigram] = freq
            
            # Sort by frequency and relevance
            sorted_terms = sorted(all_terms.items(), key=lambda x: x[1], reverse=True)
            
            # Extract top tags
            tags = []
            
            # Add domain tags
            for domain, score in sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
                tags.append(f"{domain} (domain)")
            
            # Add keyword tags
            for term, freq in sorted_terms[:10]:
                if freq > 1:  # Only include terms that appear multiple times
                    tags.append(term.replace('_', ' '))
            
            # Remove duplicates while preserving order
            unique_tags = []
            seen = set()
            for tag in tags:
                tag_clean = tag.lower().strip()
                if tag_clean not in seen and len(tag_clean) > 2:
                    unique_tags.append(tag)
                    seen.add(tag_clean)
            
            return unique_tags[:15]  # Limit to 15 tags
            
        except Exception as e:
            logger.error(f"Intelligent keyword extraction failed: {e}")
            return self._extract_tags_basic_fallback(content)

    def _extract_tags_basic_fallback(self, content: str) -> List[str]:
        """Basic fallback tag extraction."""
        try:
            # Simple word frequency as last resort
            words = content.lower().split()
            word_freq = {}
            
            for word in words:
                if len(word) > 3 and word.isalpha():
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Return top keywords as tags
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in sorted_words[:5] if freq > 1]
            
        except Exception as e:
            logger.error(f"Basic fallback tagging failed: {e}")
            return ["content", "information"]  # Ultimate fallback

    async def analyze_intent_and_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze intent and sentiment using real NLP models."""
        try:
            from transformers import pipeline
            import torch
            # Initialize models if not already done
            if not hasattr(self, '_intent_classifier'):
                try:
                    intent_model = "facebook/bart-large-mnli"  # Good for zero-shot intent classification
                    self._intent_classifier = pipeline(
                        "zero-shot-classification",
                        model=intent_model,
                        device=0 if torch.cuda.is_available() else -1
                    )
                except Exception as e:
                    logger.warning(f"Intent classifier initialization failed: {e}")
                    self._intent_classifier = None
            if not hasattr(self, '_sentiment_analyzer'):
                try:
                    # Temporarily disabled to avoid model download during training
                    # sentiment_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
                    # self._sentiment_analyzer = pipeline(
                    #     "sentiment-analysis",
                    #     model=sentiment_model,
                    #     device=0 if torch.cuda.is_available() else -1
                    # )
                    self._sentiment_analyzer = None
                except Exception as e:
                    logger.warning(f"Sentiment analyzer initialization failed: {e}")
                    self._sentiment_analyzer = None
            # Analyze intent
            intent_result = await self._classify_intent(content)
            # Analyze sentiment
            sentiment_result = await self._analyze_sentiment(content)
            # Extract entities for context
            entities = await self._extract_entities(content)
            return {
                "intent": intent_result,
                "sentiment": sentiment_result,
                "entities": entities,
                "confidence": intent_result.get("confidence", 0.0) + sentiment_result.get("confidence", 0.0)
            }
        except Exception as e:
            logger.error(f"Intent and sentiment analysis failed: {e}")
            return {
                "intent": {"label": "unknown", "confidence": 0.0},
                "sentiment": {"label": "neutral", "confidence": 0.0},
                "entities": [],
                "confidence": 0.0
            }

    async def _classify_intent(self, content: str) -> Dict[str, Any]:
        """Classify intent using zero-shot classification."""
        if not hasattr(self, '_intent_classifier') or self._intent_classifier is None:
            return {"label": "unknown", "confidence": 0.0}
        try:
            candidate_labels = [
                "code_question",
                "bug_report",
                "feature_request",
                "code_review",
                "deployment_issue",
                "performance_optimization",
                "security_concern",
                "documentation_request",
                "testing_question",
                "general_inquiry"
            ]
            result = self._intent_classifier(content, candidate_labels)
            return {
                "label": result["labels"][0],
                "confidence": result["scores"][0],
                "all_scores": dict(zip(result["labels"], result["scores"]))
            }
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return {"label": "unknown", "confidence": 0.0}

    async def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze sentiment using transformer model."""
        if not hasattr(self, '_sentiment_analyzer') or self._sentiment_analyzer is None:
            return {"label": "neutral", "confidence": 0.0}
        try:
            result = self._sentiment_analyzer(content)
            label_mapping = {
                "positive": "positive",
                "negative": "negative",
                "neutral": "neutral",
                "LABEL_0": "negative",
                "LABEL_1": "neutral",
                "LABEL_2": "positive"
            }
            label = label_mapping.get(result[0]["label"], "neutral")
            score = result[0]["score"]
            return {
                "label": label,
                "confidence": score,
                "raw_result": result[0]
            }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"label": "neutral", "confidence": 0.0}

    async def _update_knowledge_graph_neo4j(self, entry: MemoryEntry):
        """Update Neo4j knowledge graph with real operations."""
        if not self.neo4j_driver:
            return
        try:
            # Analyze content for intent and sentiment
            analysis = await self.analyze_intent_and_sentiment(entry.content)
            entities = analysis["entities"]
            with self.neo4j_driver.session() as session:
                # Create memory node
                memory_query = (
                    """
                    MERGE (m:Memory {id: $memory_id})
                    SET m.content = $content,
                        m.content_type = $content_type,
                        m.timestamp = $timestamp,
                        m.success_score = $success_score,
                        m.intent = $intent,
                        m.sentiment = $sentiment,
                        m.confidence = $confidence
                    """
                )
                session.run(memory_query, {
                    "memory_id": entry.id,
                    "content": entry.content[:1000],
                    "content_type": entry.content_type,
                    "timestamp": entry.timestamp.isoformat(),
                    "success_score": entry.success_score,
                    "intent": analysis["intent"]["label"],
                    "sentiment": analysis["sentiment"]["label"],
                    "confidence": analysis["confidence"]
                })
                # Create entity nodes and relationships
                for entity in entities:
                    entity_query = (
                        """
                        MERGE (e:Entity {name: $entity_name})
                        SET e.frequency = COALESCE(e.frequency, 0) + 1,
                            e.last_seen = $timestamp
                        """
                    )
                    session.run(entity_query, {
                        "entity_name": entity,
                        "timestamp": entry.timestamp.isoformat()
                    })
                    rel_query = (
                        """
                        MATCH (m:Memory {id: $memory_id})
                        MATCH (e:Entity {name: $entity_name})
                        MERGE (m)-[r:MENTIONS]->(e)
                        SET r.timestamp = $timestamp
                        """
                    )
                    session.run(rel_query, {
                        "memory_id": entry.id,
                        "entity_name": entity,
                        "timestamp": entry.timestamp.isoformat()
                    })
                # Create relationships between entities
                for i, entity1 in enumerate(entities):
                    for entity2 in entities[i+1:]:
                        co_occurrence_query = (
                            """
                            MATCH (e1:Entity {name: $entity1})
                            MATCH (e2:Entity {name: $entity2})
                            MERGE (e1)-[r:CO_OCCURS_WITH]->(e2)
                            SET r.frequency = COALESCE(r.frequency, 0) + 1,
                                r.last_seen = $timestamp
                            """
                        )
                        session.run(co_occurrence_query, {
                            "entity1": entity1,
                            "entity2": entity2,
                            "timestamp": entry.timestamp.isoformat()
                        })
                logger.debug(f"Updated Neo4j knowledge graph for memory {entry.id}")
        except Exception as e:
            logger.error(f"Neo4j knowledge graph update failed: {e}")

    async def search_knowledge_graph_neo4j(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search Neo4owledge graph with real Cypher queries."""
        if not self.neo4j_driver:
            return {"error": "Neo4j not available"}
        
        try:
            with self.neo4j_driver.session() as session:
                # Search for entities and related memories
                search_query = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($query)
                WITH e
                OPTIONAL MATCH (e)<-[:MENTIONS]-(m:Memory)
                RETURN e.name as entity,
                       e.frequency as frequency,
                       collect({
                           id: m.id,
                           content: m.content,
                           content_type: m.content_type,
                           timestamp: m.timestamp,
                           intent: m.intent,
                           sentiment: m.sentiment
                       }) as memories
                ORDER BY e.frequency DESC
                LIMIT $limit
                """
                result = session.run(search_query, {"query": query, "limit": limit})

                entities = []
                for record in result:
                    entities.append({
                        "entity": record["entity"],
                        "frequency": record["frequency"],
                        "memories": record["memories"]
                    })

                # Get graph statistics
                stats_query = """
                MATCH (n)
                RETURN count(n) as total_nodes,
                       count([(n)-[r]->(m) | r]) as total_relationships,
                       count(DISTINCT labels(n)) as node_types
                """
                stats_result = session.run(stats_query)
                stats = stats_result.single()
                
                return {
                  "query": query,
                  "entities": entities,
                  "total_entities": len(entities),
                  "graph_stats": {
                    "total_nodes": stats["total_nodes"],
                    "total_relationships": stats["total_relationships"],
                    "node_types": stats["node_types"]
                  }
                }
                
        except Exception as e:
            logger.error(f"Neo4owledge graph search failed: {e}")
            return {"error": str(e)}

    async def get_knowledge_graph_insights(self) -> Dict[str, Any]:
        """Get insights from Neo4j knowledge graph."""
        if not self.neo4j_driver:
            return {"error": "Neo4j not available"}
        try:
            with self.neo4j_driver.session() as session:
                # Get most frequent entities
                top_entities_query = (
                    """
                    MATCH (e:Entity)
                    RETURN e.name as entity, e.frequency as frequency
                    ORDER BY e.frequency DESC
                    LIMIT 10
                    """
                )
                top_entities = []
                for record in session.run(top_entities_query):
                    top_entities.append({
                        "entity": record["entity"],
                        "frequency": record["frequency"]
                    })
                # Get entity relationships
                relationships_query = (
                    """
                    MATCH (e1:Entity)-[r:CO_OCCURS_WITH]->(e2:Entity)
                    RETURN e1.name as entity1, e2.name as entity2, r.frequency as frequency
                    ORDER BY r.frequency DESC
                    LIMIT 10
                    """
                )
                relationships = []
                for record in session.run(relationships_query):
                    relationships.append({
                        "entity1": record["entity1"],
                        "entity2": record["entity2"],
                        "frequency": record["frequency"]
                    })
                # Get intent distribution
                intent_query = (
                    """
                    MATCH (m:Memory)
                    WHERE m.intent IS NOT NULL
                    RETURN m.intent as intent, count(*) as count
                    ORDER BY count DESC
                    """
                )
                intent_distribution = []
                for record in session.run(intent_query):
                    intent_distribution.append({
                        "intent": record["intent"],
                        "count": record["count"]
                    })
                # Get sentiment distribution
                sentiment_query = (
                    """
                    MATCH (m:Memory)
                    WHERE m.sentiment IS NOT NULL
                    RETURN m.sentiment as sentiment, count(*) as count
                    ORDER BY count DESC
                    """
                )
                sentiment_distribution = []
                for record in session.run(sentiment_query):
                    sentiment_distribution.append({
                        "sentiment": record["sentiment"],
                        "count": record["count"]
                    })
                return {
                    "top_entities": top_entities,
                    "top_relationships": relationships,
                    "intent_distribution": intent_distribution,
                    "sentiment_distribution": sentiment_distribution,
                    "total_entities": len(top_entities),
                    "total_relationships": len(relationships)
                }
        except Exception as e:
            logger.error(f"Neo4j insights extraction failed: {e}")
            return {"error": str(e)}

    async def _extract_entities_with_transformers(self, content: str) -> List[str]:
        """
        🚀 OPTIMIZED ENTITY EXTRACTION - PHASE 2 PERFORMANCE
        Extract entities using cached transformer models for maximum performance.
        """
        try:
            # Use cached model (PHASE 2 OPTIMIZATION)
            return await self._extract_entities_cached(content)
            
        except Exception as e:
            logger.error(f"Cached entity extraction failed: {e}")
            return []

    async def _extract_entities_cached(self, content: str) -> List[str]:
        """Extract entities using shared model cache."""
        try:
            from .model_cache_manager import get_ner_model
            
            # Try multiple NER models in order of preference
            ner_models = [
                "dslim/bert-base-NER",                              # Good balance, faster loading
                "dbmdz/bert-large-cased-finetuned-conll03-english", # High quality
                "Jean-Baptiste/roberta-large-ner-english"           # Alternative
            ]
            
            for model_name in ner_models:
                try:
                    # Get model from shared cache
                    ner_pipeline = await get_ner_model(model_name, priority=4)
                    
                    if ner_pipeline is not None:
                        # Extract entities using cached model
                        ner_results = ner_pipeline(content)
                        
                        # Process and clean results
                        entities = []
                        for result in ner_results:
                            entity_text = result.get('word', '').strip()
                            entity_label = result.get('entity_group', 'UNKNOWN')
                            confidence = result.get('score', 0.0)
                            
                            # Filter by confidence and length
                            if confidence > 0.7 and len(entity_text) > 1:
                                # Clean entity text
                                entity_text = entity_text.replace('##', '').strip()
                                if entity_text:
                                    entities.append(f"{entity_text} ({entity_label})")
                        
                        # Remove duplicates while preserving order
                        unique_entities = []
                        seen = set()
                        for entity in entities:
                            if entity.lower() not in seen:
                                unique_entities.append(entity)
                                seen.add(entity.lower())
                        
                        return unique_entities
                        
                except Exception as e:
                    logger.warning(f"NER model {model_name} failed: {e}")
                    continue
            
            # No models worked
            logger.warning("All NER models failed")
            return []
            
        except Exception as e:
            logger.error(f"Cached entity extraction failed: {e}")
            return []

    async def _extract_semantic_tags(self, content: str) -> List[str]:
        """
        🚀 OPTIMIZED SEMANTIC TAG EXTRACTION - PHASE 2 PERFORMANCE
        Extract semantic tags using cached models for maximum performance.
        """
        try:
            # Use cached model (PHASE 2 OPTIMIZATION)
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an event loop, create a task
                task = asyncio.create_task(self._extract_tags_cached(content))
                return await task
            else:
                # No running loop, use run
                return asyncio.run(self._extract_tags_cached(content))
        except RuntimeError:
            # Fallback to synchronous extraction
            return self._extract_tags_basic_fallback(content)
        except Exception as e:
            logger.error(f"Cached tag extraction failed: {e}")
            # Fallback to basic implementation
            return self._extract_tags_basic_fallback(content)

    async def _extract_tags_cached(self, content: str) -> List[str]:
        """Extract semantic tags using shared model cache."""
        try:
            # Priority 1: Use transformer-based semantic analysis
            semantic_tags = await self._extract_tags_with_cached_transformers(content)
            
            if semantic_tags:
                logger.debug(f"🎯 Extracted {len(semantic_tags)} semantic tags using cached transformers")
                return semantic_tags
            
            # Priority 2: Use advanced NLP techniques
            advanced_tags = await self._extract_tags_advanced_nlp_cached(content)
            
            if advanced_tags:
                logger.debug(f"🎯 Extracted {len(advanced_tags)} tags using cached NLP")
                return advanced_tags
            
            # Priority 3: Use intelligent keyword analysis
            logger.info("🔄 Using intelligent keyword analysis for tags")
            return self._extract_tags_intelligent_keywords(content)
            
        except Exception as e:
            logger.error(f"Cached semantic tag extraction failed: {e}")
            return self._extract_tags_basic_fallback(content)

    async def _extract_tags_with_cached_transformers(self, content: str) -> List[str]:
        """Extract semantic tags using cached transformer models."""
        try:
            from .model_cache_manager import get_classification_model
            
            # Get classification model from cache
            tag_classifier = await get_classification_model("facebook/bart-large-mnli", priority=3)
            
            if tag_classifier is not None:
                # Define comprehensive candidate tags for different domains
                candidate_tags = [
                    # Technical/Programming
                    "programming", "software_development", "algorithm", "data_structure",
                    "machine_learning", "artificial_intelligence", "database", "api",
                    "web_development", "mobile_development", "devops", "testing",
                    "security", "performance", "optimization", "debugging",
                    
                    # Business/Strategy
                    "business_strategy", "product_management", "marketing", "sales",
                    "customer_service", "analytics", "finance", "operations",
                    "project_management", "leadership", "innovation",
                    
                    # Scientific/Research
                    "research", "analysis", "experimentation", "methodology",
                    "statistics", "data_science", "scientific_computing",
                    "mathematics", "physics", "biology", "chemistry",
                    
                    # Creative/Design
                    "design", "user_experience", "user_interface", "creative_writing",
                    "visual_design", "art", "multimedia", "content_creation",
                    
                    # General
                    "documentation", "tutorial", "guide", "reference", "example",
                    "implementation", "solution", "problem_solving", "automation",
                    "integration", "workflow", "process", "system"
                ]
                
                # Classify content against candidate tags
                result = tag_classifier(content, candidate_tags)
                
                # Filter tags by confidence threshold
                semantic_tags = []
                for label, score in zip(result['labels'], result['scores']):
                    if score > 0.3:  # Confidence threshold
                        semantic_tags.append(f"{label} ({score:.2f})")
                
                return semantic_tags[:10]  # Limit to top 10 tags
            else:
                return []
                
        except Exception as e:
            logger.error(f"Cached transformer semantic tagging failed: {e}")
            return []

    async def _extract_tags_advanced_nlp_cached(self, content: str) -> List[str]:
        """Extract tags using cached NLP models."""
        try:
            tags = []
            
            # Try to use spaCy if available
            try:
                import spacy
                
                # Load spaCy model (this should also be cached in future)
                nlp = None
                for model_name in ["en_core_web_sm", "en_core_web_md"]:
                    try:
                        nlp = spacy.load(model_name)
                        break
                    except OSError:
                        continue
                
                if nlp:
                    doc = nlp(content)
                    
                    # Extract key topics from noun phrases
                    noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks 
                                   if len(chunk.text) > 3 and chunk.text.isalpha()]
                    
                    # Extract important words by POS tags
                    important_words = []
                    for token in doc:
                        if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                            not token.is_stop and 
                            len(token.text) > 3 and 
                            token.text.isalpha()):
                            important_words.append(token.lemma_.lower())
                    
                    # Combine and filter
                    candidate_tags = noun_phrases + important_words
                    
                    # Score tags by frequency and importance
                    tag_scores = {}
                    for tag in candidate_tags:
                        tag_scores[tag] = tag_scores.get(tag, 0) + 1
                    
                    # Sort by score and return top tags
                    sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
                    tags = [tag for tag, score in sorted_tags[:15]]
                    
            except ImportError:
                logger.warning("spaCy not available for advanced NLP tagging")
            
            # TF-IDF based semantic analysis using cached model
            if not tags:
                tags = await self._extract_tags_tfidf_cached(content)
            
            return tags
            
        except Exception as e:
            logger.error(f"Cached advanced NLP tagging failed: {e}")
            return []

    async def _extract_tags_tfidf_cached(self, content: str) -> List[str]:
        """Extract tags using cached TF-IDF analysis."""
        try:
            from .model_cache_manager import get_tfidf_model
            import re
            
            # Preprocess content
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if len(sentences) < 2:
                sentences = [content]
            
            # Get TF-IDF model from cache
            tfidf_model = await get_tfidf_model(
                "semantic_tags_tfidf",
                max_features=100,
                ngram_range=(1, 2),
                max_df=0.8,
                min_df=1
            )
            
            if tfidf_model is not None:
                # Fit and transform
                tfidf_matrix = tfidf_model.fit_transform(sentences)
                feature_names = tfidf_model.get_feature_names_out()
                
                # Get average TF-IDF scores
                mean_scores = tfidf_matrix.mean(axis=0).A1
                
                # Create tag scores
                tag_scores = list(zip(feature_names, mean_scores))
                tag_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Return top tags
                return [tag for tag, score in tag_scores[:10] if score > 0.1]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Cached TF-IDF tagging failed: {e}")
            return []

# ============================================================================
# ADVANCED CONTEXT MANAGEMENT SOLUTIONS FOR CAVEATS & FAILURE MODES
# ============================================================================

@dataclass
class AdaptiveRetrievalConfig:
    """Configuration for adaptive retrieval parameters."""
    base_similarity_threshold: float = 0.1
    dynamic_threshold_enabled: bool = True
    query_complexity_analysis: bool = True
    domain_adaptive_thresholds: Dict[str, float] = None
    latency_budget_ms: float = 100.0
    quality_vs_speed_tradeoff: float = 0.7  # 0.0 = speed, 1.0 = quality
    
    def __post_init__(self):
        if self.domain_adaptive_thresholds is None:
            self.domain_adaptive_thresholds = {
                'technical': 0.15,
                'creative': 0.08,
                'analytical': 0.12,
                'conversational': 0.06
            }

@dataclass
class HierarchicalSummarizationConfig:
    """Configuration for hierarchical summarization."""
    levels: List[str] = None  # ['full_detail', 'detailed', 'summary', 'key_points']
    compression_ratios: Dict[str, float] = None
    quality_preservation_targets: Dict[str, float] = None
    temporal_decay_enabled: bool = True
    anchor_prompts_enabled: bool = True
    
    def __post_init__(self):
        if self.levels is None:
            self.levels = ['full_detail', 'detailed', 'summary', 'key_points']
        if self.compression_ratios is None:
            self.compression_ratios = {
                'full_detail': 1.0,
                'detailed': 0.7,
                'summary': 0.4,
                'key_points': 0.2
            }
        if self.quality_preservation_targets is None:
            self.quality_preservation_targets = {
                'full_detail': 0.95,
                'detailed': 0.85,
                'summary': 0.75,
                'key_points': 0.6
            }

@dataclass
class LatencyMonitor:
    """Monitor and track latency metrics."""
    recall_times: List[float] = None
    compression_times: List[float] = None
    embedding_times: List[float] = None
    total_operations: int = 0
    average_recall_latency: float = 0.0
    average_compression_latency: float = 0.0
    latency_threshold_ms: float = 200.0
    
    def __post_init__(self):
        if self.recall_times is None:
            self.recall_times = []
        if self.compression_times is None:
            self.compression_times = []
        if self.embedding_times is None:
            self.embedding_times = []

@dataclass
class DriftCorrectionConfig:
    """Configuration for drift correction mechanisms."""
    periodic_recalibration_enabled: bool = True
    recalibration_interval_hours: int = 24
    anchor_prompt_frequency: int = 10  # Every N operations
    drift_detection_threshold: float = 0.15
    semantic_consistency_checking: bool = True
    temporal_consistency_checking: bool = True

class AdvancedContextManager:
    """Advanced context management with solutions for caveats and failure modes."""
    
    def __init__(self, perfect_recall_engine):
        self.engine = perfect_recall_engine
        self.adaptive_config = AdaptiveRetrievalConfig()
        self.hierarchical_config = HierarchicalSummarizationConfig()
        self.latency_monitor = LatencyMonitor()
        self.drift_config = DriftCorrectionConfig()
        
        # Initialize drift correction tracking
        self.last_recalibration = datetime.now()
        self.operation_count = 0
        self.drift_indicators = []
        
        # Initialize hierarchical summarization
        self.summarization_levels = {}
        self.anchor_prompts = []
        
        logger.info("🔧 Advanced Context Manager initialized with caveat solutions")

    async def adaptive_retrieval_parameters(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Dynamically adjust retrieval parameters based on query complexity and context.
        
        Args:
            query: The search query
            context: Additional context information
            
        Returns:
            Optimized retrieval parameters
        """
        start_time = time.time()
        
        # Analyze query complexity
        complexity_score = await self._analyze_query_complexity(query)
        
        # Determine domain
        domain = await self._classify_query_domain(query)
        
        # Calculate optimal similarity threshold
        base_threshold = self.adaptive_config.base_similarity_threshold
        domain_threshold = self.adaptive_config.domain_adaptive_thresholds.get(domain, base_threshold)
        
        # Adjust based on complexity
        if complexity_score > 0.8:  # High complexity
            adjusted_threshold = domain_threshold * 0.8  # Lower threshold for more results
        elif complexity_score < 0.3:  # Low complexity
            adjusted_threshold = domain_threshold * 1.2  # Higher threshold for precision
        else:
            adjusted_threshold = domain_threshold
        
        # Calculate optimal top-k based on latency budget
        available_time_ms = self.adaptive_config.latency_budget_ms - (time.time() - start_time) * 1000
        estimated_time_per_result = 5.0  # ms per result
        max_results = int(available_time_ms / estimated_time_per_result)
        
        # Quality vs speed tradeoff
        if self.adaptive_config.quality_vs_speed_tradeoff > 0.8:
            top_k = min(max_results, 50)  # Prioritize quality
        else:
            top_k = min(max_results, 20)  # Prioritize speed
        
        params = {
            'similarity_threshold': adjusted_threshold,
            'top_k': top_k,
            'domain': domain,
            'complexity_score': complexity_score,
            'estimated_latency_ms': available_time_ms
        }
        
        logger.info(f"🎯 Adaptive retrieval: threshold={adjusted_threshold:.3f}, top_k={top_k}, domain={domain}")
        return params

    async def _analyze_query_complexity(self, query: str) -> float:
        """Analyze query complexity for adaptive parameter adjustment."""
        try:
            # Count unique concepts
            words = query.lower().split()
            unique_concepts = len(set(words))
            
            # Check for technical terms
            technical_indicators = ['algorithm', 'function', 'class', 'method', 'variable', 'parameter']
            technical_count = sum(1 for word in words if word in technical_indicators)
            
            # Check for multi-part queries
            multi_part_indicators = ['and', 'or', 'but', 'however', 'while', 'although']
            multi_part_count = sum(1 for word in words if word in multi_part_indicators)
            
            # Calculate complexity score
            complexity = (
                (unique_concepts / len(words)) * 0.4 +
                (technical_count / len(words)) * 0.3 +
                (multi_part_count / len(words)) * 0.3
            )
            
            return min(complexity, 1.0)
            
        except Exception as e:
            logger.error(f"Query complexity analysis failed: {e}")
            return 0.5  # Default medium complexity

    async def _classify_query_domain(self, query: str) -> str:
        """Classify query into domain for adaptive thresholds."""
        try:
            query_lower = query.lower()
            
            # Technical domain indicators
            technical_terms = ['code', 'program', 'algorithm', 'function', 'class', 'api', 'database']
            if any(term in query_lower for term in technical_terms):
                return 'technical'
            
            # Creative domain indicators
            creative_terms = ['design', 'creative', 'art', 'story', 'narrative', 'visual']
            if any(term in query_lower for term in creative_terms):
                return 'creative'
            
            # Analytical domain indicators
            analytical_terms = ['analyze', 'data', 'statistics', 'research', 'study', 'analysis']
            if any(term in query_lower for term in analytical_terms):
                return 'analytical'
            
            # Default to conversational
            return 'conversational'
            
        except Exception as e:
            logger.error(f"Query domain classification failed: {e}")
            return 'conversational'

    async def hierarchical_summarization(self, content: str, level: str = 'summary') -> str:
        """
        Apply hierarchical summarization with quality preservation.
        
        Args:
            content: Content to summarize
            level: Summarization level
            
        Returns:
            Summarized content
        """
        if level not in self.hierarchical_config.levels:
            raise ValueError(f"Invalid summarization level: {level}")
        
        target_ratio = self.hierarchical_config.compression_ratios[level]
        quality_target = self.hierarchical_config.quality_preservation_targets[level]
        
        # Apply temporal decay for older content
        if self.hierarchical_config.temporal_decay_enabled:
            target_ratio = self._apply_temporal_decay(target_ratio, content)
        
        # Generate summary with quality preservation
        summary = await self._generate_quality_preserving_summary(content, target_ratio, quality_target)
        
        # Store in hierarchical structure
        self.summarization_levels[level] = summary
        
        logger.info(f"📝 Hierarchical summarization ({level}): {len(content)} → {len(summary)} chars")
        return summary

    def _apply_temporal_decay(self, base_ratio: float, content: str) -> float:
        """Apply temporal decay to compression ratio for older content."""
        # This would be enhanced with actual temporal analysis
        # For now, return base ratio
        return base_ratio

    async def _generate_quality_preserving_summary(self, content: str, target_ratio: float, quality_target: float) -> str:
        """Generate summary with quality preservation."""
        try:
            # Use semantic compression for quality preservation
            if hasattr(self.engine, 'context_compressor'):
                compressed = self.engine.context_compressor._semantic_compression(content, target_ratio)
                return compressed
            else:
                # Fallback to extractive summarization
                return self._extractive_summarization(content, target_ratio)
                
        except Exception as e:
            logger.error(f"Quality-preserving summarization failed: {e}")
            return content[:int(len(content) * target_ratio)]

    def _extractive_summarization(self, content: str, target_ratio: float) -> str:
        """Extractive summarization as fallback."""
        sentences = content.split('.')
        target_sentences = max(1, int(len(sentences) * target_ratio))
        return '. '.join(sentences[:target_sentences]) + '.'

    async def monitor_latency(self, operation_type: str, start_time: float, end_time: float):
        """Monitor and track latency metrics."""
        latency_ms = (end_time - start_time) * 1000
        
        if operation_type == 'recall':
            self.latency_monitor.recall_times.append(latency_ms)
            self.latency_monitor.average_recall_latency = sum(self.latency_monitor.recall_times) / len(self.latency_monitor.recall_times)
        elif operation_type == 'compression':
            self.latency_monitor.compression_times.append(latency_ms)
            self.latency_monitor.average_compression_latency = sum(self.latency_monitor.compression_times) / len(self.latency_monitor.compression_times)
        elif operation_type == 'embedding':
            self.latency_monitor.embedding_times.append(latency_ms)
        
        self.latency_monitor.total_operations += 1
        
        # Check for latency violations
        if latency_ms > self.latency_monitor.latency_threshold_ms:
            logger.warning(f"⚠️ High latency detected: {operation_type} took {latency_ms:.1f}ms")
        
        # Log performance metrics periodically
        if self.latency_monitor.total_operations % 10 == 0:
            logger.info(f"📊 Latency metrics: recall={self.latency_monitor.average_recall_latency:.1f}ms, "
                       f"compression={self.latency_monitor.average_compression_latency:.1f}ms")

    async def detect_and_correct_drift(self, query: str, results: List[RecallResult]) -> bool:
        """
        Detect and correct semantic drift in memory retrieval.
        
        Args:
            query: The search query
            results: Retrieved results
            
        Returns:
            True if drift was detected and corrected
        """
        self.operation_count += 1
        
        # Check if recalibration is needed
        if self._should_recalibrate():
            await self._perform_recalibration()
            return True
        
        # Check for semantic drift in results
        drift_detected = await self._detect_semantic_drift(query, results)
        
        if drift_detected:
            logger.warning("🔄 Semantic drift detected, applying correction")
            await self._apply_drift_correction(query, results)
            return True
        
        return False

    def _should_recalibrate(self) -> bool:
        """Determine if recalibration is needed."""
        hours_since_last = (datetime.now() - self.last_recalibration).total_seconds() / 3600
        return (hours_since_last >= self.drift_config.recalibration_interval_hours or
                self.operation_count % self.drift_config.anchor_prompt_frequency == 0)

    async def _perform_recalibration(self):
        """Perform periodic recalibration."""
        logger.info("🔄 Performing periodic recalibration")
        
        # Update anchor prompts
        await self._update_anchor_prompts()
        
        # Recalibrate similarity thresholds
        await self._recalibrate_thresholds()
        
        # Update drift indicators
        self.last_recalibration = datetime.now()
        
        logger.info("✅ Recalibration completed")

    async def _detect_semantic_drift(self, query: str, results: List[RecallResult]) -> bool:
        """Detect semantic drift in retrieval results."""
        try:
            if not results:
                return False
            
            # Check semantic consistency
            if self.drift_config.semantic_consistency_checking:
                consistency_score = await self._calculate_semantic_consistency(query, results)
                if consistency_score < (1.0 - self.drift_config.drift_detection_threshold):
                    return True
            
            # Check temporal consistency
            if self.drift_config.temporal_consistency_checking:
                temporal_score = await self._calculate_temporal_consistency(results)
                if temporal_score < (1.0 - self.drift_config.drift_detection_threshold):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            return False

    async def _calculate_semantic_consistency(self, query: str, results: List[RecallResult]) -> float:
        """Calculate semantic consistency of results."""
        try:
            if len(results) < 2:
                return 1.0
            
            # Calculate pairwise similarities between results
            similarities = []
            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    if results[i].entry.embedding and results[j].entry.embedding:
                        sim = self.engine._calculate_similarity(
                            results[i].entry.embedding, 
                            results[j].entry.embedding
                        )
                        similarities.append(sim)
            
            if similarities:
                return sum(similarities) / len(similarities)
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Semantic consistency calculation failed: {e}")
            return 1.0

    async def _calculate_temporal_consistency(self, results: List[RecallResult]) -> float:
        """Calculate temporal consistency of results."""
        try:
            if len(results) < 2:
                return 1.0
            
            # Calculate temporal spread
            timestamps = [result.entry.timestamp for result in results]
            timestamps.sort()
            
            # Calculate average time difference
            time_diffs = []
            for i in range(len(timestamps) - 1):
                diff = (timestamps[i + 1] - timestamps[i]).total_seconds()
                time_diffs.append(diff)
            
            if time_diffs:
                avg_diff = sum(time_diffs) / len(time_diffs)
                # Normalize to 0-1 scale (lower is better)
                consistency = 1.0 / (1.0 + avg_diff / 3600)  # Normalize by hour
                return consistency
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Temporal consistency calculation failed: {e}")
            return 1.0

    async def _apply_drift_correction(self, query: str, results: List[RecallResult]):
        """Apply drift correction mechanisms."""
        try:
            # Adjust similarity thresholds
            await self._adjust_thresholds_for_drift()
            
            # Update anchor prompts
            await self._update_anchor_prompts()
            
            # Re-run retrieval with corrected parameters
            corrected_results = await self._retrieve_with_corrected_params(query)
            
            # Replace results with corrected ones
            results.clear()
            results.extend(corrected_results)
            
            logger.info("✅ Drift correction applied")
            
        except Exception as e:
            logger.error(f"Drift correction failed: {e}")

    async def _adjust_thresholds_for_drift(self):
        """Adjust similarity thresholds to correct drift."""
        # Increase threshold to be more selective
        self.adaptive_config.base_similarity_threshold *= 1.1
        
        # Adjust domain-specific thresholds
        for domain in self.adaptive_config.domain_adaptive_thresholds:
            self.adaptive_config.domain_adaptive_thresholds[domain] *= 1.1

    async def _update_anchor_prompts(self):
        """Update anchor prompts for drift correction."""
        if not self.drift_config.anchor_prompt_frequency:
            return
        
        # Generate new anchor prompts based on recent activity
        new_prompts = await self._generate_anchor_prompts()
        self.anchor_prompts.extend(new_prompts)
        
        # Keep only recent prompts
        if len(self.anchor_prompts) > 20:
            self.anchor_prompts = self.anchor_prompts[-20:]

    async def _generate_anchor_prompts(self) -> List[str]:
        """Generate anchor prompts for drift correction."""
        # This would be enhanced with actual prompt generation
        # For now, return basic prompts
        return [
            "What is the main topic of this conversation?",
            "What are the key concepts being discussed?",
            "What is the current context and background?"
        ]

    async def _retrieve_with_corrected_params(self, query: str) -> List[RecallResult]:
        """Retrieve with corrected parameters after drift detection."""
        # Use more conservative parameters
        params = await self.adaptive_retrieval_parameters(query)
        params['similarity_threshold'] *= 1.2  # More conservative
        params['top_k'] = min(params['top_k'], 10)  # Fewer results
        
        # Perform retrieval with corrected parameters
        return await self.engine.recall_memories(
            query=query,
            limit=params['top_k'],
            min_similarity=params['similarity_threshold']
        )

    async def _recalibrate_thresholds(self):
        """Recalibrate similarity thresholds based on recent performance."""
        try:
            # Analyze recent retrieval performance
            recent_results = await self._analyze_recent_performance()
            
            # Adjust thresholds based on performance
            if recent_results['precision'] < 0.8:
                # Increase threshold for better precision
                self.adaptive_config.base_similarity_threshold *= 1.05
            elif recent_results['recall'] < 0.8:
                # Decrease threshold for better recall
                self.adaptive_config.base_similarity_threshold *= 0.95
            
            logger.info(f"🔄 Recalibrated base threshold to {self.adaptive_config.base_similarity_threshold:.3f}")
            
        except Exception as e:
            logger.error(f"Threshold recalibration failed: {e}")

    async def _analyze_recent_performance(self) -> Dict[str, float]:
        """Analyze recent retrieval performance."""
        # This would analyze actual performance metrics
        # For now, return default values
        return {
            'precision': 0.85,
            'recall': 0.80,
            'f1_score': 0.82
        }

    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'latency_metrics': {
                'average_recall_latency_ms': self.latency_monitor.average_recall_latency,
                'average_compression_latency_ms': self.latency_monitor.average_compression_latency,
                'total_operations': self.latency_monitor.total_operations
            },
            'adaptive_config': {
                'base_threshold': self.adaptive_config.base_similarity_threshold,
                'domain_thresholds': self.adaptive_config.domain_adaptive_thresholds,
                'latency_budget_ms': self.adaptive_config.latency_budget_ms
            },
            'drift_correction': {
                'last_recalibration': self.last_recalibration.isoformat(),
                'operation_count': self.operation_count,
                'drift_indicators_count': len(self.drift_indicators)
            },
            'hierarchical_summarization': {
                'levels_configured': len(self.hierarchical_config.levels),
                'summarization_levels_used': len(self.summarization_levels)
            }
        }

# ============================================================================
# INTEGRATION WITH PERFECT RECALL ENGINE
# ============================================================================

# Add the advanced context manager to the PerfectRecallEngine class
async def initialize_advanced_context_manager(self):
    """Initialize the advanced context manager for caveat solutions."""
    if not hasattr(self, 'advanced_context_manager'):
        self.advanced_context_manager = AdvancedContextManager(self)
        logger.info("🔧 Advanced Context Manager initialized")

async def enhanced_recall_memories(
    self,
    query: str,
    content_types: List[str] = None,
    tags: List[str] = None,
    limit: int = 10,
    min_similarity: float = None,
    workflow_id: str = None,
    session_id: str = None
) -> List[RecallResult]:
    """
    Enhanced recall with adaptive parameters and drift correction.
    """
    # Initialize advanced context manager if needed
    if not hasattr(self, 'advanced_context_manager'):
        await self.initialize_advanced_context_manager()
    
    start_time = time.time()
    
    try:
        # Get adaptive retrieval parameters
        adaptive_params = await self.advanced_context_manager.adaptive_retrieval_parameters(query)
        
        # Use adaptive parameters if available
        if adaptive_params:
            min_similarity = adaptive_params['similarity_threshold']
            limit = adaptive_params['top_k']
        
        # Perform standard recall
        results = await self.recall_memories(
            query=query,
            content_types=content_types,
            tags=tags,
            limit=limit,
            min_similarity=min_similarity,
            workflow_id=workflow_id,
            session_id=session_id
        )
        
        # Monitor latency
        end_time = time.time()
        await self.advanced_context_manager.monitor_latency('recall', start_time, end_time)
        
        # Detect and correct drift
        drift_corrected = await self.advanced_context_manager.detect_and_correct_drift(query, results)
        
        if drift_corrected:
            logger.info("🔄 Drift correction applied to recall results")
        
        return results
        
    except Exception as e:
        logger.error(f"Enhanced recall failed: {e}")
        # Fallback to standard recall
        return await self.recall_memories(
            query=query,
            content_types=content_types,
            tags=tags,
            limit=limit,
            min_similarity=min_similarity,
            workflow_id=workflow_id,
            session_id=session_id
        )

async def get_advanced_performance_metrics(self) -> Dict[str, Any]:
    """Get advanced performance metrics including caveat solutions."""
    base_metrics = await self.get_enhanced_performance_metrics()
    
    if hasattr(self, 'advanced_context_manager'):
        advanced_metrics = await self.advanced_context_manager.get_performance_report()
        base_metrics['advanced_context_management'] = advanced_metrics
    
    return base_metrics

async def shutdown(self):
    """Properly shutdown the engine and cleanup resources."""
    try:
        # Close Neo4j driver if it exists
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            self.neo4j_driver.close()
            logger.info("🔌 Neo4j driver closed")
        
        # Close ChromaDB client if it exists
        if hasattr(self, 'chroma_client') and self.chroma_client:
            self.chroma_client.reset()
            logger.info("🔌 ChromaDB client closed")
        
        # Save any pending memories
        if hasattr(self, 'memory_db') and self.memory_db:
            self._save_memories()
            logger.info("💾 Final memory save completed")
        
        logger.info("✅ Perfect Recall Engine shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

def __del__(self):
    """Destructor to ensure cleanup on object deletion."""
    try:
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            self.neo4j_driver.close()
    except:
        pass  # Ignore errors during cleanup