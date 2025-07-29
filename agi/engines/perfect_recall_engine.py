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

# Fallback EngineOutput definition
@dataclass
class EngineOutput:
    engine_id: str
    confidence: float
    processing_time: float
    result: Any
    metadata: Dict[str, Any] = None
    reasoning_trace: List[str] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.reasoning_trace is None:
            self.reasoning_trace = []
        if self.dependencies is None:
            self.dependencies = []

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

class PerfectRecallEngine:
    """
    🧠 Perfect Recall Engine
    
    Memory management system that stores and retrieves coding interactions
    with semantic understanding and pattern matching.
    """
    
    def __init__(self, storage_path: str = "data/memory", config: Dict[str, Any] = None):
        
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
        """Generate embeddings using sentence transformers or fallback methods."""
        try:
            # Try to use sentence transformers
            from sentence_transformers import SentenceTransformer
            
            if not hasattr(self, '_embedding_model'):
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            embedding = self._embedding_model.encode(text)
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            
        except ImportError:
            logger.warning("SentenceTransformers not available, using hash-based embedding")
            return self._generate_hash_embedding(text)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return self._generate_hash_embedding(text)
    
    def _generate_hash_embedding(self, text: str) -> List[float]:
        """Generate hash-based embedding as fallback."""
        # Simple hash-based embedding
        words = text.lower().split()
        embedding = [0.0] * self.embedding_dimension
        
        for word in words:
            hash_val = hash(word) % self.embedding_dimension
            embedding[hash_val] += 1.0
        
        # Normalize
        magnitude = sum(x*x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
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
        """Store a new memory entry with semantic understanding and workflow tracking."""
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
        
        # Create memory entry
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
            session_id=session_id
        )
        
        # Store in memory database
        self.memory_db[memory_id] = entry
        
        # Store in vector database if available
        if hasattr(self, 'collection') and self.collection:
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
                        "session_id": session_id or ""
                    }],
                    ids=[memory_id]
                )
            except Exception as e:
                logger.warning(f"Failed to store in ChromaDB: {e}")
        
        # Save to persistent storage
        self._save_memories()
        
        logger.info(f"🧠 Stored memory: {content_type} - {len(content)} chars")
        return memory_id
    
    async def recall_memories(
        self,
        query: str,
        content_types: List[str] = None,
        tags: List[str] = None,
        limit: int = 10,
        min_similarity: float = None
    ) -> List[RecallResult]:
        """Recall memories based on semantic similarity and filters."""
        query_embedding = self._generate_embedding(query)
        results = []
        
        min_sim = min_similarity or self.similarity_threshold
        
        # Search through all memories
        for entry in self.memory_db.values():
            # Apply filters
            if content_types and entry.content_type not in content_types:
                continue
            
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            
            # Calculate similarity
            if entry.embedding:
                similarity = self._calculate_similarity(query_embedding, entry.embedding)
                
                if similarity >= min_sim:
                    relevance = (similarity * 0.7) + (entry.success_score * 0.2) + (entry.usage_count * 0.1)
                    
                    results.append(RecallResult(
                        entry=entry,
                        similarity_score=similarity,
                        relevance_score=relevance
                    ))
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Update usage counts
        for result in results[:limit]:
            result.entry.usage_count += 1
        
        return results[:limit]
    
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
        
        # Handle context parameter safely
        if context:
            if isinstance(context, dict):
                if context.get("type"):
                    content_types = [context.get("type")]
                limit = context.get("limit", 5)
            elif isinstance(context, str):
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
            "vector_db_available": hasattr(self, 'collection') and self.collection is not None,
            "knowledge_graph_available": hasattr(self, 'neo4j_driver') and self.neo4j_driver is not None
        }
    
    # Infinite Context Management Methods
    async def add_infinite_context(self, content: str, importance_score: float = 0.5, context_type: str = "text") -> List[str]:
        """Add content to infinite context system with semantic chunking."""
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
        """Retrieve relevant context from infinite context system."""
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
    
    def _extract_semantic_tags(self, content: str) -> List[str]:
        """Extract semantic tags from content."""
        # Simple tag extraction based on keywords
        words = content.lower().split()
        
        # Common technical terms
        tech_terms = ['function', 'class', 'method', 'variable', 'algorithm', 'data', 'code', 'program']
        tags = [word for word in words if word in tech_terms]
        
        # Add content-based tags
        if 'error' in content.lower():
            tags.append('error')
        if 'solution' in content.lower():
            tags.append('solution')
        if 'example' in content.lower():
            tags.append('example')
        
        return list(set(tags))[:10]  # Limit to 10 unique tags
    
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
    
    # Engine interface methods
    async def run(self, context, shared_state) -> EngineOutput:
        """Run the Perfect Recall Engine."""
        try:
            logger.debug('PerfectRecallEngine.run() called')
            start_time = datetime.utcnow()
            
            # Extract query from context
            query = getattr(context, 'query', None) or context.get('query', '') if hasattr(context, 'get') else ''
            
            # Perform recall
            recall_results = await self.recall_knowledge(query, context=context.__dict__ if hasattr(context, '__dict__') else context)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = {
                'recall_results': recall_results,
                'memory_count': len(self.memory_db)
            }
            
            logger.debug('PerfectRecallEngine.run() completed')
            
            # Calculate confidence based on recall quality
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
    
    def _calculate_recall_confidence(self, recall_results: List[Dict[str, Any]], query: str) -> float:
        """Calculate confidence based on recall quality and memory coverage."""
        if not recall_results:
            return 0.1
        
        # Calculate average similarity score
        similarities = [result.get('similarity_score', 0.0) for result in recall_results]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        # Consider memory coverage
        memory_coverage = min(len(recall_results) / 10, 1.0)  # Normalize by expected results
        
        # Consider query complexity
        query_complexity = min(len(query.split()) / 5, 1.0)  # Normalize by word count
        
        # Calculate confidence
        confidence = (avg_similarity * 0.6 + memory_coverage * 0.3 + query_complexity * 0.1)
        
        return min(0.95, max(0.1, confidence))
    
    async def process(self, task_type: str, input_data: Any) -> Dict[str, Any]:
        """Process a task using the Perfect Recall Engine."""
        try:
            logger.debug(f'PerfectRecallEngine.process() called with task_type: {task_type}')
            start_time = datetime.utcnow()
            
            # Extract query from input_data
            query = ""
            context = {}
            
            if isinstance(input_data, dict):
                query = input_data.get('query', '')
                context = input_data.get('context', {})
            elif isinstance(input_data, str):
                query = input_data
                context = {}
            elif hasattr(input_data, 'query'):
                query = getattr(input_data, 'query', '')
                context = getattr(input_data, 'context', {})
            else:
                query = str(input_data) if input_data is not None else ""
                context = {}
            
            # Process based on task type
            if task_type == "recall":
                result = await self.recall_knowledge(query, context)
            elif task_type == "store":
                result = await self.store_knowledge(query, context)
            elif task_type == "search":
                result = await self.recall_knowledge(query, context)
            else:
                result = await self.recall_knowledge(query, context)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Calculate confidence
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
    
    async def shutdown(self):
        """Properly shutdown the engine and cleanup resources."""
        try:
            # Close Neo4j driver if it exists
            if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
                self.neo4j_driver.close()
                logger.info("🔌 Neo4j driver closed")
            
            # Close ChromaDB client if it exists
            if hasattr(self, 'chroma_client') and self.chroma_client:
                # ChromaDB doesn't have a close method, but we can reset
                logger.info("🔌 ChromaDB client cleaned up")
            
            # Save any pending memories
            if hasattr(self, 'memory_db') and self.memory_db:
                self._save_memories()
                logger.info("💾 Final memory save completed")
            
            logger.info("✅ Perfect Recall Engine shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")