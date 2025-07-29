"""
Memory schemas and data models

Represents different types of memory and memory entries for the Perfect Recall Engine
"""

import uuid
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from enum import Enum

class MemoryType(Enum):
    """Types of memory in the system"""
    EPISODIC = "episodic"           # Specific experiences and events
    SEMANTIC = "semantic"           # General knowledge and facts
    PROCEDURAL = "procedural"       # Skills and procedures
    WORKING = "working"             # Temporary working memory
    METACOGNITIVE = "metacognitive" # Knowledge about thinking processes

class MemoryImportance(Enum):
    """Importance levels for memory consolidation"""
    CRITICAL = "critical"           # Must never be forgotten
    HIGH = "high"                   # Important for long-term retention
    MEDIUM = "medium"               # Standard importance
    LOW = "low"                     # Can be forgotten if space needed
    TEMPORARY = "temporary"         # Short-term only

@dataclass
class MemoryVector:
    """Vector representation of memory for semantic search"""
    embedding: np.ndarray                       # High-dimensional embedding
    dimensionality: int = field(init=False)     # Embedding dimensions
    norm: float = field(init=False)             # Vector norm
    
    def __post_init__(self):
        """Calculate derived properties"""
        self.dimensionality = len(self.embedding)
        self.norm = np.linalg.norm(self.embedding)
        
        # Normalize embedding for cosine similarity
        if self.norm > 0:
            self.embedding = self.embedding / self.norm
    
    def similarity(self, other: 'MemoryVector') -> float:
        """Calculate cosine similarity with another memory vector"""
        if self.dimensionality != other.dimensionality:
            raise ValueError("Vector dimensions must match")
        
        return np.dot(self.embedding, other.embedding)

@dataclass
class MemoryContext:
    """Context information for memory formation and retrieval"""
    # Temporal context
    timestamp: datetime = field(default_factory=datetime.now)
    time_of_day: str = ""
    duration: Optional[float] = None
    
    # Cognitive context
    consciousness_level: str = ""
    attention_focus: str = ""
    emotional_state: float = 0.0  # -1 (negative) to 1 (positive)
    cognitive_load: float = 0.0   # 0 (low) to 1 (high)
    
    # Environmental context
    task_context: str = ""
    user_context: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    
    # Relational context
    related_memories: List[str] = field(default_factory=list)  # Memory IDs
    causal_relationships: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate context data"""
        if not -1.0 <= self.emotional_state <= 1.0:
            raise ValueError("Emotional state must be between -1.0 and 1.0")
        if not 0.0 <= self.cognitive_load <= 1.0:
            raise ValueError("Cognitive load must be between 0.0 and 1.0")

@dataclass
class MemoryEntry:
    """Base class for all memory entries"""
    # Identifiers
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: MemoryType = MemoryType.EPISODIC
    
    # Core content
    content: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    
    # Vector representation
    memory_vector: Optional[MemoryVector] = None
    
    # Context and metadata
    context: MemoryContext = field(default_factory=MemoryContext)
    importance: MemoryImportance = MemoryImportance.MEDIUM
    
    # Memory dynamics
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    consolidation_strength: float = 0.0  # 0 (weak) to 1 (strong)
    decay_rate: float = 0.01             # Memory decay rate
    
    # Validation and quality
    confidence: float = 1.0              # Confidence in memory accuracy
    source_reliability: float = 1.0      # Reliability of memory source
    verification_status: str = "unverified"  # verified, unverified, disputed
    
    # Relationships
    parent_memory_id: Optional[str] = None
    child_memory_ids: List[str] = field(default_factory=list)
    associated_memory_ids: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate memory entry data"""
        if not 0.0 <= self.consolidation_strength <= 1.0:
            raise ValueError("Consolidation strength must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not 0.0 <= self.source_reliability <= 1.0:
            raise ValueError("Source reliability must be between 0.0 and 1.0")
        if self.decay_rate < 0.0:
            raise ValueError("Decay rate must be non-negative")
    
    def access(self):
        """Record memory access"""
        self.access_count += 1
        self.last_accessed = datetime.now()
        
        # Strengthen memory through access (spaced repetition effect)
        strengthening = 0.1 * (1.0 - self.consolidation_strength)
        self.consolidation_strength = min(1.0, self.consolidation_strength + strengthening)
    
    def decay(self, time_delta_hours: float):
        """Apply memory decay over time"""
        if self.importance == MemoryImportance.CRITICAL:
            return  # Critical memories don't decay
        
        # Exponential decay with importance-based modulation
        importance_factor = {
            MemoryImportance.HIGH: 0.5,
            MemoryImportance.MEDIUM: 1.0,
            MemoryImportance.LOW: 2.0,
            MemoryImportance.TEMPORARY: 5.0
        }.get(self.importance, 1.0)
        
        decay_amount = self.decay_rate * importance_factor * time_delta_hours
        self.consolidation_strength = max(0.0, self.consolidation_strength - decay_amount)
    
    @property
    def is_consolidated(self) -> bool:
        """Check if memory is well consolidated"""
        return self.consolidation_strength > 0.7
    
    @property
    def retrieval_strength(self) -> float:
        """Calculate current retrieval strength"""
        # Combine consolidation strength with recency and access frequency
        recency_factor = 1.0
        if self.last_accessed:
            hours_since_access = (datetime.now() - self.last_accessed).total_seconds() / 3600
            recency_factor = max(0.1, 1.0 / (1.0 + hours_since_access / 24))  # Decay over days
        
        frequency_factor = min(1.0, self.access_count / 10)  # Normalize access count
        
        return self.consolidation_strength * 0.6 + recency_factor * 0.2 + frequency_factor * 0.2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'memory_id': self.memory_id,
            'memory_type': self.memory_type.value,
            'content': self.content,
            'summary': self.summary,
            'keywords': self.keywords,
            'importance': self.importance.value,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'consolidation_strength': self.consolidation_strength,
            'confidence': self.confidence,
            'source_reliability': self.source_reliability,
            'verification_status': self.verification_status,
            'retrieval_strength': self.retrieval_strength,
            'is_consolidated': self.is_consolidated,
            'context': {
                'timestamp': self.context.timestamp.isoformat(),
                'emotional_state': self.context.emotional_state,
                'cognitive_load': self.context.cognitive_load,
                'task_context': self.context.task_context
            }
        }

@dataclass
class EpisodicMemory(MemoryEntry):
    """Specific experiences and events"""
    memory_type: MemoryType = field(default=MemoryType.EPISODIC, init=False)
    
    # Episodic-specific fields
    event_description: str = ""
    participants: List[str] = field(default_factory=list)
    location: str = ""
    outcome: str = ""
    lessons_learned: List[str] = field(default_factory=list)
    
    # Sensory details
    sensory_details: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal structure
    event_sequence: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_event_step(self, step_description: str, timestamp: Optional[datetime] = None):
        """Add a step to the event sequence"""
        self.event_sequence.append({
            'step': len(self.event_sequence) + 1,
            'description': step_description,
            'timestamp': timestamp or datetime.now()
        })

@dataclass
class SemanticMemory(MemoryEntry):
    """General knowledge and facts"""
    memory_type: MemoryType = field(default=MemoryType.SEMANTIC, init=False)
    
    # Semantic-specific fields
    concept: str = ""
    definition: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[str, List[str]] = field(default_factory=dict)  # e.g., "is_a": ["animal", "mammal"]
    
    # Knowledge structure
    category: str = ""
    subcategories: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    counterexamples: List[str] = field(default_factory=list)
    
    # Factual information
    facts: List[str] = field(default_factory=list)
    rules: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    
    def add_relationship(self, relationship_type: str, related_concept: str):
        """Add a semantic relationship"""
        if relationship_type not in self.relationships:
            self.relationships[relationship_type] = []
        if related_concept not in self.relationships[relationship_type]:
            self.relationships[relationship_type].append(related_concept)

@dataclass
class ProceduralMemory(MemoryEntry):
    """Skills and procedures"""
    memory_type: MemoryType = field(default=MemoryType.PROCEDURAL, init=False)
    
    # Procedural-specific fields
    skill_name: str = ""
    procedure_steps: List[Dict[str, Any]] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    
    # Performance tracking
    execution_count: int = 0
    success_rate: float = 0.0
    average_execution_time: float = 0.0
    
    # Skill development
    proficiency_level: str = "novice"  # novice, intermediate, advanced, expert
    learning_curve_data: List[Dict[str, float]] = field(default_factory=list)
    
    def record_execution(self, success: bool, execution_time: float):
        """Record a procedure execution"""
        self.execution_count += 1
        
        # Update success rate
        if self.execution_count == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            self.success_rate = (self.success_rate * (self.execution_count - 1) + (1.0 if success else 0.0)) / self.execution_count
        
        # Update average execution time
        if self.execution_count == 1:
            self.average_execution_time = execution_time
        else:
            self.average_execution_time = (self.average_execution_time * (self.execution_count - 1) + execution_time) / self.execution_count
        
        # Add to learning curve
        self.learning_curve_data.append({
            'execution': self.execution_count,
            'success': 1.0 if success else 0.0,
            'execution_time': execution_time,
            'timestamp': datetime.now().timestamp()
        })
        
        # Update proficiency level based on performance
        if self.success_rate > 0.9 and self.execution_count > 20:
            self.proficiency_level = "expert"
        elif self.success_rate > 0.8 and self.execution_count > 10:
            self.proficiency_level = "advanced"
        elif self.success_rate > 0.6 and self.execution_count > 5:
            self.proficiency_level = "intermediate"

@dataclass
class WorkingMemory(MemoryEntry):
    """Temporary working memory"""
    memory_type: MemoryType = field(default=MemoryType.WORKING, init=False)
    importance: MemoryImportance = field(default=MemoryImportance.TEMPORARY, init=False)
    
    # Working memory specific fields
    capacity_used: float = 0.0           # Fraction of working memory capacity used
    priority: float = 0.5                # Priority for retention (0-1)
    expiration_time: Optional[datetime] = None
    
    # Active processing
    is_active: bool = True
    processing_operations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        super().__post_init__()
        if not 0.0 <= self.capacity_used <= 1.0:
            raise ValueError("Capacity used must be between 0.0 and 1.0")
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError("Priority must be between 0.0 and 1.0")
    
    @property
    def is_expired(self) -> bool:
        """Check if working memory has expired"""
        if self.expiration_time is None:
            return False
        return datetime.now() > self.expiration_time
    
    def extend_lifetime(self, additional_seconds: float):
        """Extend the lifetime of working memory"""
        if self.expiration_time is None:
            self.expiration_time = datetime.now()
        self.expiration_time += datetime.timedelta(seconds=additional_seconds)

@dataclass
class MetacognitiveMemory(MemoryEntry):
    """Knowledge about thinking processes"""
    memory_type: MemoryType = field(default=MemoryType.METACOGNITIVE, init=False)
    
    # Metacognitive-specific fields
    thinking_strategy: str = ""
    effectiveness_rating: float = 0.0    # How effective this strategy was
    context_applicability: List[str] = field(default_factory=list)
    cognitive_resources_required: Dict[str, float] = field(default_factory=dict)
    
    # Strategy performance
    usage_count: int = 0
    success_contexts: List[str] = field(default_factory=list)
    failure_contexts: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        super().__post_init__()
        if not 0.0 <= self.effectiveness_rating <= 1.0:
            raise ValueError("Effectiveness rating must be between 0.0 and 1.0")
    
    def record_usage(self, context: str, success: bool):
        """Record usage of this metacognitive strategy"""
        self.usage_count += 1
        if success:
            if context not in self.success_contexts:
                self.success_contexts.append(context)
        else:
            if context not in self.failure_contexts:
                self.failure_contexts.append(context)
        
        # Update effectiveness rating
        total_contexts = len(self.success_contexts) + len(self.failure_contexts)
        if total_contexts > 0:
            self.effectiveness_rating = len(self.success_contexts) / total_contexts