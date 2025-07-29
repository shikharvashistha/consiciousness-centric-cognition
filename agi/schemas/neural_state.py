"""
Neural state schemas and data models

Standardized data interfaces for neural processing to ensure consistent data format
between components and prevent handoff errors.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import numpy as np
import torch

@dataclass
class NeuralStateData:
    """
    Standardized neural state data representation
    
    This class provides a consistent interface for neural data exchange between
    components, ensuring compatibility and preventing format mismatches.
    """
    # Core neural data (required)
    neural_data: np.ndarray  # 2D array [n_nodes, n_timesteps]
    semantic_features: Dict[str, float]
    confidence: float
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    source_component: str = "neural_substrate"
    processing_time_ms: Optional[float] = None
    
    # Optional tensor data (will be converted to numpy for compatibility)
    tensor_data: Optional[Dict[str, Any]] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize neural data"""
        # Ensure neural_data is a 2D numpy array
        if isinstance(self.neural_data, torch.Tensor):
            self.neural_data = self.neural_data.detach().cpu().numpy()
        
        # Ensure neural_data is 2D
        if not isinstance(self.neural_data, np.ndarray):
            raise ValueError("neural_data must be a numpy array")
        
        if self.neural_data.ndim != 2:
            raise ValueError(f"neural_data must be 2D, got shape {self.neural_data.shape}")
        
        # Convert any tensor data to numpy
        if self.tensor_data:
            for key, value in self.tensor_data.items():
                if isinstance(value, torch.Tensor):
                    self.tensor_data[key] = value.detach().cpu().numpy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            'neural_data': self.neural_data.tolist(),
            'semantic_features': self.semantic_features,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'source_component': self.source_component,
            'processing_time_ms': self.processing_time_ms,
            'metadata': self.metadata
        }
        
        if self.tensor_data:
            result['tensor_data'] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.tensor_data.items()
            }
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuralStateData':
        """Create from dictionary"""
        neural_data = np.array(data['neural_data'])
        semantic_features = data['semantic_features']
        confidence = data['confidence']
        
        # Parse timestamp
        timestamp = datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now()
        
        # Create instance
        instance = cls(
            neural_data=neural_data,
            semantic_features=semantic_features,
            confidence=confidence,
            timestamp=timestamp,
            source_component=data.get('source_component', 'unknown'),
            processing_time_ms=data.get('processing_time_ms'),
            metadata=data.get('metadata', {})
        )
        
        # Add tensor data if present
        if 'tensor_data' in data:
            instance.tensor_data = {
                k: np.array(v) if isinstance(v, list) else v
                for k, v in data['tensor_data'].items()
            }
        
        return instance
    
    def get_shape(self) -> tuple:
        """Get shape of neural data"""
        return self.neural_data.shape
    
    def get_dimensions(self) -> Dict[str, int]:
        """Get dimensions of neural data"""
        n_nodes, n_timesteps = self.neural_data.shape
        return {
            'n_nodes': n_nodes,
            'n_timesteps': n_timesteps
        }
    
    def is_compatible_with(self, other: 'NeuralStateData') -> bool:
        """Check if compatible with another neural state"""
        return self.neural_data.shape == other.neural_data.shape

@dataclass
class CreativePlan:
    """
    Standardized creative plan representation
    
    This class provides a consistent interface for creative plans generated
    by the Creative Engine and used by other components.
    """
    # Core plan data
    solution_plan: str
    creativity_score: float
    innovation_level: float
    approach_type: str
    confidence: float
    
    # Additional data
    implementation_steps: List[str] = field(default_factory=list)
    resources_required: List[str] = field(default_factory=list)
    estimated_completion_time: Optional[float] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate creative plan data"""
        for score in [self.creativity_score, self.innovation_level, self.confidence]:
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Score {score} must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'solution_plan': self.solution_plan,
            'creativity_score': self.creativity_score,
            'innovation_level': self.innovation_level,
            'approach_type': self.approach_type,
            'confidence': self.confidence,
            'implementation_steps': self.implementation_steps,
            'resources_required': self.resources_required,
            'estimated_completion_time': self.estimated_completion_time,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CreativePlan':
        """Create from dictionary"""
        # Parse timestamp
        timestamp = datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now()
        
        return cls(
            solution_plan=data['solution_plan'],
            creativity_score=data['creativity_score'],
            innovation_level=data['innovation_level'],
            approach_type=data['approach_type'],
            confidence=data['confidence'],
            implementation_steps=data.get('implementation_steps', []),
            resources_required=data.get('resources_required', []),
            estimated_completion_time=data.get('estimated_completion_time'),
            timestamp=timestamp,
            metadata=data.get('metadata', {})
        )

@dataclass
class EthicalReview:
    """
    Standardized ethical review representation
    
    This class provides a consistent interface for ethical reviews generated
    by the Ethical Governor and used by other components.
    """
    # Core review data
    approved: bool
    ethical_score: float
    risk_level: str
    concerns: List[str]
    recommendations: List[str]
    
    # Additional data
    framework_scores: Dict[str, float] = field(default_factory=dict)
    principle_scores: Dict[str, float] = field(default_factory=dict)
    bias_detected: bool = False
    bias_types: List[str] = field(default_factory=list)
    bias_severity: float = 0.0
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate ethical review data"""
        if not 0.0 <= self.ethical_score <= 1.0:
            raise ValueError(f"Ethical score {self.ethical_score} must be between 0.0 and 1.0")
        
        if not 0.0 <= self.bias_severity <= 1.0:
            raise ValueError(f"Bias severity {self.bias_severity} must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'approved': self.approved,
            'ethical_score': self.ethical_score,
            'risk_level': self.risk_level,
            'concerns': self.concerns,
            'recommendations': self.recommendations,
            'framework_scores': self.framework_scores,
            'principle_scores': self.principle_scores,
            'bias_detected': self.bias_detected,
            'bias_types': self.bias_types,
            'bias_severity': self.bias_severity,
            'timestamp': self.timestamp.isoformat(),
            'processing_time_ms': self.processing_time_ms,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EthicalReview':
        """Create from dictionary"""
        # Parse timestamp
        timestamp = datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now()
        
        return cls(
            approved=data['approved'],
            ethical_score=data['ethical_score'],
            risk_level=data['risk_level'],
            concerns=data['concerns'],
            recommendations=data['recommendations'],
            framework_scores=data.get('framework_scores', {}),
            principle_scores=data.get('principle_scores', {}),
            bias_detected=data.get('bias_detected', False),
            bias_types=data.get('bias_types', []),
            bias_severity=data.get('bias_severity', 0.0),
            timestamp=timestamp,
            processing_time_ms=data.get('processing_time_ms'),
            metadata=data.get('metadata', {})
        )