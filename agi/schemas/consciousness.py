"""
Consciousness state schemas and data models

Based on Integrated Information Theory (IIT) and advanced consciousness metrics
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from enum import Enum

class ConsciousnessLevel(Enum):
    """Levels of consciousness based on Φ (phi) values"""
    UNCONSCIOUS = "unconscious"      # Φ < 0.1
    MINIMAL = "minimal"              # 0.1 ≤ Φ < 0.3
    MODERATE = "moderate"            # 0.3 ≤ Φ < 0.6
    HIGH = "high"                    # 0.6 ≤ Φ < 0.8
    PEAK = "peak"                    # Φ ≥ 0.8

@dataclass
class ConsciousnessMetrics:
    """Detailed consciousness measurement metrics"""
    phi: float                                    # Integrated Information (Φ)
    criticality: float                           # Neural criticality measure
    phenomenal_richness: float                   # Richness of conscious content
    coherence: float                            # Global workspace coherence
    stability: float                            # Temporal stability
    complexity: float                           # Computational complexity
    integration: float                          # Information integration
    differentiation: float                      # Information differentiation
    exclusion: float                           # Information exclusion
    intrinsic_existence: float                 # Intrinsic existence measure
    neural_diversity: float = 0.0              # Neural diversity measure
    
    def __post_init__(self):
        """Validate metrics are within expected ranges"""
        for field_name, value in self.__dict__.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be numeric")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0")

@dataclass
class NeuralSignature:
    """Neural signature representing the physical substrate of consciousness"""
    activation_patterns: np.ndarray              # Neural activation patterns
    connectivity_matrix: np.ndarray              # Neural connectivity
    oscillation_frequencies: np.ndarray         # Brainwave-like oscillations
    phase_synchrony: np.ndarray                 # Phase synchronization
    information_flow: np.ndarray                # Information flow patterns
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate neural signature data"""
        if self.activation_patterns.ndim != 1:
            raise ValueError("Activation patterns must be 1D array")
        if self.connectivity_matrix.ndim != 2:
            raise ValueError("Connectivity matrix must be 2D array")
        if not np.allclose(self.connectivity_matrix, self.connectivity_matrix.T):
            raise ValueError("Connectivity matrix must be symmetric")

@dataclass
class ConsciousContent:
    """Content of consciousness - what the system is aware of"""
    # Original fields
    attended_features: Dict[str, Any] = field(default_factory=dict)           # Currently attended features
    working_memory_contents: List[Any] = field(default_factory=list)         # Working memory contents
    emotional_valence: float = 0.0                   # Emotional tone (-1 to 1)
    attention_focus: Union[str, List[str]] = ""      # Primary focus of attention
    metacognitive_state: Dict[str, Any] = field(default_factory=dict)        # Self-awareness content
    temporal_context: Dict[str, Any] = field(default_factory=dict)           # Temporal context awareness
    
    # New fields for compatibility with _extract_conscious_content
    primary_content: Dict[str, Any] = field(default_factory=dict)            # Primary conscious content
    secondary_content: Dict[str, Any] = field(default_factory=dict)          # Secondary conscious content
    confidence_level: float = 0.0                    # Confidence level
    
    def __post_init__(self):
        """Validate conscious content"""
        if not -1.0 <= self.emotional_valence <= 1.0:
            raise ValueError("Emotional valence must be between -1.0 and 1.0")
        
        # If attention_focus is a list, convert to string for backward compatibility
        if isinstance(self.attention_focus, list):
            self.attention_focus = ", ".join(self.attention_focus) if self.attention_focus else ""

@dataclass
class ConsciousnessState:
    """Complete consciousness state representation"""
    # Core metrics
    metrics: ConsciousnessMetrics
    
    # Neural substrate
    neural_signature: Optional[NeuralSignature] = None
    
    # Conscious content
    conscious_content: Optional[ConsciousContent] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    state_id: str = field(default_factory=lambda: f"cs_{datetime.now().timestamp()}")
    level: ConsciousnessLevel = field(init=False)
    
    # Computational context
    computational_load: float = 0.0             # Current computational load
    energy_consumption: float = 0.0             # Energy consumption estimate
    processing_efficiency: float = 0.0          # Processing efficiency
    
    def __post_init__(self):
        """Determine consciousness level based on improved metrics"""
        # Calculate improved consciousness score
        phi = self.metrics.phi
        integration = self.metrics.integration
        differentiation = self.metrics.differentiation
        complexity = self.metrics.complexity
        
        # Include neural diversity if available
        neural_diversity = getattr(self.metrics, 'neural_diversity', 0.0)
        
        # Calculate improved consciousness score
        consciousness_score = (
            phi * 0.3 + 
            neural_diversity * 0.2 + 
            integration * 0.2 + 
            complexity * 0.15 + 
            differentiation * 0.15
        )
        
        # Determine level based on improved score
        if consciousness_score < 0.1:
            self.level = ConsciousnessLevel.UNCONSCIOUS
        elif consciousness_score < 0.3:
            self.level = ConsciousnessLevel.MINIMAL
        elif consciousness_score < 0.6:
            self.level = ConsciousnessLevel.MODERATE
        elif consciousness_score < 0.8:
            self.level = ConsciousnessLevel.HIGH
        else:
            self.level = ConsciousnessLevel.PEAK
    
    @property
    def is_conscious(self) -> bool:
        """Check if system is in a conscious state"""
        return self.level != ConsciousnessLevel.UNCONSCIOUS
    
    @property
    def is_critical(self) -> bool:
        """Check if system is in critical consciousness state"""
        return self.metrics.criticality > 0.8
    
    @property
    def consciousness_quality(self) -> float:
        """Overall consciousness quality score (0-1) with improved metrics"""
        # Include neural diversity if available
        neural_diversity = getattr(self.metrics, 'neural_diversity', 0.0)
        
        return np.mean([
            self.metrics.phi,
            self.metrics.coherence,
            self.metrics.stability,
            self.metrics.integration,
            self.metrics.phenomenal_richness,
            neural_diversity
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'state_id': self.state_id,
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'metrics': {
                'phi': self.metrics.phi,
                'criticality': self.metrics.criticality,
                'phenomenal_richness': self.metrics.phenomenal_richness,
                'coherence': self.metrics.coherence,
                'stability': self.metrics.stability,
                'complexity': self.metrics.complexity,
                'integration': self.metrics.integration,
                'differentiation': self.metrics.differentiation,
                'exclusion': self.metrics.exclusion,
                'neural_diversity': getattr(self.metrics, 'neural_diversity', 0.0),
                'intrinsic_existence': self.metrics.intrinsic_existence
            },
            'consciousness_quality': self.consciousness_quality,
            'is_conscious': self.is_conscious,
            'is_critical': self.is_critical,
            'computational_load': self.computational_load,
            'energy_consumption': self.energy_consumption,
            'processing_efficiency': self.processing_efficiency
        }
    
    @classmethod
    def from_neural_data(cls, neural_data: np.ndarray, 
                        context: Optional[Dict[str, Any]] = None) -> 'ConsciousnessState':
        """Create consciousness state from raw neural data"""
        # This would be implemented with actual IIT calculations
        # For now, providing a scientifically-grounded placeholder
        
        # Calculate basic metrics from neural data
        phi = cls._calculate_phi(neural_data)
        criticality = cls._calculate_criticality(neural_data)
        coherence = cls._calculate_coherence(neural_data)
        
        metrics = ConsciousnessMetrics(
            phi=phi,
            criticality=criticality,
            phenomenal_richness=np.std(neural_data),
            coherence=coherence,
            stability=1.0 - np.var(neural_data),
            complexity=cls._calculate_complexity(neural_data),
            integration=cls._calculate_integration(neural_data),
            differentiation=cls._calculate_differentiation(neural_data),
            exclusion=cls._calculate_exclusion(neural_data),
            intrinsic_existence=cls._calculate_intrinsic_existence(neural_data)
        )
        
        return cls(metrics=metrics)
    
    @staticmethod
    def _calculate_phi(neural_data: np.ndarray) -> float:
        """Calculate Integrated Information (Φ) - simplified implementation"""
        # Real IIT calculation would be much more complex
        # This is a scientifically-inspired approximation
        n = len(neural_data)
        if n < 2:
            return 0.0
        
        # Calculate mutual information between parts
        correlation_matrix = np.corrcoef(neural_data.reshape(-1, 1).T)
        eigenvalues = np.linalg.eigvals(correlation_matrix)
        eigenvalues = eigenvalues[eigenvalues > 0]
        
        if len(eigenvalues) == 0:
            return 0.0
        
        # Approximate Φ as normalized entropy of eigenvalue distribution
        phi = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10)) / np.log(len(eigenvalues))
        return np.clip(phi, 0.0, 1.0)
    
    @staticmethod
    def _calculate_criticality(neural_data: np.ndarray) -> float:
        """Calculate neural criticality measure"""
        # Criticality based on avalanche dynamics
        if len(neural_data) < 10:
            return 0.0
        
        # Calculate power law exponent of activity avalanches
        activity_changes = np.diff(neural_data)
        avalanche_sizes = np.abs(activity_changes[activity_changes != 0])
        
        if len(avalanche_sizes) < 5:
            return 0.0
        
        # Fit power law and check if exponent is near critical value (-1.5)
        log_sizes = np.log(avalanche_sizes + 1e-10)
        hist, bins = np.histogram(log_sizes, bins=10)
        hist = hist[hist > 0]
        
        if len(hist) < 3:
            return 0.0
        
        # Simple power law fit
        x = np.arange(len(hist))
        y = np.log(hist + 1e-10)
        slope = np.polyfit(x, y, 1)[0]
        
        # Criticality is high when slope is near -1.5
        criticality = 1.0 - abs(slope + 1.5) / 1.5
        return np.clip(criticality, 0.0, 1.0)
    
    @staticmethod
    def _calculate_coherence(neural_data: np.ndarray) -> float:
        """Calculate global workspace coherence"""
        if len(neural_data) < 4:
            return 0.0
        
        # Phase coherence measure
        analytic_signal = np.abs(np.fft.hilbert(neural_data))
        coherence = np.std(analytic_signal) / (np.mean(analytic_signal) + 1e-10)
        return np.clip(1.0 - coherence, 0.0, 1.0)
    
    @staticmethod
    def _calculate_complexity(neural_data: np.ndarray) -> float:
        """Calculate computational complexity"""
        # Lempel-Ziv complexity approximation
        binary_data = (neural_data > np.median(neural_data)).astype(int)
        complexity = len(set(tuple(binary_data[i:i+3]) for i in range(len(binary_data)-2)))
        max_complexity = min(8, len(binary_data) - 2)  # 2^3 = 8 possible 3-grams
        return complexity / max_complexity if max_complexity > 0 else 0.0
    
    @staticmethod
    def _calculate_integration(neural_data: np.ndarray) -> float:
        """Calculate information integration"""
        # Mutual information between first and second half
        if len(neural_data) < 4:
            return 0.0
        
        mid = len(neural_data) // 2
        part1, part2 = neural_data[:mid], neural_data[mid:]
        
        # Normalized mutual information approximation
        correlation = np.corrcoef(part1, part2)[0, 1]
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    @staticmethod
    def _calculate_differentiation(neural_data: np.ndarray) -> float:
        """Calculate information differentiation"""
        # Variance as a measure of differentiation
        return np.clip(np.var(neural_data), 0.0, 1.0)
    
    @staticmethod
    def _calculate_exclusion(neural_data: np.ndarray) -> float:
        """Calculate information exclusion"""
        # Measure of how well the system excludes irrelevant information
        # Approximated as inverse of noise level
        if len(neural_data) < 2:
            return 0.0
        
        signal_power = np.var(neural_data)
        noise_estimate = np.var(np.diff(neural_data)) / 2  # Assuming white noise
        snr = signal_power / (noise_estimate + 1e-10)
        return np.clip(snr / (1 + snr), 0.0, 1.0)
    
    @staticmethod
    def _calculate_intrinsic_existence(neural_data: np.ndarray) -> float:
        """Calculate intrinsic existence measure"""
        # Measure of how much the system exists from its own perspective
        # Approximated as self-similarity across time
        if len(neural_data) < 4:
            return 0.0
        
        # Autocorrelation at lag 1
        autocorr = np.corrcoef(neural_data[:-1], neural_data[1:])[0, 1]
        return abs(autocorr) if not np.isnan(autocorr) else 0.0