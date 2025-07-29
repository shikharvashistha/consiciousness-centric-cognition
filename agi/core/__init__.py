"""
AGI Core Components

This module contains the core components of the AGI system:
- Enhanced Consciousness Core: Scientific IIT implementation
- Neural Substrate: Real neural network operations
- AGI Orchestrator: Executive mind coordination
- Enhanced AGI System: Improved system with dynamic consciousness
"""

from .agi_orchestrator import AGICoreOrchestrator
from .consciousness_core import EnhancedConsciousnessCore
from .neural_substrate import NeuralSubstrate

# Import new enhanced components
try:
    from .enhanced_consciousness_core import EnhancedConsciousnessCore as NewEnhancedConsciousnessCore
    from .enhanced_agi_system import EnhancedAGISystem
    __all__ = [
        "AGICoreOrchestrator",
        "EnhancedConsciousnessCore",
        "NeuralSubstrate",
        "EnhancedAGISystem"
    ]
except ImportError:
    __all__ = [
        "AGICoreOrchestrator",
        "EnhancedConsciousnessCore",
        "NeuralSubstrate"
    ]