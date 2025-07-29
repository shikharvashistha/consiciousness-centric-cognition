"""
AGI Consciousness Centric Cognition System

A revolutionary AGI system built on Consciousness-Centric Cognition principles,
designed as a symbiotic ecosystem of specialized core engines orchestrated by
a central executive mind.

Version: 2.0
Architecture: Consciousness-Centric Cognition
"""

__version__ = "2.0.0"
__author__ = "Shikhar Vashistha"
__description__ = "AGI Consciousness Centric Cognition System"

from .core.agi_orchestrator import AGICoreOrchestrator
from .core.consciousness_core import EnhancedConsciousnessCore
from .core.neural_substrate import NeuralSubstrate

from .engines.perfect_recall_engine import PerfectRecallEngine
from .engines.creative_engine import AdvancedCreativeEngine
from .engines.parallel_mind_engine import ParallelMindEngine
from .engines.code_introspection_engine import CodeIntrospectionEngine
from .engines.adaptation_engine import AdaptationEngine

from .governance.ethical_governor import EthicalGovernor

# Main AGI System Class
class AGI:
    """
    🧠 AGI Consciousness System
    
    Main interface to the complete consciousness-centric AGI system.
    Orchestrates all components through the cognitive cycle.
    """
    
    def __init__(self, config=None):
        """Initialize the complete AGI system."""
        self.config = config or {}
        
        # Initialize the core orchestrator which manages all components
        self.orchestrator = AGICoreOrchestrator(self.config)
        
        # Expose individual components for direct access
        self.consciousness_core = self.orchestrator.consciousness_core
        self.neural_substrate = self.orchestrator.neural_substrate
        self.perfect_recall_engine = self.orchestrator.perfect_recall
        self.creative_engine = self.orchestrator.creative_engine
        self.parallel_mind_engine = self.orchestrator.parallel_mind_engine
        self.code_introspection_engine = self.orchestrator.code_introspection_engine
        self.adaptation_engine = self.orchestrator.adaptation_engine
        self.ethical_governor = self.orchestrator.ethical_governor
    
    async def process_goal(self, goal, context=None):
        """
        Process a goal through the complete cognitive cycle.
        
        Args:
            goal: The goal or task to process
            context: Optional context information
            
        Returns:
            Complete cognitive cycle results
        """
        user_context = context or {}
        return await self.orchestrator.execute_cognitive_cycle(goal, user_context)
    
    async def shutdown(self):
        """Shutdown the complete AGI system."""
        await self.orchestrator.shutdown()

__all__ = [
    "AGI",
    "AGICoreOrchestrator",
    "EnhancedConsciousnessCore", 
    "NeuralSubstrate",
    "PerfectRecallEngine",
    "AdvancedCreativeEngine",
    "ParallelMindEngine",
    "CodeIntrospectionEngine",
    "AdaptationEngine",
    "EthicalGovernor"
]