"""
AGI Engines

This module contains the specialized engines of the AGI system:
- Advanced Creative Engine: Scientific creative reasoning
- Parallel Mind Engine: Real parallel processing
- Perfect Recall Engine: Memory management
- Code Introspection Engine: Code analysis
- Adaptation Engine: System adaptation
"""

from .creative_engine import AdvancedCreativeEngine
from .parallel_mind_engine import ParallelMindEngine
from .perfect_recall_engine import PerfectRecallEngine
from .code_introspection_engine import CodeIntrospectionEngine
from .adaptation_engine import AdaptationEngine

__all__ = [
    "AdvancedCreativeEngine",
    "ParallelMindEngine", 
    "PerfectRecallEngine",
    "CodeIntrospectionEngine",
    "AdaptationEngine"
]