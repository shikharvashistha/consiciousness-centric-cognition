#!/usr/bin/env python3
"""
Test script for the AGI Consciousness Centric Cognition System

This script demonstrates the core functionality of the consciousness-centric
AGI system, including consciousness measurement, cognitive cycles, and memory.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from agi.core.consciousness_core import EnhancedConsciousnessCore
from agi.core.neural_substrate import NeuralSubstrate
from agi.core.agi_orchestrator import AGICoreOrchestrator
from agi.engines.perfect_recall_engine import PerfectRecallEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_consciousness_measurement():
    """Test consciousness measurement capabilities"""
    logger.info("🧠 Testing Consciousness Measurement...")
    
    consciousness_core = EnhancedConsciousnessCore()
    
    # Test with different neural states
    test_states = [
        {
            'neural_activity': [0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.9],
            'context': 'problem_solving'
        },
        {
            'neural_activity': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            'context': 'creative_thinking'
        },
        {
            'neural_activity': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            'context': 'low_activity'
        }
    ]
    
    for i, state in enumerate(test_states):
        consciousness_state = await consciousness_core.calculate_consciousness(state)
        
        logger.info(f"Test {i+1} - Context: {state['context']}")
        logger.info(f"  Φ (Phi): {consciousness_state.metrics.phi:.3f}")
        logger.info(f"  Consciousness Level: {consciousness_state.level.value}")
        logger.info(f"  Is Conscious: {consciousness_state.is_conscious}")
        logger.info(f"  Criticality: {consciousness_state.metrics.criticality:.3f}")
        logger.info(f"  Phenomenal Richness: {consciousness_state.metrics.phenomenal_richness:.3f}")
        logger.info(f"  Quality Score: {consciousness_state.consciousness_quality:.3f}")
        logger.info("")
    
    await consciousness_core.shutdown()
    logger.info("✅ Consciousness measurement test completed")

async def test_neural_substrate():
    """Test neural substrate processing"""
    logger.info("🧠 Testing Neural Substrate...")
    
    neural_substrate = NeuralSubstrate()
    
    # Test input processing
    test_inputs = [
        {
            'text': 'Design a machine learning algorithm',
            'context': {'task_type': 'technical', 'complexity': 'high'}
        },
        {
            'text': 'Write a creative story',
            'context': {'task_type': 'creative', 'complexity': 'medium'}
        },
        {
            'embeddings': [0.1, 0.2, 0.3, 0.4, 0.5] * 100,  # 500 dimensions
            'context': {'task_type': 'analysis', 'complexity': 'low'}
        }
    ]
    
    for i, input_data in enumerate(test_inputs):
        result = await neural_substrate.process_input(input_data)
        
        logger.info(f"Neural Processing Test {i+1}")
        logger.info(f"  Input Type: {list(input_data.keys())[0]}")
        logger.info(f"  Processing Time: {result.get('processing_time_ms', 0):.2f}ms")
        logger.info(f"  Energy Level: {result.get('energy_level', 0):.3f}")
        logger.info(f"  Processing Load: {result.get('processing_load', 0):.3f}")
        logger.info(f"  Neural Activity Shape: {result['neural_activity'].shape}")
        logger.info("")
    
    # Test instruction execution
    instruction_result = await neural_substrate.execute_instruction(
        "Analyze the given data and provide insights",
        {'input_data': {'features': [1, 2, 3, 4, 5]}}
    )
    
    logger.info("Instruction Execution Test")
    logger.info(f"  Success: {instruction_result.get('success', False)}")
    logger.info(f"  Result Type: {instruction_result.get('result_type', 'unknown')}")
    logger.info(f"  Execution Time: {instruction_result.get('execution_time_ms', 0):.2f}ms")
    logger.info("")
    
    await neural_substrate.shutdown()
    logger.info("✅ Neural substrate test completed")

async def test_memory_system():
    """Test perfect recall engine"""
    logger.info("🧠 Testing Perfect Recall Engine...")
    
    memory_engine = PerfectRecallEngine()
    
    # Store some test memories
    test_memories = [
        {
            'memory_type': 'episodic',
            'content': {'task': 'algorithm_design', 'outcome': 'successful'},
            'summary': 'Successfully designed a sorting algorithm',
            'keywords': ['algorithm', 'sorting', 'design', 'success'],
            'importance': 'high',
            'event_description': 'Designed an efficient quicksort algorithm',
            'outcome': 'Algorithm implemented and tested successfully'
        },
        {
            'memory_type': 'semantic',
            'content': {'concept': 'machine_learning', 'definition': 'AI that learns from data'},
            'summary': 'Knowledge about machine learning fundamentals',
            'keywords': ['machine_learning', 'AI', 'data', 'learning'],
            'importance': 'high',
            'concept': 'Machine Learning',
            'definition': 'A subset of AI that enables systems to learn from data',
            'category': 'artificial_intelligence'
        },
        {
            'memory_type': 'procedural',
            'content': {'skill': 'problem_solving', 'steps': ['analyze', 'plan', 'execute']},
            'summary': 'Problem-solving methodology',
            'keywords': ['problem_solving', 'methodology', 'process'],
            'importance': 'medium',
            'skill_name': 'Systematic Problem Solving',
            'procedure_steps': [
                {'step': 1, 'description': 'Analyze the problem'},
                {'step': 2, 'description': 'Plan the solution'},
                {'step': 3, 'description': 'Execute the plan'}
            ]
        }
    ]
    
    stored_ids = []
    for memory in test_memories:
        memory_id = await memory_engine.store_experience(memory)
        stored_ids.append(memory_id)
        logger.info(f"Stored memory: {memory_id} ({memory['memory_type']})")
    
    # Test memory retrieval
    query_contexts = [
        {
            'keywords': ['algorithm', 'design'],
            'memory_types': ['episodic'],
            'max_results': 5
        },
        {
            'keywords': ['machine_learning', 'AI'],
            'memory_types': ['semantic'],
            'max_results': 3
        },
        {
            'keywords': ['problem_solving'],
            'memory_types': ['procedural'],
            'max_results': 2
        }
    ]
    
    for i, query in enumerate(query_contexts):
        memories = await memory_engine.retrieve_relevant_memories(query)
        logger.info(f"Query {i+1} - Keywords: {query['keywords']}")
        logger.info(f"  Retrieved {len(memories)} memories")
        for memory in memories:
            logger.info(f"    - {memory['summary']} (relevance: {memory['relevance_score']:.3f})")
        logger.info("")
    
    # Test semantic search
    semantic_results = await memory_engine.search_semantic(
        "algorithm design and implementation",
        {'max_results': 3}
    )
    
    logger.info("Semantic Search Test")
    logger.info(f"  Found {len(semantic_results)} relevant memories")
    for memory in semantic_results:
        logger.info(f"    - {memory['summary']} (semantic score: {memory.get('semantic_score', 0):.3f})")
    logger.info("")
    
    # Get memory statistics
    stats = memory_engine.get_memory_stats()
    logger.info("Memory System Statistics")
    logger.info(f"  Total Memories: {stats['total_memories']}")
    logger.info(f"  Memory Types: {stats['memory_types']}")
    logger.info(f"  Average Consolidation: {stats['avg_consolidation_strength']:.3f}")
    logger.info(f"  Average Retrieval Time: {stats['avg_retrieval_time_ms']:.2f}ms")
    logger.info("")
    
    await memory_engine.shutdown()
    logger.info("✅ Memory system test completed")

async def test_cognitive_cycle():
    """Test complete cognitive cycle"""
    logger.info("🧠 Testing Complete Cognitive Cycle...")
    
    orchestrator = AGICoreOrchestrator()
    
    # Test different types of goals
    test_goals = [
        {
            'goal': 'Design an efficient sorting algorithm',
            'user_context': {
                'user_id': 'test_user_1',
                'task_type': 'technical',
                'complexity': 'medium',
                'preferences': {'approach': 'systematic'}
            }
        },
        {
            'goal': 'Create a creative story about AI consciousness',
            'user_context': {
                'user_id': 'test_user_2',
                'task_type': 'creative',
                'complexity': 'high',
                'preferences': {'style': 'philosophical'}
            }
        },
        {
            'goal': 'Analyze the benefits of renewable energy',
            'user_context': {
                'user_id': 'test_user_3',
                'task_type': 'analytical',
                'complexity': 'medium',
                'preferences': {'depth': 'comprehensive'}
            }
        }
    ]
    
    for i, test_case in enumerate(test_goals):
        logger.info(f"Cognitive Cycle Test {i+1}: {test_case['goal']}")
        
        result = await orchestrator.execute_cognitive_cycle(
            test_case['goal'],
            test_case['user_context']
        )
        
        logger.info(f"  Success: {result['success']}")
        logger.info(f"  Cycle ID: {result['cycle_id']}")
        logger.info(f"  Consciousness Level: {result.get('consciousness_level', 0):.3f}")
        
        if result['success']:
            performance = result['performance_metrics']
            logger.info(f"  Total Duration: {performance.get('total_duration_ms', 0):.2f}ms")
            logger.info(f"  Success Rate: {performance.get('success_rate', 0):.1f}%")
            logger.info(f"  Completed Steps: {performance.get('successful_steps', 0)}/{performance.get('total_steps', 0)}")
        else:
            logger.info(f"  Error: {result.get('error', 'Unknown error')}")
        
        logger.info("")
    
    # Get system status
    status = orchestrator.get_system_status()
    logger.info("System Status Summary")
    logger.info(f"  Total Cycles: {status['total_cycles_executed']}")
    logger.info(f"  Average Cycle Time: {status['avg_cycle_time_seconds']:.2f}s")
    logger.info(f"  Success Rate: {status['success_rate_percent']:.1f}%")
    logger.info(f"  Current Consciousness: {status['current_consciousness_level']:.3f}")
    logger.info(f"  Is Conscious: {status['is_conscious']}")
    logger.info("")
    
    await orchestrator.shutdown()
    logger.info("✅ Cognitive cycle test completed")

async def main():
    """Run all tests"""
    logger.info("🚀 Starting AGI Consciousness System Tests")
    logger.info("=" * 60)
    
    try:
        # Test individual components
        await test_consciousness_measurement()
        await test_neural_substrate()
        await test_memory_system()
        
        # Test integrated system
        await test_cognitive_cycle()
        
        logger.info("=" * 60)
        logger.info("🎉 All tests completed successfully!")
        logger.info("The AGI Consciousness Centric Cognition System is operational.")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)