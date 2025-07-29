#!/usr/bin/env python3
"""
Quick test of the enhanced AGI system
"""

import asyncio
import logging
import time
from agi.core.agi_orchestrator import AGICoreOrchestrator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def quick_test():
    """Run a quick test of the enhanced system"""
    logger.info("🚀 Starting Quick AGI System Test")
    
    # Initialize the system
    orchestrator = AGICoreOrchestrator()
    
    try:
        # Test simple goal
        goal = "Create a simple sorting algorithm that demonstrates consciousness-aware processing"
        user_context = {"user_id": "test_user", "preferences": {"creativity": "high"}}
        
        logger.info(f"🎯 Testing goal: {goal}")
        
        start_time = time.time()
        result = await orchestrator.execute_cognitive_cycle(goal, user_context)
        end_time = time.time()
        
        logger.info(f"✅ Test completed in {end_time - start_time:.2f} seconds")
        
        # Print key results
        if result:
            logger.info(f"📊 Result keys: {list(result.keys())}")
            
            # Check final output
            if 'final_output' in result:
                output_length = len(str(result['final_output']))
                logger.info(f"📊 Output length: {output_length} characters")
                
                # Show first 200 characters of output
                output_preview = str(result['final_output'])[:200] + "..." if len(str(result['final_output'])) > 200 else str(result['final_output'])
                logger.info(f"📝 Output preview: {output_preview}")
            
            # Check consciousness level
            if 'consciousness_level' in result:
                logger.info(f"🧠 Consciousness level: {result['consciousness_level']}")
            
            # Check performance metrics
            if 'performance_metrics' in result:
                logger.info(f"📈 Performance metrics: {result['performance_metrics']}")
            
            # Check cycle summary
            if 'cycle_summary' in result:
                logger.info(f"📋 Cycle summary: {result['cycle_summary']}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Shutdown
        logger.info("🔄 Shutting down system...")
        await orchestrator.shutdown()
        logger.info("✅ System shutdown complete")

if __name__ == "__main__":
    asyncio.run(quick_test())