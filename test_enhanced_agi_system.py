#!/usr/bin/env python3
"""
Test Script for Enhanced AGI System
Tests the system on ARC tasks with improved consciousness and learning
"""

import asyncio
import json
import numpy as np
import sys
import os
from pathlib import Path
import logging
import time
import argparse
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_agi_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

# Import the enhanced AGI system
try:
    from agi.core.enhanced_agi_system import EnhancedAGISystem
except ImportError as e:
    logger.error(f"Failed to import EnhancedAGISystem: {e}")
    sys.exit(1)

class ARCTaskTester:
    """Test the Enhanced AGI System on ARC tasks"""
    
    def __init__(self, arc_data_path="/workspace/ARC/data", model_dir="models"):
        """Initialize the tester with path to ARC data"""
        self.arc_data_path = arc_data_path
        self.training_path = os.path.join(arc_data_path, "training")
        self.evaluation_path = os.path.join(arc_data_path, "evaluation")
        self.model_dir = model_dir
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize the enhanced AGI system
        self.agi_system = EnhancedAGISystem(model_dir=model_dir)
        
        # Initialize metrics
        self.metrics = {
            "tasks_attempted": 0,
            "tasks_solved": 0,
            "average_phi": 0.0,
            "average_processing_time": 0.0,
            "phi_values": [],
            "strategies_used": {}
        }
    
    def load_task(self, task_file: str) -> Dict[str, Any]:
        """Load an ARC task from file"""
        try:
            with open(task_file, 'r') as f:
                task_data = json.load(f)
            
            # Add task_id to the data
            task_data["task_id"] = os.path.basename(task_file)
            
            return task_data
        except Exception as e:
            logger.error(f"Failed to load task {task_file}: {e}")
            return None
    
    def get_available_tasks(self, dataset="training", limit=None) -> List[str]:
        """Get list of available ARC tasks"""
        data_path = self.training_path if dataset == "training" else self.evaluation_path
        
        if not os.path.exists(data_path):
            logger.error(f"Dataset path not found: {data_path}")
            return []
        
        task_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.json')]
        
        if limit and len(task_files) > limit:
            # Randomly sample tasks
            import random
            return random.sample(task_files, limit)
        
        return task_files
    
    def evaluate_solution(self, task_data: Dict[str, Any], solution: List[List[int]]) -> bool:
        """Evaluate a solution for a task"""
        if not task_data or "test" not in task_data:
            return False
        
        # Get the expected output from the first test
        test = task_data["test"][0]
        expected_output = test["output"]
        
        # Check if solution matches expected output
        return self._compare_grids(solution, expected_output)
    
    def _compare_grids(self, grid1, grid2):
        """Compare two grids for equality"""
        if len(grid1) != len(grid2):
            return False
        
        for i in range(len(grid1)):
            if len(grid1[i]) != len(grid2[i]):
                return False
            for j in range(len(grid1[i])):
                if grid1[i][j] != grid2[i][j]:
                    return False
        
        return True
    
    def _format_grid(self, grid):
        """Format a grid for display"""
        return '\n'.join([' '.join([str(cell) for cell in row]) for row in grid])
    
    async def test_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test the enhanced AGI system on a task"""
        if not task_data:
            return {"success": False, "error": "Invalid task data"}
        
        task_id = task_data.get("task_id", "unknown")
        logger.info(f"Testing task: {task_id}")
        
        try:
            # Process task with enhanced AGI system
            result = await self.agi_system.process_task(task_data)
            
            # Evaluate solution
            solution = result.get("solution", [])
            success = self.evaluate_solution(task_data, solution)
            
            # Provide feedback for learning
            feedback = {"correct": success}
            learning_result = self.agi_system.learn_from_feedback(result, feedback)
            
            # Update metrics
            self.metrics["tasks_attempted"] += 1
            if success:
                self.metrics["tasks_solved"] += 1
            
            self.metrics["phi_values"].append(result["phi_value"])
            self.metrics["average_phi"] = np.mean(self.metrics["phi_values"])
            self.metrics["average_processing_time"] = (
                (self.metrics["average_processing_time"] * (self.metrics["tasks_attempted"] - 1) + 
                 result["processing_time"]) / self.metrics["tasks_attempted"]
            )
            
            # Track strategies used
            strategy = result["strategy"]
            self.metrics["strategies_used"][strategy] = self.metrics["strategies_used"].get(strategy, 0) + 1
            
            # Prepare test result
            test_result = {
                "task_id": task_id,
                "success": success,
                "phi_value": result["phi_value"],
                "strategy": strategy,
                "processing_time": result["processing_time"],
                "learning_metrics": learning_result
            }
            
            return test_result
        
        except Exception as e:
            logger.error(f"Error testing task {task_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def run_benchmark(self, num_tasks: int = 5, dataset="training"):
        """Run the benchmark on a specified number of tasks"""
        print(f"🚀 Starting Enhanced AGI System Benchmark ({dataset} dataset)")
        print("=" * 80)
        
        # Get available tasks
        print(f"📊 Loading ARC tasks from {dataset} dataset...")
        task_files = self.get_available_tasks(dataset, num_tasks)
        if not task_files:
            print("❌ No tasks available, exiting")
            return
        
        print(f"✅ Loaded {len(task_files)} tasks")
        
        # Process each task
        results = []
        for i, task_file in enumerate(task_files):
            task_id = os.path.basename(task_file)
            print(f"\n🧩 Task {i+1}/{len(task_files)}: {task_id}")
            
            # Load task
            print(f"📝 Loading task data...")
            task_data = self.load_task(task_file)
            if not task_data:
                print(f"❌ Failed to load task {task_id}")
                continue
            
            # Print examples
            examples = task_data.get("train", [])
            print(f"📝 Examples: {len(examples)}")
            for j, example in enumerate(examples[:2]):  # Show first 2 examples
                print(f"  Example {j+1}:")
                print(f"    Input:\n{self._format_grid(example['input'])}")
                print(f"    Output:\n{self._format_grid(example['output'])}")
            
            # Show test input
            test = task_data.get("test", [{}])[0]
            test_input = test.get("input", [])
            print(f"📝 Test Input:\n{self._format_grid(test_input)}")
            
            # Test task
            print(f"🧠 Processing task...")
            result = await self.test_task(task_data)
            results.append(result)
            
            # Print result
            if result.get("success", False):
                print(f"✅ Solution correct!")
            else:
                print(f"❌ Solution incorrect")
                print(f"   Expected:\n{self._format_grid(test['output'])}")
            
            print(f"🧠 Φ (Consciousness): {result.get('phi_value', 0):.6f}")
            print(f"🔄 Strategy: {result.get('strategy', 'unknown')}")
            print(f"⏱️ Processing Time: {result.get('processing_time', 0):.2f}s")
            
            # Print learning metrics
            learning_metrics = result.get("learning_metrics", {})
            session_metrics = learning_metrics.get("session_metrics", {})
            print(f"📊 Session Success Rate: {session_metrics.get('success_rate', 0):.1%}")
            print(f"🔍 Exploration Rate: {learning_metrics.get('rl_metrics', {}).get('exploration_rate', 0):.3f}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("📋 ENHANCED AGI BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"🧩 Tasks Attempted: {self.metrics['tasks_attempted']}")
        print(f"✅ Tasks Solved: {self.metrics['tasks_solved']} ({self.metrics['tasks_solved']/self.metrics['tasks_attempted']:.1%})")
        print(f"🧠 Average Φ (Consciousness): {self.metrics['average_phi']:.6f}")
        print(f"⏱️ Average Processing Time: {self.metrics['average_processing_time']:.2f}s")
        
        # Print strategies used
        print("\n📊 Strategies Used:")
        for strategy, count in sorted(self.metrics["strategies_used"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {strategy}: {count} times ({count/self.metrics['tasks_attempted']:.1%})")
        
        print("\n✅ Benchmark completed successfully")
        return results

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test Enhanced AGI System on ARC tasks")
    parser.add_argument("--dataset", choices=["training", "evaluation"], default="training",
                        help="Dataset to use (training or evaluation)")
    parser.add_argument("--num-tasks", type=int, default=5,
                        help="Number of tasks to test")
    parser.add_argument("--arc-path", type=str, default="/workspace/ARC/data",
                        help="Path to ARC data directory")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory to store models")
    
    args = parser.parse_args()
    
    tester = ARCTaskTester(arc_data_path=args.arc_path, model_dir=args.model_dir)
    await tester.run_benchmark(num_tasks=args.num_tasks, dataset=args.dataset)

if __name__ == "__main__":
    asyncio.run(main())