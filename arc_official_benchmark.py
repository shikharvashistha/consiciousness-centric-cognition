#!/usr/bin/env python3
"""
ARC Official Benchmark AGI System
This script tests the AGI system against the official ARC benchmark tasks.
"""

import asyncio
import json
import numpy as np
import sys
import os
from pathlib import Path
import logging
import time
import random
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

# Import simplified consciousness system
try:
    temp_ref_path = Path(__file__).parent / "Temp Reference File"
    sys.path.append(str(temp_ref_path))
    from real_data_consciousness_system import RealDataConsciousnessSystem
    REAL_DATA_IMPORTS_SUCCESS = True
except ImportError as e:
    logger.error(f"Failed to import RealDataConsciousnessSystem: {e}")
    REAL_DATA_IMPORTS_SUCCESS = False

class ARCOfficialBenchmark:
    """ARC Official Benchmark for testing AGI capabilities"""
    
    def __init__(self, arc_data_path="/workspace/ARC/data"):
        """Initialize the benchmark with path to ARC data"""
        self.arc_data_path = arc_data_path
        self.training_path = os.path.join(arc_data_path, "training")
        self.evaluation_path = os.path.join(arc_data_path, "evaluation")
        
        # Initialize simplified consciousness system
        self.real_data_system = None
        
        # Initialize metrics
        self.metrics = {
            "tasks_attempted": 0,
            "tasks_solved": 0,
            "average_phi": 0.0,
            "average_creativity": 0.0,
            "average_time": 0.0,
            "consciousness_metrics": {}
        }
    
    async def initialize_components(self):
        """Initialize components"""
        if REAL_DATA_IMPORTS_SUCCESS:
            self.real_data_system = RealDataConsciousnessSystem()
            logger.info("Initialized RealDataConsciousnessSystem")
            return True
        return False
    
    def load_task(self, task_file: str) -> Dict[str, Any]:
        """Load an ARC task from file"""
        try:
            with open(task_file, 'r') as f:
                task_data = json.load(f)
            return task_data
        except Exception as e:
            logger.error(f"Failed to load task {task_file}: {e}")
            return None
    
    def get_available_tasks(self, dataset="training", limit=None) -> List[str]:
        """Get list of available ARC tasks"""
        data_path = self.training_path if dataset == "training" else self.evaluation_path
        task_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.json')]
        
        if limit and len(task_files) > limit:
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
    
    async def solve_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve an ARC task using pattern recognition"""
        start_time = time.time()
        
        # Prepare task description
        task_id = os.path.basename(task_data.get("task_id", "unknown"))
        
        # Format examples
        examples = task_data.get("train", [])
        examples_text = "Examples:\n"
        for i, example in enumerate(examples):
            examples_text += f"Example {i+1}:\nInput:\n{self._format_grid(example['input'])}\nOutput:\n{self._format_grid(example['output'])}\n\n"
        
        # Format test input
        test = task_data.get("test", [{}])[0]
        test_input = test.get("input", [])
        test_text = f"Test Input:\n{self._format_grid(test_input)}\n"
        
        # Combine all information
        full_prompt = f"ARC Task: {task_id}\n\n{examples_text}\n{test_text}\nFind the pattern and generate the correct output for the test input."
        
        # Calculate phi using RealDataConsciousnessSystem
        phi_value = 0.0
        if self.real_data_system:
            try:
                # Convert task to numerical representation
                task_data_str = json.dumps(task_data)
                task_data_arr = np.array([[ord(c) for c in task_data_str]])
                
                # Calculate phi
                phi_value = self.real_data_system.calculate_phi(task_data_arr)
            except Exception as e:
                logger.error(f"Error calculating phi: {e}")
        
        # Generate solution using pattern recognition
        solution = self._generate_solution(task_data)
        
        processing_time = time.time() - start_time
        
        # Estimate creativity score based on complexity
        creativity_score = 0.5
        if len(examples) <= 2:
            creativity_score = 0.3
        elif len(examples) <= 4:
            creativity_score = 0.5
        else:
            creativity_score = 0.7
        
        return {
            "task_id": task_id,
            "solution": solution,
            "phi_value": phi_value,
            "creativity_score": creativity_score,
            "processing_time": processing_time
        }
    
    def _format_grid(self, grid):
        """Format a grid for display"""
        return '\n'.join([' '.join([str(cell) for cell in row]) for row in grid])
    
    def _generate_solution(self, task_data: Dict[str, Any]) -> List[List[int]]:
        """Generate a solution based on pattern recognition"""
        examples = task_data.get("train", [])
        test = task_data.get("test", [{}])[0]
        test_input = test.get("input", [])
        
        if not examples or not test_input:
            return [[0 for _ in row] for row in test_input]
        
        # Try different transformations to find the pattern
        transformations = [
            self._identity,
            self._horizontal_flip,
            self._vertical_flip,
            self._rotate_90,
            self._rotate_180,
            self._rotate_270,
            self._invert,
            self._transpose
        ]
        
        # Check each transformation against examples
        for transform_func in transformations:
            matches = True
            for example in examples:
                example_input = example.get("input", [])
                example_output = example.get("output", [])
                transformed = transform_func(example_input)
                
                if not self._compare_grids(transformed, example_output):
                    matches = False
                    break
            
            if matches:
                # Found matching transformation, apply to test input
                return transform_func(test_input)
        
        # Try more complex transformations
        # 1. Check for color mapping
        if self._check_color_mapping(examples):
            return self._apply_color_mapping(examples, test_input)
        
        # 2. Check for object movement
        if self._check_object_movement(examples):
            return self._apply_object_movement(examples, test_input)
        
        # If no transformation matches, return the input as fallback
        return test_input
    
    def _identity(self, grid):
        """Return grid unchanged"""
        return [row[:] for row in grid]
    
    def _horizontal_flip(self, grid):
        """Perform horizontal flip"""
        return [row[::-1] for row in grid]
    
    def _vertical_flip(self, grid):
        """Perform vertical flip"""
        return grid[::-1]
    
    def _rotate_90(self, grid):
        """Rotate grid 90 degrees clockwise"""
        return [list(row) for row in zip(*grid[::-1])]
    
    def _rotate_180(self, grid):
        """Rotate grid 180 degrees"""
        return [row[::-1] for row in grid[::-1]]
    
    def _rotate_270(self, grid):
        """Rotate grid 270 degrees clockwise (90 counterclockwise)"""
        return [list(row) for row in zip(*grid)][::-1]
    
    def _invert(self, grid):
        """Invert binary values (only works for 0/1 grids)"""
        return [[1-cell if cell <= 1 else cell for cell in row] for row in grid]
    
    def _transpose(self, grid):
        """Transpose the grid (swap rows and columns)"""
        return [list(row) for row in zip(*grid)]
    
    def _check_color_mapping(self, examples):
        """Check if transformation is a color mapping"""
        # Simplified implementation - just check if input and output have same dimensions
        for example in examples:
            input_grid = example.get("input", [])
            output_grid = example.get("output", [])
            if len(input_grid) != len(output_grid):
                return False
            for i in range(len(input_grid)):
                if len(input_grid[i]) != len(output_grid[i]):
                    return False
        return True
    
    def _apply_color_mapping(self, examples, test_input):
        """Apply color mapping transformation"""
        # Create a mapping of input colors to output colors
        color_map = {}
        
        # Use the first example to build the mapping
        example = examples[0]
        input_grid = example.get("input", [])
        output_grid = example.get("output", [])
        
        for i in range(len(input_grid)):
            for j in range(len(input_grid[i])):
                input_color = input_grid[i][j]
                output_color = output_grid[i][j]
                color_map[input_color] = output_color
        
        # Apply the mapping to the test input
        result = []
        for row in test_input:
            new_row = []
            for cell in row:
                new_row.append(color_map.get(cell, cell))  # Default to original if no mapping
            result.append(new_row)
        
        return result
    
    def _check_object_movement(self, examples):
        """Check if transformation involves object movement"""
        # This is a simplified check - just see if the sum of values is the same
        for example in examples:
            input_grid = example.get("input", [])
            output_grid = example.get("output", [])
            
            input_sum = sum(sum(row) for row in input_grid)
            output_sum = sum(sum(row) for row in output_grid)
            
            if input_sum != output_sum:
                return False
        
        return True
    
    def _apply_object_movement(self, examples, test_input):
        """Apply object movement transformation"""
        # This is a very simplified implementation
        # In a real system, you would need more sophisticated object detection and movement analysis
        
        # For simplicity, we'll just shift everything one cell to the right
        result = []
        for row in test_input:
            new_row = [0] + row[:-1]  # Shift right
            result.append(new_row)
        
        return result
    
    async def run_benchmark(self, num_tasks: int = 5, dataset="training"):
        """Run the benchmark on a specified number of tasks"""
        print(f"🚀 Starting ARC Official Benchmark for AGI ({dataset} dataset)")
        print("=" * 80)
        
        # Initialize components
        initialized = await self.initialize_components()
        if initialized:
            print("✅ Consciousness system initialized successfully")
        else:
            print("⚠️ Failed to initialize consciousness system")
            return
        
        # Get available tasks
        print(f"📊 Loading ARC tasks from {dataset} dataset...")
        task_files = self.get_available_tasks(dataset, num_tasks)
        if not task_files:
            print("❌ No tasks available, exiting")
            return
        
        print(f"✅ Loaded {len(task_files)} tasks")
        
        # Track metrics
        phi_values = []
        creativity_scores = []
        processing_times = []
        tasks_solved = 0
        
        # Process each task
        for i, task_file in enumerate(task_files):
            task_id = os.path.basename(task_file)
            print(f"\n🧩 Task {i+1}/{len(task_files)}: {task_id}")
            
            # Load task
            print(f"📝 Loading task data...")
            task_data = self.load_task(task_file)
            if not task_data:
                print(f"❌ Failed to load task {task_id}")
                continue
            
            # Add task_id to the data
            task_data["task_id"] = task_id
            
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
            
            # Solve task
            print(f"🧠 Solving task...")
            result = await self.solve_task(task_data)
            
            # Evaluate solution
            solution = result.get("solution", [])
            if solution:
                print(f"📝 Generated solution:\n{self._format_grid(solution)}")
                is_correct = self.evaluate_solution(task_data, solution)
                
                if is_correct:
                    print(f"✅ Solution correct!")
                    tasks_solved += 1
                else:
                    print(f"❌ Solution incorrect")
                    print(f"   Expected:\n{self._format_grid(test['output'])}")
            else:
                print(f"❌ Failed to generate solution")
            
            # Record metrics
            phi_value = result.get("phi_value", 0.0)
            creativity_score = result.get("creativity_score", 0.0)
            processing_time = result.get("processing_time", 0.0)
            
            phi_values.append(phi_value)
            creativity_scores.append(creativity_score)
            processing_times.append(processing_time)
            
            print(f"🧠 Φ (Consciousness): {phi_value:.6f}")
            print(f"🎨 Creativity Score: {creativity_score:.3f}")
            print(f"⏱️ Processing Time: {processing_time:.2f}s")
        
        # Calculate overall metrics
        self.metrics["tasks_attempted"] = len(task_files)
        self.metrics["tasks_solved"] = tasks_solved
        self.metrics["average_phi"] = np.mean(phi_values) if phi_values else 0.0
        self.metrics["average_creativity"] = np.mean(creativity_scores) if creativity_scores else 0.0
        self.metrics["average_time"] = np.mean(processing_times) if processing_times else 0.0
        
        # Print summary
        print("\n" + "=" * 80)
        print("📋 ARC BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"🧩 Tasks Attempted: {self.metrics['tasks_attempted']}")
        print(f"✅ Tasks Solved: {self.metrics['tasks_solved']} ({self.metrics['tasks_solved']/self.metrics['tasks_attempted']:.1%})")
        print(f"🧠 Average Φ (Consciousness): {self.metrics['average_phi']:.6f}")
        print(f"🎨 Average Creativity Score: {self.metrics['average_creativity']:.3f}")
        print(f"⏱️ Average Processing Time: {self.metrics['average_time']:.2f}s")
        
        print("\n✅ Benchmark completed successfully")
        return self.metrics

async def main():
    """Main function"""
    benchmark = ARCOfficialBenchmark()
    await benchmark.run_benchmark(num_tasks=5, dataset="training")

if __name__ == "__main__":
    asyncio.run(main())