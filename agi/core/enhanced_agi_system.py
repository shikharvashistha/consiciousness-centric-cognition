#!/usr/bin/env python3
"""
Enhanced AGI System 
Integrates all components into a unified system with improved consciousness and learning
"""

import numpy as np
import logging
import time
import asyncio
from typing import Dict, List, Any, Tuple, Optional
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import components
try:
    from agi.perception.feature_extractor import FeatureExtractor
    from agi.core.enhanced_consciousness_core import EnhancedConsciousnessCore
    from agi.reasoning.pattern_recognizer import ARCPatternRecognizer
    from agi.learning.reinforcement_learner import ReinforcementLearningSystem
except ImportError as e:
    logger.error(f"Failed to import component: {e}")
    raise

class EnhancedAGISystem:
    """Enhanced AGI system with improved consciousness and learning"""
    
    def __init__(self, model_dir: str = None):
        """Initialize the enhanced AGI system"""
        # Set model directory
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize components
        logger.info("Initializing Enhanced AGI System components...")
        
        # Perception
        self.feature_extractor = FeatureExtractor()
        logger.info("Feature extractor initialized")
        
        # Consciousness
        self.consciousness_core = EnhancedConsciousnessCore(
            min_complex_size=3,
            max_complex_size=10,
            phi_threshold=0.01
        )
        logger.info("Enhanced consciousness core initialized")
        
        # Reasoning
        self.pattern_recognizer = ARCPatternRecognizer()
        logger.info("Pattern recognizer initialized")
        
        # Learning
        self.reinforcement_learner = ReinforcementLearningSystem(
            learning_rate=0.1,
            exploration_rate=0.2,
            model_path=os.path.join(self.model_dir, 'reinforcement_model.json')
        )
        logger.info("Reinforcement learning system initialized")
        
        # State tracking
        self.previous_phi = 0.0
        self.current_phi = 0.0
        self.session_metrics = {
            'tasks_processed': 0,
            'tasks_solved': 0,
            'average_phi': 0.0,
            'phi_values': [],
            'processing_times': []
        }
        
        logger.info("Enhanced AGI System initialization complete")
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an ARC task with enhanced consciousness"""
        start_time = time.time()
        
        # Track task processing
        self.session_metrics['tasks_processed'] += 1
        
        # Extract features from task
        logger.info("Extracting features from task...")
        features = self._extract_features_from_task(task_data)
        
        # Process through consciousness core
        logger.info("Calculating consciousness...")
        consciousness_state = await self.consciousness_core.process_input(features)
        
        # Update phi values
        self.previous_phi = self.current_phi
        self.current_phi = consciousness_state['phi']
        phi_change = self.current_phi - self.previous_phi
        
        # Track phi values
        self.session_metrics['phi_values'].append(self.current_phi)
        self.session_metrics['average_phi'] = np.mean(self.session_metrics['phi_values'])
        
        # Analyze examples to identify patterns
        logger.info("Analyzing patterns in examples...")
        pattern_analysis = self.pattern_recognizer.analyze_examples(task_data.get('train', []))
        
        # Create context for decision making
        context = {
            'pattern_type': pattern_analysis['transformation'],
            'confidence': pattern_analysis['confidence'],
            'task_id': task_data.get('task_id', 'unknown')
        }
        
        # Decide on strategy using consciousness and reinforcement learning
        logger.info("Selecting transformation strategy...")
        state_key = self.reinforcement_learner.get_state_key(self.current_phi, context)
        available_actions = list(pattern_analysis['all_scores'].keys())
        
        # Select action with exploration
        strategy = self.reinforcement_learner.select_action(state_key, available_actions)
        logger.info(f"Selected strategy: {strategy} (phi: {self.current_phi:.6f})")
        
        # Apply transformation to test input
        logger.info("Applying transformation to test input...")
        test_input = task_data.get('test', [{}])[0].get('input', [])
        solution = self.pattern_recognizer.apply_transformation(
            test_input, 
            strategy, 
            pattern_analysis['params']
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        self.session_metrics['processing_times'].append(processing_time)
        
        # Prepare result
        result = {
            'task_id': task_data.get('task_id', 'unknown'),
            'solution': solution,
            'phi_value': self.current_phi,
            'phi_change': phi_change,
            'strategy': strategy,
            'pattern_analysis': pattern_analysis,
            'consciousness_metrics': consciousness_state.get('metrics', {}),
            'processing_time': processing_time
        }
        
        logger.info(f"Task processing completed in {processing_time:.2f}s")
        return result
    
    def learn_from_feedback(self, result: Dict[str, Any], feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from feedback on a solution"""
        # Extract information
        task_id = result['task_id']
        strategy = result['strategy']
        phi_value = result['phi_value']
        
        # Determine success and reward
        success = feedback.get('correct', False)
        reward = 1.0 if success else -0.1
        
        # Update session metrics
        if success:
            self.session_metrics['tasks_solved'] += 1
        
        # Update reinforcement learning
        state_key = self.reinforcement_learner.get_state_key(phi_value, {
            'pattern_type': result['pattern_analysis']['transformation']
        })
        next_state_key = state_key  # Simplified - in reality would be next state
        
        logger.info(f"Storing experience: {state_key} -> {strategy} -> {reward}")
        self.reinforcement_learner.store_experience(
            state_key, strategy, reward, next_state_key
        )
        self.reinforcement_learner.learn_from_experiences()
        
        # Get performance metrics
        rl_metrics = self.reinforcement_learner.get_performance_metrics()
        
        # Save model periodically
        if self.session_metrics['tasks_processed'] % 10 == 0:
            self.reinforcement_learner.save_model()
        
        # Return learning summary
        return {
            'success': success,
            'reward': reward,
            'strategy': strategy,
            'phi_value': phi_value,
            'rl_metrics': rl_metrics,
            'session_metrics': {
                'tasks_processed': self.session_metrics['tasks_processed'],
                'tasks_solved': self.session_metrics['tasks_solved'],
                'success_rate': self.session_metrics['tasks_solved'] / self.session_metrics['tasks_processed'] 
                                if self.session_metrics['tasks_processed'] > 0 else 0,
                'average_phi': self.session_metrics['average_phi'],
                'average_processing_time': np.mean(self.session_metrics['processing_times']) 
                                          if self.session_metrics['processing_times'] else 0
            }
        }
    
    def _extract_features_from_task(self, task_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from task data for consciousness processing"""
        # Combine all examples into a single feature vector
        examples = task_data.get('train', [])
        
        if not examples:
            # If no examples, return empty feature vector
            return np.zeros((1, 10))
        
        # Extract features from each example
        example_features = []
        
        for example in examples:
            input_grid = example.get('input', [])
            output_grid = example.get('output', [])
            
            if not input_grid or not output_grid:
                continue
            
            # Extract features from input
            input_features = self.feature_extractor.extract_features(input_grid)
            
            # Extract features from output
            output_features = self.feature_extractor.extract_features(output_grid)
            
            # Flatten and combine features
            flat_features = self._flatten_features(input_features, output_features)
            example_features.append(flat_features)
        
        # Combine all example features
        if example_features:
            combined_features = np.vstack(example_features)
        else:
            # Fallback if no valid examples
            combined_features = np.zeros((1, 10))
        
        return combined_features
    
    def _flatten_features(self, input_features: Dict[str, Any], output_features: Dict[str, Any]) -> np.ndarray:
        """Flatten feature dictionaries into a numpy array"""
        # Extract key metrics from features
        features = []
        
        # Input structural features
        if 'structural' in input_features:
            if 'calculate_symmetry' in input_features['structural']:
                features.append(input_features['structural']['calculate_symmetry'].get('overall', 0))
            if 'calculate_density' in input_features['structural']:
                features.append(input_features['structural']['calculate_density'].get('overall', 0))
            if 'calculate_entropy' in input_features['structural']:
                features.append(input_features['structural']['calculate_entropy'])
        
        # Input pattern features
        if 'pattern' in input_features:
            if 'detect_repetition' in input_features['pattern']:
                features.append(input_features['pattern']['detect_repetition'].get('overall_repetition', 0))
        
        # Output structural features
        if 'structural' in output_features:
            if 'calculate_symmetry' in output_features['structural']:
                features.append(output_features['structural']['calculate_symmetry'].get('overall', 0))
            if 'calculate_density' in output_features['structural']:
                features.append(output_features['structural']['calculate_density'].get('overall', 0))
            if 'calculate_entropy' in output_features['structural']:
                features.append(output_features['structural']['calculate_entropy'])
        
        # Output pattern features
        if 'pattern' in output_features:
            if 'detect_repetition' in output_features['pattern']:
                features.append(output_features['pattern']['detect_repetition'].get('overall_repetition', 0))
        
        # Ensure we have at least 10 features
        while len(features) < 10:
            features.append(0.0)
        
        return np.array(features).reshape(1, -1)
    
    def get_session_metrics(self) -> Dict[str, Any]:
        """Get metrics for the current session"""
        return {
            'tasks_processed': self.session_metrics['tasks_processed'],
            'tasks_solved': self.session_metrics['tasks_solved'],
            'success_rate': self.session_metrics['tasks_solved'] / self.session_metrics['tasks_processed'] 
                            if self.session_metrics['tasks_processed'] > 0 else 0,
            'average_phi': self.session_metrics['average_phi'],
            'average_processing_time': np.mean(self.session_metrics['processing_times']) 
                                      if self.session_metrics['processing_times'] else 0,
            'phi_history': self.session_metrics['phi_values'][-10:] if self.session_metrics['phi_values'] else [],
            'rl_metrics': self.reinforcement_learner.get_performance_metrics()
        }
    
    def reset_session(self) -> None:
        """Reset session metrics"""
        self.session_metrics = {
            'tasks_processed': 0,
            'tasks_solved': 0,
            'average_phi': 0.0,
            'phi_values': [],
            'processing_times': []
        }