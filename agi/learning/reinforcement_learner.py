#!/usr/bin/env python3
"""
Reinforcement Learning System for AGI
Learns from feedback to improve decision-making
"""

import numpy as np
import random
from typing import Dict, List, Any, Tuple, Optional
import logging
import json
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReinforcementLearningSystem:
    """Learn from feedback to improve decision-making"""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2, 
                 min_exploration_rate=0.05, exploration_decay=0.99, model_path=None):
        """Initialize the reinforcement learning system"""
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay
        
        # Q-table for state-action values
        self.q_table = {}
        
        # Experience buffer for batch learning
        self.experience_buffer = []
        self.max_buffer_size = 10000
        
        # Performance tracking
        self.total_rewards = 0
        self.episode_count = 0
        self.success_count = 0
        
        # Model path for saving/loading
        self.model_path = model_path
        if model_path:
            self.load_model()
    
    def get_state_key(self, phi_value: float, context: Optional[Dict] = None) -> str:
        """Convert phi value and context to a state key"""
        # Discretize phi value
        phi_bin = round(phi_value * 10) / 10
        
        # Add context information if available
        if context and 'pattern_type' in context:
            return f"{phi_bin}_{context['pattern_type']}"
        
        return str(phi_bin)
    
    def select_action(self, state_key: str, available_actions: List[str]) -> str:
        """Select action using epsilon-greedy policy"""
        # Initialize state in Q-table if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in available_actions}
        elif not all(action in self.q_table[state_key] for action in available_actions):
            # Add any new actions
            for action in available_actions:
                if action not in self.q_table[state_key]:
                    self.q_table[state_key][action] = 0.0
        
        # Exploration: random action
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)
        
        # Exploitation: best action
        return self._get_best_action(state_key, available_actions)
    
    def _get_best_action(self, state_key: str, available_actions: List[str]) -> str:
        """Get the action with the highest Q-value"""
        # Filter to only include available actions
        valid_actions = {a: self.q_table[state_key][a] for a in available_actions if a in self.q_table[state_key]}
        
        if not valid_actions:
            return random.choice(available_actions)
        
        # Find action with maximum Q-value
        max_q = max(valid_actions.values())
        best_actions = [a for a, q in valid_actions.items() if q == max_q]
        
        # If multiple actions have the same Q-value, choose randomly
        return random.choice(best_actions)
    
    def store_experience(self, state: str, action: str, reward: float, next_state: str) -> None:
        """Store experience for batch learning"""
        self.experience_buffer.append((state, action, reward, next_state))
        
        # Limit buffer size
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
        
        # Update performance metrics
        self.total_rewards += reward
        self.episode_count += 1
        if reward > 0:
            self.success_count += 1
    
    def learn_from_experiences(self, batch_size=32) -> None:
        """Learn from stored experiences using Q-learning"""
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample batch of experiences
        batch = random.sample(self.experience_buffer, batch_size)
        
        for state, action, reward, next_state in batch:
            # Initialize if needed
            if state not in self.q_table:
                self.q_table[state] = {}
            if action not in self.q_table[state]:
                self.q_table[state][action] = 0.0
            
            # Q-learning update
            if next_state in self.q_table:
                max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
                self.q_table[state][action] += self.learning_rate * (
                    reward + self.discount_factor * max_next_q - self.q_table[state][action]
                )
            else:
                self.q_table[state][action] += self.learning_rate * (
                    reward - self.q_table[state][action]
                )
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics of the learning system"""
        success_rate = self.success_count / self.episode_count if self.episode_count > 0 else 0
        avg_reward = self.total_rewards / self.episode_count if self.episode_count > 0 else 0
        
        return {
            'success_rate': success_rate,
            'average_reward': avg_reward,
            'exploration_rate': self.exploration_rate,
            'episode_count': self.episode_count
        }
    
    def save_model(self, path: Optional[str] = None) -> None:
        """Save the Q-table to a file"""
        save_path = path or self.model_path
        if not save_path:
            logger.warning("No path specified for saving model")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert Q-table to serializable format
        serializable_q_table = {}
        for state, actions in self.q_table.items():
            serializable_q_table[state] = {str(a): float(v) for a, v in actions.items()}
        
        # Save to file
        with open(save_path, 'w') as f:
            json.dump({
                'q_table': serializable_q_table,
                'metrics': {
                    'total_rewards': self.total_rewards,
                    'episode_count': self.episode_count,
                    'success_count': self.success_count,
                    'exploration_rate': self.exploration_rate
                }
            }, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, path: Optional[str] = None) -> bool:
        """Load the Q-table from a file"""
        load_path = path or self.model_path
        if not load_path:
            logger.warning("No path specified for loading model")
            return False
        
        # Check if file exists
        if not os.path.exists(load_path):
            logger.warning(f"Model file not found: {load_path}")
            return False
        
        try:
            # Load from file
            with open(load_path, 'r') as f:
                data = json.load(f)
            
            # Convert to Q-table format
            self.q_table = {}
            for state, actions in data['q_table'].items():
                self.q_table[state] = {a: float(v) for a, v in actions.items()}
            
            # Load metrics
            metrics = data.get('metrics', {})
            self.total_rewards = metrics.get('total_rewards', 0)
            self.episode_count = metrics.get('episode_count', 0)
            self.success_count = metrics.get('success_count', 0)
            self.exploration_rate = metrics.get('exploration_rate', self.exploration_rate)
            
            logger.info(f"Model loaded from {load_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def reset_metrics(self) -> None:
        """Reset performance metrics"""
        self.total_rewards = 0
        self.episode_count = 0
        self.success_count = 0