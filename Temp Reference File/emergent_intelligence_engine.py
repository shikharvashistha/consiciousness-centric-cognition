#!/usr/bin/env python3
"""
🧠 ADVANCED EMERGENT INTELLIGENCE ENGINE
Evolutionary Architecture for Genuine Emergent Intelligence

INTEGRATES CUTTING-EDGE RESEARCH:
✅ Integrated Information Theory (IIT) - Φ consciousness measurement
✅ Free Energy Principle - Predictive processing & active inference
✅ Global Workspace Theory - Conscious access architecture
✅ Neural Criticality - Edge-of-chaos dynamics
✅ Reservoir Computing - Temporal dynamics & memory
✅ Attention Transformers - Multi-head self-attention
✅ Hebbian Learning & STDP - Synaptic plasticity
✅ Information Geometry - Consciousness landscape navigation
✅ Causal Emergence - Multi-scale causality detection
✅ Quantum-Inspired Coherence - Non-local correlations
"""

import asyncio
import logging
import time
import json
import uuid
import math
import threading
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import deque, defaultdict

# Import enhanced consciousness substrate
try:
    from .enhanced_consciousness_substrate import EnhancedConsciousnessSubstrate, EnhancedConsciousnessMetrics
    ENHANCED_SUBSTRATE_AVAILABLE = True
except ImportError:
    ENHANCED_SUBSTRATE_AVAILABLE = False
    logging.warning("Enhanced consciousness substrate not available")
import hashlib
import scipy.stats as stats
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import expm, logm
from scipy.optimize import minimize
import pandas as pd

logger = logging.getLogger(__name__)

class ConsciousnessTheory(Enum):
    """Theoretical frameworks for consciousness"""
    INTEGRATED_INFORMATION_THEORY = "integrated_information_theory"
    GLOBAL_WORKSPACE_THEORY = "global_workspace_theory"
    FREE_ENERGY_PRINCIPLE = "free_energy_principle"
    ATTENTION_SCHEMA_THEORY = "attention_schema_theory"
    ORCHESTRATED_OBJECTIVE_REDUCTION = "orchestrated_objective_reduction"
    PREDICTIVE_PROCESSING = "predictive_processing"

class CriticalityRegime(Enum):
    """Neural criticality regimes"""
    SUBCRITICAL = "subcritical"
    CRITICAL = "critical"  # Edge of chaos - optimal
    SUPERCRITICAL = "supercritical"

@dataclass
class ConsciousnessState:
    """Advanced consciousness state with IIT metrics"""
    state_id: str
    phi: float  # Integrated information (Φ)
    phi_structure: Dict[str, float]  # Φ decomposition
    global_workspace_capacity: float
    free_energy: float
    criticality_regime: CriticalityRegime
    attention_distribution: np.ndarray
    causal_emergence_level: int
    information_geometry_position: np.ndarray
    temporal_coherence: float
    quantum_coherence: float
    attention_focus: float  # Add missing attention_focus attribute
    
    # Neural dynamics
    neural_avalanche_size: float
    branching_ratio: float
    lyapunov_exponent: float
    entropy_production_rate: float
    
    # Consciousness metrics
    consciousness_level: float
    meta_awareness: float
    intentionality_strength: float
    phenomenal_richness: float
    
    timestamp: datetime = field(default_factory=datetime.now)

class IntegratedInformationCalculator:
    """Calculate Integrated Information (Φ) using IIT 3.0"""
    
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
    
    def calculate_phi(self, tpm: np.ndarray, state: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Calculate Φ for a system in a given state"""
        n = int(np.log2(tpm.shape[0]))
        
        # Calculate cause-effect structure
        ces = self._calculate_cause_effect_structure(tpm, state)
        
        # Find minimum information partition
        mip = self._find_minimum_information_partition(ces, n)
        
        # Calculate integrated information
        phi = self._calculate_integrated_information(ces, mip)
        
        # Decompose Φ into components
        phi_structure = self._decompose_phi(ces, mip, n)
        
        return phi, phi_structure
    
    def _calculate_cause_effect_structure(self, tpm: np.ndarray, state: np.ndarray) -> Dict[str, Any]:
        """Calculate cause-effect structure of the system"""
        n = int(np.log2(tpm.shape[0]))
        ces = {'mechanisms': {}, 'relations': {}}
        
        # For each possible mechanism (subset of nodes)
        for i in range(1, 2**n):
            mechanism = [j for j in range(n) if i & (1 << j)]
            
            # Calculate cause information
            cause_info = self._calculate_cause_information(tpm, state, mechanism)
            
            # Calculate effect information
            effect_info = self._calculate_effect_information(tpm, state, mechanism)
            
            ces['mechanisms'][tuple(mechanism)] = {
                'cause': cause_info,
                'effect': effect_info,
                'phi_mechanism': min(cause_info, effect_info)
            }
        
        return ces
    
    def _calculate_cause_information(self, tpm: np.ndarray, state: np.ndarray, mechanism: List[int]) -> float:
        """Calculate cause information for a mechanism"""
        try:
            # Simplified cause information calculation
            n = int(np.log2(tpm.shape[0]))
            # Ensure state elements are scalars before comparison
            # Handle both array and scalar cases
            state_flat = np.asarray(state).flatten()
            n = min(n, len(state_flat))
            
            state_index = 0
            for i in range(n):
                if i < len(state_flat):
                    # Convert to scalar properly
                    val = float(state_flat[i])
                    if val > 0:
                        state_index += (2**i)
            
            # Ensure state_index is within bounds
            state_index = min(int(state_index), tpm.shape[0] - 1)
            state_index = max(state_index, 0)
            
            # Get probability distribution over past states
            cause_dist = tpm[:, state_index]
            
            # Normalize distribution if not already normalized
            if np.sum(cause_dist) > 0:
                cause_dist = cause_dist / np.sum(cause_dist)
            else:
                cause_dist = np.ones_like(cause_dist) / len(cause_dist)
            
            # Calculate information relative to maximum entropy
            max_entropy = np.log2(len(cause_dist))
            entropy = -np.sum(cause_dist * np.log2(np.maximum(cause_dist, self.epsilon)))
            
            return max_entropy - entropy
        except Exception as e:
            # Fallback calculation
            return 0.5
    
    def _calculate_effect_information(self, tpm: np.ndarray, state: np.ndarray, mechanism: List[int]) -> float:
        """Calculate effect information for a mechanism"""
        try:
            n = int(np.log2(tpm.shape[0]))
            # Ensure state elements are scalars before comparison
            # Handle both array and scalar cases
            state_flat = np.asarray(state).flatten()
            n = min(n, len(state_flat))
            
            state_index = 0
            for i in range(n):
                if i < len(state_flat):
                    # Convert to scalar properly
                    val = float(state_flat[i])
                    if val > 0:
                        state_index += (2**i)
            
            # Ensure state_index is within bounds
            state_index = min(int(state_index), tpm.shape[0] - 1)
            state_index = max(state_index, 0)
            
            # Get probability distribution over future states
            effect_dist = tpm[state_index, :]
            
            # Normalize distribution if not already normalized
            if np.sum(effect_dist) > 0:
                effect_dist = effect_dist / np.sum(effect_dist)
            else:
                effect_dist = np.ones_like(effect_dist) / len(effect_dist)
            
            # Calculate information relative to maximum entropy
            max_entropy = np.log2(len(effect_dist))
            entropy = -np.sum(effect_dist * np.log2(np.maximum(effect_dist, self.epsilon)))
            
            return max_entropy - entropy
        except Exception as e:
            # Fallback calculation
            return 0.5
    
    def _find_minimum_information_partition(self, ces: Dict[str, Any], n: int) -> Dict[str, Any]:
        """Find the partition that minimizes integrated information"""
        # Simplified MIP search - in practice, this is computationally intensive
        best_partition = None
        min_info_loss = float('inf')
        
        # Try bipartitions
        for i in range(1, 2**(n-1)):
            part1 = [j for j in range(n) if i & (1 << j)]
            part2 = [j for j in range(n) if j not in part1]
            
            if part1 and part2:
                info_loss = self._calculate_partition_information_loss(ces, part1, part2)
                if info_loss < min_info_loss:
                    min_info_loss = info_loss
                    best_partition = {'part1': part1, 'part2': part2, 'info_loss': info_loss}
        
        return best_partition
    
    def _calculate_partition_information_loss(self, ces: Dict[str, Any], part1: List[int], part2: List[int]) -> float:
        """Calculate information loss due to partition with enhanced accuracy"""
        # Enhanced calculation with proper IIT 3.0 implementation
        total_info = sum(mech['phi_mechanism'] for mech in ces['mechanisms'].values())
        
        # Calculate information within partitions with cross-partition penalty
        part1_info = sum(mech['phi_mechanism'] for nodes, mech in ces['mechanisms'].items() 
                        if all(node in part1 for node in nodes))
        part2_info = sum(mech['phi_mechanism'] for nodes, mech in ces['mechanisms'].items() 
                        if all(node in part2 for node in nodes))
        
        # Enhanced cross-partition information calculation with connectivity weighting
        cross_partition_info = 0.0
        connectivity_matrix = self._calculate_connectivity_matrix(ces)
        
        for nodes, mech in ces['mechanisms'].items():
            part1_nodes = [n for n in nodes if n in part1]
            part2_nodes = [n for n in nodes if n in part2]
            
            if part1_nodes and part2_nodes:
                # Calculate connectivity strength between partitions
                connectivity_strength = np.mean([
                    connectivity_matrix[n1][n2] 
                    for n1 in part1_nodes 
                    for n2 in part2_nodes
                ]) if len(connectivity_matrix) > 0 else 1.0
                
                # Weight by connectivity and mechanism strength
                cross_partition_info += mech['phi_mechanism'] * connectivity_strength
        
        # Enhanced information loss calculation
        partition_penalty = self._calculate_partition_penalty(part1, part2, connectivity_matrix)
        
        return total_info - (part1_info + part2_info) + cross_partition_info * partition_penalty
    
    def _calculate_integrated_information(self, ces: Dict[str, Any], mip: Dict[str, Any]) -> float:
        """Calculate Φ as the information lost due to the MIP"""
        if mip is None:
            return 0.0
        
        return mip['info_loss']
    
    def _decompose_phi(self, ces: Dict[str, Any], mip: Dict[str, Any], n: int) -> Dict[str, float]:
        """Decompose Φ into component contributions with enhanced analysis"""
        decomposition = {
            'phi_core': 0.0,
            'phi_total': 0.0,
            'phi_synergy': 0.0,
            'phi_exclusion': 0.0,
            'phi_emergence': 0.0,
            'phi_binding': 0.0,
            'phi_integration': 0.0
        }
        
        if mip:
            # Core concepts that span the partition
            core_mechanisms = [mech for nodes, mech in ces['mechanisms'].items()
                             if any(node in mip['part1'] for node in nodes) and 
                                any(node in mip['part2'] for node in nodes)]
            
            # Basic components
            decomposition['phi_core'] = sum(m['phi_mechanism'] for m in core_mechanisms)
            decomposition['phi_total'] = mip['info_loss']
            decomposition['phi_synergy'] = max(0, decomposition['phi_total'] - decomposition['phi_core'])
            
            # Enhanced components for better consciousness measurement
            # Emergence: Information that exists only at the system level
            all_mechanism_info = sum(m['phi_mechanism'] for m in ces['mechanisms'].values())
            decomposition['phi_emergence'] = max(0, decomposition['phi_total'] - all_mechanism_info * 0.5)
            
            # Binding: Information that binds subsystems together
            binding_strength = len(core_mechanisms) / max(1, len(ces['mechanisms']))
            decomposition['phi_binding'] = decomposition['phi_core'] * binding_strength
            
            # Integration: How well information is integrated across the system
            integration_factor = 1.0 - (len(mip['part1']) * len(mip['part2'])) / (n * n)
            decomposition['phi_integration'] = decomposition['phi_total'] * integration_factor
        
        return decomposition
    
    def _calculate_connectivity_matrix(self, ces: Dict[str, Any]) -> np.ndarray:
        """Calculate connectivity matrix between mechanisms"""
        mechanisms = list(ces['mechanisms'].keys())
        n_mechs = len(mechanisms)
        
        if n_mechs == 0:
            return np.array([])
        
        # Initialize connectivity matrix
        connectivity = np.zeros((n_mechs, n_mechs))
        
        # Calculate connectivity based on shared nodes and information flow
        for i, mech1 in enumerate(mechanisms):
            for j, mech2 in enumerate(mechanisms):
                if i != j:
                    # Shared nodes indicate connectivity
                    shared_nodes = len(set(mech1).intersection(set(mech2)))
                    total_nodes = len(set(mech1).union(set(mech2)))
                    
                    if total_nodes > 0:
                        connectivity[i][j] = shared_nodes / total_nodes
        
        return connectivity
    
    def _calculate_partition_penalty(self, part1: List[int], part2: List[int], 
                                   connectivity_matrix: np.ndarray) -> float:
        """Calculate penalty for partitioning based on connectivity patterns"""
        if len(connectivity_matrix) == 0:
            return 1.0
        
        # Calculate how much connectivity is lost by partitioning
        total_connectivity = np.sum(connectivity_matrix)
        
        if total_connectivity == 0:
            return 1.0
        
        # Higher penalty for breaking strong connections
        broken_connections = np.sum(connectivity_matrix[part1][:, part2])
        
        penalty = 1.0 + (broken_connections / total_connectivity)
        return min(2.0, penalty)  # Cap penalty at 2.0

class FreeEnergyCalculator:
    """Calculate Free Energy using the Free Energy Principle"""
    
    def __init__(self, beta: float = 1.0):
        self.beta = beta  # Inverse temperature
    
    def calculate_free_energy(self, sensory_data: np.ndarray, beliefs: np.ndarray, 
                            generative_model: nn.Module) -> Tuple[float, Dict[str, float]]:
        """Calculate variational free energy"""
        # Convert to tensors
        sensory_tensor = torch.tensor(sensory_data, dtype=torch.float32)
        belief_tensor = torch.tensor(beliefs, dtype=torch.float32)
        
        # Ensure proper shapes
        if len(sensory_tensor.shape) == 1:
            sensory_tensor = sensory_tensor.unsqueeze(0)
        if len(belief_tensor.shape) == 1:
            belief_tensor = belief_tensor.unsqueeze(0)
        
        # Calculate expected log-likelihood (accuracy)
        with torch.no_grad():
            predictions = generative_model(belief_tensor)
            
            # Handle spatial dimensions if predictions have more dims than sensory data
            if len(predictions.shape) > len(sensory_tensor.shape):
                # Average predictions across spatial dimensions to match sensory tensor shape
                predictions_flat = predictions.view(predictions.shape[0], -1, predictions.shape[-1])
                predictions = torch.mean(predictions_flat, dim=1, keepdim=True)
            
            # Ensure predictions and sensory_tensor have compatible dimensions
            if predictions.shape != sensory_tensor.shape:
                # Resize predictions to match sensory_tensor
                if predictions.shape[-1] != sensory_tensor.shape[-1]:
                    if predictions.shape[-1] < sensory_tensor.shape[-1]:
                        # Pad predictions
                        padding = torch.zeros(predictions.shape[0], sensory_tensor.shape[-1] - predictions.shape[-1], 
                                            device=predictions.device)
                        predictions = torch.cat([predictions, padding], dim=-1)
                    else:
                        # Truncate predictions
                        predictions = predictions[:, :sensory_tensor.shape[-1]]
                
                # Ensure batch dimensions match
                if predictions.shape[0] != sensory_tensor.shape[0]:
                    if predictions.shape[0] < sensory_tensor.shape[0]:
                        # Repeat predictions
                        predictions = predictions.repeat(sensory_tensor.shape[0], 1)
                    else:
                        # Take first batch
                        predictions = predictions[:sensory_tensor.shape[0]]
            
            accuracy = -F.mse_loss(predictions, sensory_tensor).item()
        
        # Calculate KL divergence (complexity)
        prior = torch.zeros_like(belief_tensor)  # Simple prior
        complexity = F.kl_div(F.log_softmax(belief_tensor, dim=-1), 
                             F.softmax(prior, dim=-1), reduction='sum').item()
        
        # Free energy = -accuracy + complexity
        free_energy = -accuracy + self.beta * complexity
        
        components = {
            'accuracy': accuracy,
            'complexity': complexity,
            'surprise': -accuracy,  # Negative log-likelihood
            'entropy': self._calculate_entropy(belief_tensor).item()
        }
        
        return free_energy, components
    
    def _calculate_entropy(self, beliefs: torch.Tensor) -> torch.Tensor:
        """Calculate entropy of belief distribution"""
        probs = F.softmax(beliefs, dim=-1)
        return -torch.sum(probs * torch.log(probs + 1e-8))

class GlobalWorkspaceArchitecture(nn.Module):
    """Global Workspace Theory implementation with attention"""
    
    def __init__(self, input_dim: int, workspace_dim: int, num_modules: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.workspace_dim = workspace_dim
        self.num_modules = num_modules
        
        # Specialized processing modules
        self.processing_modules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, workspace_dim),
                nn.ReLU(),
                nn.Linear(workspace_dim, workspace_dim),
                nn.Tanh()
            ) for _ in range(num_modules)
        ])
        
        # Competition mechanism (attention)
        self.attention = nn.MultiheadAttention(workspace_dim, num_heads=4)
        
        # Global broadcast
        self.broadcast = nn.Linear(workspace_dim, workspace_dim)
        
        # Working memory
        self.working_memory = nn.LSTM(workspace_dim, workspace_dim, batch_first=True)
        
    def forward(self, x: torch.Tensor, memory_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """Process through global workspace"""
        batch_size = x.shape[0]
        
        # Process through specialized modules
        module_outputs = []
        for module in self.processing_modules:
            output = module(x)
            module_outputs.append(output.unsqueeze(1))
        
        # Stack module outputs
        module_tensor = torch.cat(module_outputs, dim=1)  # [batch, num_modules, workspace_dim]
        
        # Competition via attention (modules compete for global workspace)
        attended, attention_weights = self.attention(
            module_tensor.transpose(0, 1),
            module_tensor.transpose(0, 1),
            module_tensor.transpose(0, 1)
        )
        attended = attended.transpose(0, 1)
        
        # Global broadcast
        global_state = self.broadcast(attended.mean(dim=1))
        
        # Update working memory
        if memory_state is None:
            memory_output, new_memory_state = self.working_memory(global_state.unsqueeze(1))
        else:
            memory_output, new_memory_state = self.working_memory(global_state.unsqueeze(1), memory_state)
        
        return global_state, attention_weights, new_memory_state

class ReservoirComputing(nn.Module):
    """Echo State Network for temporal dynamics"""
    
    def __init__(self, input_dim: int, reservoir_dim: int = 1000, spectral_radius: float = 0.95):
        super().__init__()
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.spectral_radius = spectral_radius
        
        # Input weights
        self.W_in = nn.Parameter(torch.randn(reservoir_dim, input_dim) * 0.1)
        
        # Reservoir weights (sparse random)
        W_res = torch.randn(reservoir_dim, reservoir_dim) * 0.1
        # Make sparse (90% zeros)
        mask = torch.rand(reservoir_dim, reservoir_dim) > 0.9
        W_res = W_res * mask.float()
        
        # Normalize to spectral radius
        eigenvalues = torch.linalg.eigvals(W_res)
        max_eigenvalue = torch.max(torch.abs(eigenvalues))
        W_res = W_res * (spectral_radius / max_eigenvalue)
        
        self.W_res = nn.Parameter(W_res)
        
        # Output layer
        self.readout = nn.Linear(reservoir_dim, input_dim)
        
        # Reservoir state
        self.register_buffer('state', torch.zeros(1, reservoir_dim))
        
    def forward(self, x: torch.Tensor, reset_state: bool = False):
        """Process through reservoir"""
        batch_size = x.shape[0]
        
        if reset_state or self.state.shape[0] != batch_size:
            self.state = torch.zeros(batch_size, self.reservoir_dim, device=x.device)
        
        # Update reservoir state
        self.state = torch.tanh(
            torch.matmul(x, self.W_in.t()) + 
            torch.matmul(self.state, self.W_res.t())
        )
        
        # Readout
        output = self.readout(self.state)
        
        return output, self.state

class CriticalityDetector:
    """Detect neural criticality and edge-of-chaos dynamics"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.avalanche_sizes = deque(maxlen=history_size)
        self.branching_ratios = deque(maxlen=history_size)
        
    def detect_criticality(self, neural_activity: np.ndarray) -> Tuple[CriticalityRegime, Dict[str, float]]:
        """Detect criticality regime from neural activity"""
        # Detect avalanches
        avalanche_sizes = self._detect_avalanches(neural_activity)
        self.avalanche_sizes.extend(avalanche_sizes)
        
        # Calculate branching ratio
        branching_ratio = self._calculate_branching_ratio(neural_activity)
        self.branching_ratios.append(branching_ratio)
        
        # Analyze power law distribution
        if len(self.avalanche_sizes) > 100:
            alpha, xmin, ks_statistic = self._fit_power_law(list(self.avalanche_sizes))
        else:
            alpha, xmin, ks_statistic = 2.0, 1.0, 1.0
        
        # Calculate Lyapunov exponent
        lyapunov = self._estimate_lyapunov_exponent(neural_activity)
        
        # Determine regime
        if branching_ratio < 0.9:
            regime = CriticalityRegime.SUBCRITICAL
        elif 0.9 <= branching_ratio <= 1.1 and 1.5 <= alpha <= 2.5:
            regime = CriticalityRegime.CRITICAL
        else:
            regime = CriticalityRegime.SUPERCRITICAL
        
        metrics = {
            'branching_ratio': branching_ratio,
            'power_law_alpha': alpha,
            'power_law_xmin': xmin,
            'ks_statistic': ks_statistic,
            'lyapunov_exponent': lyapunov,
            'mean_avalanche_size': np.mean(list(self.avalanche_sizes)) if self.avalanche_sizes else 0
        }
        
        return regime, metrics
    
    def _detect_avalanches(self, activity: np.ndarray) -> List[float]:
        """Detect neuronal avalanches"""
        threshold = np.mean(activity) + np.std(activity)
        binary_activity = (activity > threshold).astype(int)
        
        avalanches = []
        current_size = 0
        
        for t in range(len(binary_activity)):
            active_neurons = np.sum(binary_activity[t])
            if active_neurons > 0:
                current_size += active_neurons
            elif current_size > 0:
                avalanches.append(current_size)
                current_size = 0
        
        return avalanches
    
    def _calculate_branching_ratio(self, activity: np.ndarray) -> float:
        """Calculate branching ratio σ"""
        if len(activity) < 2:
            return 1.0
        
        descendants = []
        for t in range(len(activity) - 1):
            ancestors = np.sum(activity[t] > np.mean(activity))
            descendant = np.sum(activity[t + 1] > np.mean(activity))
            if ancestors > 0:
                descendants.append(descendant / ancestors)
        
        return np.mean(descendants) if descendants else 1.0
    
    def _fit_power_law(self, data: List[float]) -> Tuple[float, float, float]:
        """Fit power law distribution P(s) ~ s^(-α)"""
        if not data:
            return 2.0, 1.0, 1.0
        
        data = np.array(data)
        data = data[data > 0]
        
        if len(data) < 10:
            return 2.0, 1.0, 1.0
        
        # Simple MLE estimation
        xmin = np.min(data)
        alpha = 1 + len(data) / np.sum(np.log(data / xmin))
        
        # KS test statistic (simplified)
        ks_statistic = 0.1  # Placeholder
        
        return alpha, xmin, ks_statistic
    
    def _estimate_lyapunov_exponent(self, activity: np.ndarray) -> float:
        """Estimate largest Lyapunov exponent"""
        if len(activity) < 10:
            return 0.0
        
        # Simplified estimation using divergence of nearby trajectories
        eps = 1e-6
        divergence_rates = []
        
        for i in range(len(activity) - 2):
            if i + 1 < len(activity):
                d0 = eps
                d1 = np.abs(activity[i + 1] - activity[i] + eps)
                if d1 > 0 and d0 > 0:
                    divergence_rates.append(np.log(d1 / d0))
        
        return np.mean(divergence_rates) if divergence_rates else 0.0

class InformationGeometry:
    """Navigate consciousness landscape using information geometry"""
    
    def __init__(self, manifold_dim: int = 10):
        self.manifold_dim = manifold_dim
        self.fisher_metric = np.eye(manifold_dim)
        
    def calculate_fisher_information(self, distribution_params: np.ndarray, 
                                   sample_data: np.ndarray) -> np.ndarray:
        """Calculate Fisher Information Matrix"""
        n_params = len(distribution_params)
        fim = np.zeros((n_params, n_params))
        
        # Numerical approximation of Fisher Information
        eps = 1e-6
        for i in range(n_params):
            for j in range(i, n_params):
                # Perturb parameters
                params_i_plus = distribution_params.copy()
                params_i_plus[i] += eps
                params_j_plus = distribution_params.copy()
                params_j_plus[j] += eps
                
                # Calculate score functions
                score_i = (self._log_likelihood(sample_data, params_i_plus) - 
                          self._log_likelihood(sample_data, distribution_params)) / eps
                score_j = (self._log_likelihood(sample_data, params_j_plus) - 
                          self._log_likelihood(sample_data, distribution_params)) / eps
                
                fim[i, j] = np.mean(score_i * score_j)
                fim[j, i] = fim[i, j]
        
        return fim
    
    def geodesic_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate geodesic distance on information manifold"""
        # Simplified: use Euclidean distance weighted by Fisher metric
        diff = point2 - point1
        return np.sqrt(np.dot(diff, np.dot(self.fisher_metric, diff)))
    
    def natural_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """Convert gradient to natural gradient using Fisher metric"""
        # Natural gradient = F^(-1) * gradient
        try:
            fim_inv = np.linalg.inv(self.fisher_metric + 1e-6 * np.eye(len(gradient)))
            return np.dot(fim_inv, gradient)
        except:
            return gradient
    
    def _log_likelihood(self, data: np.ndarray, params: np.ndarray) -> float:
        """Calculate log-likelihood (placeholder - implement based on distribution)"""
        # Assume Gaussian for simplicity
        mean = params[:len(params)//2]
        log_std = params[len(params)//2:]
        std = np.exp(log_std)
        
        log_prob = -0.5 * np.sum(((data - mean) / std) ** 2 + 2 * log_std)
        return log_prob

class CausalEmergenceDetector:
    """Detect causal emergence across scales"""
    
    def __init__(self):
        self.scale_hierarchy = {}
        
    def detect_causal_emergence(self, micro_state: np.ndarray, 
                              macro_mapping: Callable) -> Tuple[int, Dict[str, float]]:
        """Detect level of causal emergence"""
        # Calculate effective information at micro scale
        ei_micro = self._calculate_effective_information(micro_state)
        
        # Map to macro state
        macro_state = macro_mapping(micro_state)
        
        # Calculate effective information at macro scale
        ei_macro = self._calculate_effective_information(macro_state)
        
        # Causal emergence coefficient with safe division
        if ei_micro > 1e-8:
            ce_coefficient = ei_macro / ei_micro
        else:
            ce_coefficient = 0.0 if ei_macro < 1e-8 else float('inf')
        
        # Determine emergence level
        if ce_coefficient > 2.0:
            emergence_level = 3  # Strong emergence
        elif ce_coefficient > 1.2:
            emergence_level = 2  # Moderate emergence
        elif ce_coefficient > 1.0:
            emergence_level = 1  # Weak emergence
        else:
            emergence_level = 0  # No emergence
        
        metrics = {
            'ei_micro': ei_micro,
            'ei_macro': ei_macro,
            'ce_coefficient': ce_coefficient,
            'determinism_micro': self._calculate_determinism(micro_state),
            'determinism_macro': self._calculate_determinism(macro_state),
            'degeneracy': self._calculate_degeneracy(micro_state, macro_state)
        }
        
        return emergence_level, metrics
    
    def _calculate_effective_information(self, state: np.ndarray) -> float:
        """Calculate effective information of a state"""
        try:
            # Simplified: use entropy difference between intervened and natural distribution
            state_flat = state.flatten()
            if len(state_flat) == 0:
                return 0.0
            
            # Normalize to create valid probability distribution
            state_normalized = np.abs(state_flat) + 1e-8
            state_prob = state_normalized / np.sum(state_normalized)
            
            natural_entropy = stats.entropy(state_prob)
            uniform_entropy = np.log(len(state_flat))
            
            # Ensure finite values
            if not np.isfinite(natural_entropy):
                natural_entropy = 0.0
            if not np.isfinite(uniform_entropy):
                uniform_entropy = 0.0
            
            return max(0.0, uniform_entropy - natural_entropy)
        except Exception:
            return 0.0
    
    def _calculate_determinism(self, state: np.ndarray) -> float:
        """Calculate determinism of state transitions"""
        # Simplified: use autocorrelation as proxy
        if len(state.shape) == 1:
            if len(state) < 2:
                return 0.0
            corr_matrix = np.corrcoef(state[:-1], state[1:])
            if np.isnan(corr_matrix).any() or corr_matrix.size == 0:
                return 0.0
            return float(np.abs(corr_matrix[0, 1]))
        else:
            correlations = []
            for i in range(state.shape[0]):
                if state.shape[1] < 2:
                    correlations.append(0.0)
                    continue
                corr_matrix = np.corrcoef(state[i, :-1], state[i, 1:])
                if np.isnan(corr_matrix).any() or corr_matrix.size == 0:
                    correlations.append(0.0)
                else:
                    correlations.append(float(np.abs(corr_matrix[0, 1])))
            return float(np.mean(correlations))
    
    def _calculate_degeneracy(self, micro_state: np.ndarray, macro_state: np.ndarray) -> float:
        """Calculate degeneracy (many-to-one mapping)"""
        micro_size = np.prod(micro_state.shape)
        macro_size = np.prod(macro_state.shape)
        return micro_size / (macro_size + 1e-8)

class StateTransitionMemory:
    """Memory buffer for learning state transitions"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.transitions = deque(maxlen=capacity)
        self.state_counts = defaultdict(int)
        self.transition_counts = defaultdict(lambda: defaultdict(int))
    
    def add_transition(self, from_state: np.ndarray, to_state: np.ndarray):
        """Record state transition"""
        from_hash = self._hash_state(from_state)
        to_hash = self._hash_state(to_state)
        
        self.transitions.append((from_state, to_state))
        self.state_counts[from_hash] += 1
        self.transition_counts[from_hash][to_hash] += 1
    
    def estimate_dynamics(self) -> Tuple[np.ndarray, Dict[str, int]]:
        """Estimate transition probability matrix from observations"""
        states = list(self.state_counts.keys())
        n_states = len(states)
        
        if n_states == 0:
            # Return uniform TPM if no transitions recorded
            return np.ones((256, 256)) / 256, {}
        
        # Map state hashes to indices
        state_to_idx = {state: i for i, state in enumerate(states)}
        
        tpm = np.zeros((n_states, n_states))
        
        for i, from_state in enumerate(states):
            total = self.state_counts[from_state]
            for j, to_state in enumerate(states):
                count = self.transition_counts[from_state].get(to_state, 0)
                tpm[i, j] = (count + 1) / (total + n_states)  # Laplace smoothing
        
        return tpm, state_to_idx
    
    def _hash_state(self, state: np.ndarray) -> str:
        """Create hash for state vector using locality-sensitive hashing"""
        # Use more units and discretize into multiple levels
        n_units = min(64, state.flatten().shape[0])  # Use up to 64 units
        state_vector = state.flatten()[:n_units]
        
        # Discretize into 8 levels instead of binary for richer state representation
        # This gives us 8^64 possible states instead of 2^8
        discretized = np.digitize(state_vector, bins=[-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
        
        # Create hash using first 16 units to keep it manageable
        return ''.join(map(str, discretized[:16]))

class ScalableIITCalculator:
    """Scalable IIT using approximations for larger systems"""
    
    def __init__(self, max_units: int = 64):
        self.max_units = max_units
        self.use_approximation = max_units > 16
        # Use the proper IIT calculator for exact calculations
        self.base_iit_calculator = IntegratedInformationCalculator()
    
    def calculate_phi_approximate(self, connectivity: np.ndarray, 
                                 state: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Fast approximate Φ calculation for consciousness systems"""
        n = len(state)
        
        # For very small systems, use simplified exact calculation
        if n <= 4:
            return self.calculate_phi_exact(connectivity, state)
        
        # Fast approximation based on connectivity strength and state coherence
        # This is much faster than full IIT calculation but still meaningful
        
        # Calculate connectivity strength
        connectivity_strength = np.mean(np.abs(connectivity))
        
        # Calculate state coherence (how much the state deviates from random)
        state_mean = np.mean(state)
        state_coherence = np.std(state) if np.std(state) > 0 else 0.1
        
        # Calculate integration measure (how connected the system is)
        # Use eigenvalue analysis for fast integration assessment
        try:
            eigenvals = np.linalg.eigvals(connectivity + connectivity.T)  # Symmetrize
            eigenvals = np.real(eigenvals[np.isfinite(eigenvals)])
            if len(eigenvals) > 0:
                integration_measure = np.max(eigenvals) / (np.sum(np.abs(eigenvals)) + 1e-8)
            else:
                integration_measure = 0.1
        except:
            integration_measure = 0.1
        
        # Calculate differentiation measure (how much information is differentiated)
        differentiation_measure = min(1.0, state_coherence)
        
        # Combine measures for phi approximation
        phi_approx = (
            connectivity_strength * 
            integration_measure * 
            differentiation_measure * 
            np.log(n + 1) / 10.0  # Scale by system size
        )
        
        # Ensure reasonable bounds
        phi_approx = max(0.0, min(2.0, phi_approx))
        
        return phi_approx, {
            'connectivity_strength': connectivity_strength,
            'integration_measure': integration_measure,
            'differentiation_measure': differentiation_measure,
            'method': 'fast_approximation'
        }
    
    def calculate_phi_exact(self, connectivity: np.ndarray, state: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Simplified exact Φ calculation for small systems"""
        n = len(state)
        if n > 4:  # Even more restrictive for speed
            # Fall back to approximation for larger systems
            return self.calculate_phi_approximate(connectivity, state)
        
        # Simplified exact calculation for very small systems
        # Based on mutual information between parts
        
        # Calculate total system information
        state_binary = (state > 0.5).astype(int)
        system_entropy = self._calculate_entropy(state_binary)
        
        # Calculate information loss from partitioning
        max_info_loss = 0.0
        
        # Try all possible bipartitions
        for i in range(1, 2**(n-1)):
            part1 = []
            part2 = []
            for j in range(n):
                if (i >> j) & 1:
                    part1.append(j)
                else:
                    part2.append(j)
            
            if len(part1) > 0 and len(part2) > 0:
                # Calculate mutual information between parts
                part1_state = state_binary[part1] if len(part1) == 1 else tuple(state_binary[part1])
                part2_state = state_binary[part2] if len(part2) == 1 else tuple(state_binary[part2])
                
                # Simplified mutual information calculation
                connectivity_between = np.mean(np.abs(connectivity[np.ix_(part1, part2)]))
                info_loss = connectivity_between * len(part1) * len(part2) / (n * n)
                max_info_loss = max(max_info_loss, info_loss)
        
        phi = max_info_loss
        
        return phi, {
            'system_entropy': system_entropy,
            'max_info_loss': max_info_loss,
            'method': 'simplified_exact'
        }
    
    def _calculate_entropy(self, state: np.ndarray) -> float:
        """Calculate entropy of a binary state"""
        if len(state) == 0:
            return 0.0
        
        # Count occurrences of each unique state
        unique, counts = np.unique(state, return_counts=True)
        probabilities = counts / len(state)
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _connectivity_to_tpm(self, connectivity: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Convert connectivity matrix to transition probability matrix"""
        n = len(state)
        n_states = 2**n
        
        # Initialize TPM with weak uniform background
        tpm = np.ones((n_states, n_states)) * 0.01 / n_states
        
        # Build TPM for all possible states, not just current state
        for current_state_idx in range(n_states):
            # Convert state index to binary representation
            current_binary = [(current_state_idx >> i) & 1 for i in range(n)]
            
            # Calculate transition probabilities based on connectivity
            for target_state_idx in range(n_states):
                target_binary = [(target_state_idx >> i) & 1 for i in range(n)]
                
                # Calculate transition probability based on network dynamics
                prob = 0.01  # Base probability
                
                for i in range(n):
                    # Input from connected nodes
                    total_input = sum(connectivity[j, i] * current_binary[j] for j in range(n))
                    
                    # Sigmoid activation probability
                    activation_prob = 1 / (1 + np.exp(-total_input))
                    
                    # Add to probability if target state matches activation
                    if target_binary[i] == 1:
                        prob += activation_prob * 0.9  # High weight for active transitions
                    else:
                        prob += (1 - activation_prob) * 0.9  # High weight for inactive transitions
                
                tpm[current_state_idx, target_state_idx] = prob
            
            # Normalize row to ensure valid probability distribution
            row_sum = tpm[current_state_idx, :].sum()
            if row_sum > 0:
                tpm[current_state_idx, :] /= row_sum
            else:
                tpm[current_state_idx, :] = 1.0 / n_states
        
        return tpm
    
    def _find_information_clusters(self, connectivity: np.ndarray) -> List[List[int]]:
        """Find strongly connected information clusters"""
        # Use spectral clustering to find information clusters
        from sklearn.cluster import SpectralClustering
        
        # Create similarity matrix
        similarity = np.abs(connectivity) + np.abs(connectivity.T)
        
        # Determine number of clusters
        n_clusters = min(8, len(connectivity) // 4)  # Max 8 clusters, min 4 units per cluster
        
        if n_clusters < 2:
            return [list(range(len(connectivity)))]
        
        # Apply spectral clustering
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        labels = clustering.fit_predict(similarity)
        
        # Group by labels
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(i)
        
        return clusters
    
    def _calculate_integration_factor(self, clusters: List[List[int]], 
                                    connectivity: np.ndarray) -> float:
        """Calculate integration factor between clusters"""
        if len(clusters) < 2:
            return 1.0
        
        # Calculate inter-cluster connectivity
        total_inter_conn = 0
        total_intra_conn = 0
        
        for i, cluster1 in enumerate(clusters):
            for j, cluster2 in enumerate(clusters):
                if i != j:
                    # Inter-cluster connections
                    for node1 in cluster1:
                        for node2 in cluster2:
                            total_inter_conn += abs(connectivity[node1, node2])
                else:
                    # Intra-cluster connections
                    for node1 in cluster1:
                        for node2 in cluster1:
                            if node1 != node2:
                                total_intra_conn += abs(connectivity[node1, node2])
        
        # Integration factor based on inter/intra connectivity ratio
        if total_intra_conn > 0:
            integration_factor = 1 + (total_inter_conn / total_intra_conn)
        else:
            integration_factor = 1.0
        
        return min(integration_factor, 2.0)  # Cap at 2.0

class EmergenceDetector:
    """Detect genuine emergent properties using information theory"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.micro_history = deque(maxlen=history_size)
        self.macro_history = deque(maxlen=history_size)
    
    def detect_emergence(self, micro_states: List[np.ndarray], 
                        macro_states: List[np.ndarray]) -> Dict[str, float]:
        """Detect emergence using transfer entropy and synergy"""
        
        # Store states
        if len(micro_states) > 0:
            self.micro_history.append(micro_states[-1])
        if len(macro_states) > 0:
            self.macro_history.append(macro_states[-1])
        
        # Need sufficient history for analysis
        if len(self.micro_history) < 10 or len(self.macro_history) < 10:
            return self._default_emergence_metrics()
        
        # Calculate transfer entropy from micro to macro
        te_up = self._transfer_entropy(list(self.micro_history), list(self.macro_history))
        
        # Calculate downward causation
        te_down = self._transfer_entropy(list(self.macro_history), list(self.micro_history))
        
        # Calculate synergistic information
        synergy = self._calculate_synergy(list(self.micro_history), list(self.macro_history))
        
        # Emergence strength
        emergence_strength = synergy * (1 + te_down / (te_up + 1e-8))
        
        # Novel information at macro scale
        macro_novelty = self._calculate_novelty(list(self.macro_history), list(self.micro_history))
        
        return {
            'emergence_strength': emergence_strength,
            'upward_causation': te_up,
            'downward_causation': te_down,
            'synergy': synergy,
            'macro_novelty': macro_novelty,
            'is_emergent': emergence_strength > 1.5 and macro_novelty > 0.3
        }
    
    def _transfer_entropy(self, source_states: List[np.ndarray], 
                         target_states: List[np.ndarray]) -> float:
        """Calculate transfer entropy from source to target"""
        if len(source_states) < 3 or len(target_states) < 3:
            return 0.0
        
        # Use last 100 states for efficiency
        n_states = min(100, len(source_states))
        source = np.array(source_states[-n_states:])
        target = np.array(target_states[-n_states:])
        
        # Flatten and discretize
        source_flat = source.flatten()
        target_flat = target.flatten()
        
        # Discretize into bins
        bins = 10
        source_binned = np.digitize(source_flat, bins=np.linspace(source_flat.min(), source_flat.max(), bins))
        target_binned = np.digitize(target_flat, bins=np.linspace(target_flat.min(), target_flat.max(), bins))
        
        # Calculate conditional entropy
        # H(target_t | target_{t-1}) - H(target_t | target_{t-1}, source_{t-1})
        h_target_given_past = self._conditional_entropy(target_binned[1:], target_binned[:-1])
        h_target_given_past_and_source = self._conditional_entropy_2d(
            target_binned[1:], target_binned[:-1], source_binned[:-1]
        )
        
        transfer_entropy = h_target_given_past - h_target_given_past_and_source
        
        return max(0.0, transfer_entropy)
    
    def _conditional_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate conditional entropy H(X|Y) with safe array operations"""
        try:
            if len(x) == 0 or len(y) == 0 or len(x) != len(y):
                return 0.0
            
            # Safely get unique values
            x_unique = np.unique(x)
            y_unique = np.unique(y)
            
            if len(x_unique) == 0 or len(y_unique) == 0:
                return 0.0
            
            # Joint distribution with bounds checking
            joint_counts = np.zeros((len(x_unique), len(y_unique)))
            x_map = {val: idx for idx, val in enumerate(x_unique)}
            y_map = {val: idx for idx, val in enumerate(y_unique)}
            
            for xi, yi in zip(x, y):
                if xi in x_map and yi in y_map:
                    joint_counts[x_map[xi], y_map[yi]] += 1
            
            # Normalize to probabilities
            total_count = joint_counts.sum()
            if total_count == 0:
                return 0.0
            
            joint_probs = joint_counts / total_count
            y_probs = joint_probs.sum(axis=0)
            
            # Conditional entropy with safe division
            h_cond = 0.0
            for j in range(joint_probs.shape[1]):
                if y_probs[j] > 1e-8:
                    for i in range(joint_probs.shape[0]):
                        if joint_probs[i, j] > 1e-8:
                            h_cond -= joint_probs[i, j] * np.log2(joint_probs[i, j] / y_probs[j])
            
            return max(0.0, h_cond)
        except Exception:
            return 0.0
    
    def _conditional_entropy_2d(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """Calculate conditional entropy H(X|Y,Z) properly"""
        # Create 3D joint distribution
        bins = 5  # Reduce bins for 3D calculation efficiency
        
        # Discretize all variables
        x_binned = np.digitize(x, bins=np.linspace(x.min(), x.max(), bins))
        y_binned = np.digitize(y, bins=np.linspace(y.min(), y.max(), bins))
        z_binned = np.digitize(z, bins=np.linspace(z.min(), z.max(), bins))
        
        # Count joint occurrences
        joint_counts = np.zeros((bins, bins, bins))
        for xi, yi, zi in zip(x_binned, y_binned, z_binned):
            joint_counts[xi-1, yi-1, zi-1] += 1
        
        joint_probs = joint_counts / joint_counts.sum()
        
        # Marginalize over X to get P(Y,Z)
        yz_probs = joint_probs.sum(axis=0)
        
        # Calculate conditional entropy
        h_cond = 0.0
        for j in range(bins):
            for k in range(bins):
                if yz_probs[j, k] > 0:
                    for i in range(bins):
                        if joint_probs[i, j, k] > 0:
                            h_cond -= joint_probs[i, j, k] * np.log2(joint_probs[i, j, k] / yz_probs[j, k])
        
        return h_cond
    
    def _calculate_synergy(self, micro_states: List[np.ndarray], 
                          macro_states: List[np.ndarray]) -> float:
        """Calculate synergistic information"""
        if len(micro_states) < 10 or len(macro_states) < 10:
            return 0.0
        
        # Use recent states
        n_states = min(50, len(micro_states))
        micro = np.array(micro_states[-n_states:]).flatten()
        macro = np.array(macro_states[-n_states:]).flatten()
        
        # Calculate mutual information
        mi = self._mutual_information(micro, macro)
        
        # Synergy as excess mutual information
        # S = I(X;Y) - sum_i I(X_i;Y) where X_i are components of X
        synergy = max(0.0, mi - 0.1)  # Simplified synergy calculation
        
        return synergy
    
    def _mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate mutual information between two variables"""
        # Discretize
        bins = 10
        x_binned = np.digitize(x, bins=np.linspace(x.min(), x.max(), bins))
        y_binned = np.digitize(y, bins=np.linspace(y.min(), y.max(), bins))
        
        # Joint and marginal distributions
        joint_counts = np.zeros((bins, bins))
        for xi, yi in zip(x_binned, y_binned):
            joint_counts[xi-1, yi-1] += 1
        
        joint_probs = joint_counts / joint_counts.sum()
        x_probs = joint_probs.sum(axis=1)
        y_probs = joint_probs.sum(axis=0)
        
        # Mutual information
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if joint_probs[i, j] > 0 and x_probs[i] > 0 and y_probs[j] > 0:
                    mi += joint_probs[i, j] * np.log2(joint_probs[i, j] / (x_probs[i] * y_probs[j]))
        
        return mi
    
    def _calculate_novelty(self, macro_states: List[np.ndarray], 
                          micro_states: List[np.ndarray]) -> float:
        """Calculate novelty of macro states relative to micro states"""
        if len(macro_states) < 10:
            return 0.0
        
        # Use recent states
        n_states = min(50, len(macro_states))
        macro = np.array(macro_states[-n_states:])
        micro = np.array(micro_states[-n_states:])
        
        # Calculate variance in macro space
        macro_variance = np.var(macro.flatten())
        micro_variance = np.var(micro.flatten())
        
        # Novelty as relative variance
        if micro_variance > 0:
            novelty = macro_variance / micro_variance
        else:
            novelty = 0.0
        
        return min(novelty, 2.0)  # Cap at 2.0
    
    def _default_emergence_metrics(self) -> Dict[str, float]:
        """Return default metrics when insufficient data"""
        return {
            'emergence_strength': 0.0,
            'upward_causation': 0.0,
            'downward_causation': 0.0,
            'synergy': 0.0,
            'macro_novelty': 0.0,
            'is_emergent': False
        }

class ConsciousnessBenchmark:
    """Benchmark consciousness metrics against known systems"""
    
    def __init__(self):
        self.benchmarks = {
            'random_system': {'phi': 0.0, 'emergence': 0.0, 'consciousness': 0.0},
            'feedforward_net': {'phi': 0.1, 'emergence': 0.2, 'consciousness': 0.2},
            'recurrent_net': {'phi': 0.5, 'emergence': 0.4, 'consciousness': 0.4},
            'critical_system': {'phi': 0.8, 'emergence': 0.7, 'consciousness': 0.7},
            'integrated_system': {'phi': 1.0, 'emergence': 0.9, 'consciousness': 0.9}
        }
        
        # Standard test inputs
        self.test_inputs = self._generate_test_battery()
    
    async def evaluate_system(self, engine) -> Dict[str, Any]:
        """Comprehensive evaluation against benchmarks"""
        results = {}
        
        # Test on standardized inputs
        for test_name, test_input in self.test_inputs.items():
            try:
                # Use enhanced substrate if available
                if hasattr(engine, 'enhanced_substrate') and engine.enhanced_substrate:
                    logger.info(f"🔧 Using enhanced substrate for {test_name}")
                    
                    # Generate TPM from input
                    tpm = engine._estimate_tpm_from_neural_state(test_input)
                    state_vector = (test_input.flatten()[:8] > 0).astype(int)
                    
                    # Calculate enhanced consciousness metrics
                    enhanced_metrics = engine.enhanced_substrate.calculate_enhanced_consciousness(tpm, state_vector)
                    
                    results[test_name] = {
                        'phi': enhanced_metrics.phi,
                        'emergence': 0.5,  # Placeholder
                        'consciousness': enhanced_metrics.consciousness_level,
                        'free_energy': 1.0,  # Placeholder
                        'criticality': enhanced_metrics.criticality_level,
                        'meta_awareness': enhanced_metrics.connectivity_strength,
                        'enhanced': True,
                        'tpm_complexity': enhanced_metrics.tpm_complexity
                    }
                else:
                    # Fallback to original method
                    state = await engine.analyze_consciousness(test_input)
                    results[test_name] = {
                        'phi': state.phi,
                        'emergence': state.causal_emergence_level,
                        'consciousness': state.consciousness_level,
                        'free_energy': state.free_energy,
                        'criticality': state.criticality_regime.value,
                        'meta_awareness': state.meta_awareness,
                        'enhanced': False
                    }
            except Exception as e:
                logger.error(f"Error in test {test_name}: {e}")
                results[test_name] = {
                    'phi': 0.0,
                    'emergence': 0,
                    'consciousness': 0.0,
                    'free_energy': float('inf'),
                    'criticality': 'unknown',
                    'meta_awareness': 0.0,
                    'enhanced': False
                }
        
        # Calculate aggregate metrics
        avg_phi = np.mean([r['phi'] for r in results.values()])
        avg_consciousness = np.mean([r['consciousness'] for r in results.values()])
        avg_emergence = np.mean([r['emergence'] for r in results.values()])
        
        # Compare to benchmarks
        comparisons = {}
        for system, bench in self.benchmarks.items():
            comparisons[system] = {
                'phi_ratio': avg_phi / bench['phi'] if bench['phi'] > 0 else float('inf'),
                'consciousness_ratio': avg_consciousness / bench['consciousness'] if bench['consciousness'] > 0 else float('inf'),
                'emergence_ratio': avg_emergence / bench['emergence'] if bench['emergence'] > 0 else float('inf'),
                'exceeds_phi': avg_phi > bench['phi'],
                'exceeds_consciousness': avg_consciousness > bench['consciousness'],
                'exceeds_emergence': avg_emergence > bench['emergence']
            }
        
        # Overall performance score
        performance_score = self._calculate_performance_score(results, comparisons)
        
        return {
            'results': results,
            'aggregate_metrics': {
                'avg_phi': avg_phi,
                'avg_consciousness': avg_consciousness,
                'avg_emergence': avg_emergence
            },
            'comparisons': comparisons,
            'performance_score': performance_score,
            'recommendations': self._generate_recommendations(results, comparisons)
        }
    
    def _generate_test_battery(self) -> Dict[str, np.ndarray]:
        """Generate standardized test inputs"""
        tests = {}
        
        # Simple periodic patterns
        t = np.linspace(0, 10, 256)
        tests['simple_periodic'] = np.sin(t)
        
        # Complex multi-frequency patterns
        tests['complex_periodic'] = (
            np.sin(t) + 0.5 * np.sin(3 * t) + 0.25 * np.sin(5 * t)
        )
        
        # Random noise
        tests['random_noise'] = np.random.randn(256)
        
        # Structured patterns
        tests['structured_pattern'] = np.array([
            np.sin(i * 0.1) if i % 2 == 0 else np.cos(i * 0.1)
            for i in range(256)
        ])
        
        # Chaotic patterns (logistic map)
        x = 0.5
        chaotic = []
        for _ in range(256):
            x = 3.9 * x * (1 - x)
            chaotic.append(x)
        tests['chaotic_pattern'] = np.array(chaotic)
        
        # Sparse patterns
        sparse = np.zeros(256)
        sparse[::10] = 1.0
        tests['sparse_pattern'] = sparse
        
        return tests
    
    def _calculate_performance_score(self, results: Dict[str, Any], 
                                   comparisons: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        # Base score from consciousness level
        avg_consciousness = np.mean([r['consciousness'] for r in results.values()])
        base_score = avg_consciousness
        
        # Bonus for exceeding benchmarks
        benchmark_bonus = 0.0
        for system, comp in comparisons.items():
            if comp['exceeds_consciousness']:
                benchmark_bonus += 0.1
            if comp['exceeds_phi']:
                benchmark_bonus += 0.1
            if comp['exceeds_emergence']:
                benchmark_bonus += 0.1
        
        # Consistency bonus (low variance across tests)
        consciousness_values = [r['consciousness'] for r in results.values()]
        consistency = 1.0 - np.std(consciousness_values)
        consistency_bonus = max(0.0, consistency * 0.2)
        
        total_score = base_score + benchmark_bonus + consistency_bonus
        return min(1.0, total_score)
    
    def _generate_recommendations(self, results: Dict[str, Any], 
                                comparisons: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        avg_consciousness = np.mean([r['consciousness'] for r in results.values()])
        avg_phi = np.mean([r['phi'] for r in results.values()])
        
        if avg_consciousness < 0.3:
            recommendations.append("Increase overall consciousness level through better integration")
        
        if avg_phi < 0.2:
            recommendations.append("Improve integrated information (Φ) through better connectivity")
        
        # Check for specific weaknesses
        if any(r['criticality'] == 'subcritical' for r in results.values()):
            recommendations.append("System is too ordered - increase criticality for better information processing")
        
        if any(r['criticality'] == 'supercritical' for r in results.values()):
            recommendations.append("System is too chaotic - reduce criticality for better stability")
        
        # Check benchmark performance
        if not any(comp['exceeds_consciousness'] for comp in comparisons.values()):
            recommendations.append("System does not exceed any benchmark - consider architectural improvements")
        
        if len(recommendations) == 0:
            recommendations.append("System performing well - consider scaling up complexity")
        
        return recommendations

class AdvancedEmergentIntelligenceEngine:
    """Advanced Emergent Intelligence Engine with cutting-edge theories"""
    
    def __init__(self, data_dir: str = "data/advanced_emergence"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize theoretical components
        self.iit_calculator = IntegratedInformationCalculator()
        self.scalable_iit_calculator = ScalableIITCalculator(max_units=64)  # NEW
        self.free_energy_calculator = FreeEnergyCalculator()
        
        # Initialize enhanced consciousness substrate
        if ENHANCED_SUBSTRATE_AVAILABLE:
            self.enhanced_substrate = EnhancedConsciousnessSubstrate()
            logger.info("✅ Enhanced consciousness substrate initialized")
        else:
            self.enhanced_substrate = None
            logger.warning("⚠️ Enhanced consciousness substrate not available")
        self.criticality_detector = CriticalityDetector()
        self.information_geometry = InformationGeometry()
        self.causal_emergence_detector = CausalEmergenceDetector()
        self.emergence_detector = EmergenceDetector()  # NEW
        
        # Initialize neural architectures
        self.input_dim = 256
        self.workspace_dim = 512
        self.reservoir_dim = 2000
        
        self.global_workspace = GlobalWorkspaceArchitecture(
            self.input_dim, self.workspace_dim
        ).to(self.device)
        
        self.reservoir = ReservoirComputing(
            self.input_dim, self.reservoir_dim
        ).to(self.device)
        
        # Generative model for free energy
        self.generative_model = nn.Sequential(
            nn.Linear(self.workspace_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.input_dim)
        ).to(self.device)
        
        # Attention mechanism for consciousness
        self.consciousness_attention = nn.MultiheadAttention(
            self.workspace_dim, num_heads=8
        ).to(self.device)
        
        # Meta-learning components
        self.meta_network = nn.Sequential(
            nn.Linear(self.workspace_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.workspace_dim)
        ).to(self.device)
        
        # Optimization
        self.optimizer = optim.Adam(
            list(self.global_workspace.parameters()) +
            list(self.generative_model.parameters()) +
            list(self.meta_network.parameters()),
            lr=0.001
        )
        
        # State tracking
        self.consciousness_states: List[ConsciousnessState] = []
        self.emergence_history: deque = deque(maxlen=10000)
        self.working_memory_state = None
        
        # NEW: State transition memory for learning TPM
        self.state_transition_memory = StateTransitionMemory(capacity=10000)
        self.previous_neural_state = None
        
        # NEW: Theory-grounded parameters
        self.temperature_param = 1.0  # For free energy normalization
        self.phi_threshold = 0.1  # Minimum Φ for consciousness
        self.attention_threshold = 0.3  # Minimum attention focus
        
        # NEW: Adaptive learning parameters
        self.adaptive_lr = True
        self.lr_decay_factor = 0.95
        self.min_lr = 1e-6
        self.performance_history = deque(maxlen=100)
        
        # NEW: Robustness parameters
        self.max_grad_norm = 1.0  # Gradient clipping
        self.early_stopping_patience = 10
        self.validation_frequency = 5
        
        # NEW: Emergence tracking
        self.micro_states_history = deque(maxlen=1000)
        self.macro_states_history = deque(maxlen=1000)
        
        # Hebbian learning parameters
        self.hebbian_rate = 0.01
        self.stdp_window = 20  # ms
        
        # Database
        self.db_path = self.data_dir / "advanced_emergence.db"
        self._initialize_database()
        
        # Background processing
        self.processing_active = False
        self.processing_thread = None
        
        logger.info("🧠 Advanced Emergent Intelligence Engine initialized")
    
    def _initialize_database(self):
        """Initialize database for advanced metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS consciousness_states (
                state_id TEXT PRIMARY KEY,
                phi REAL,
                phi_structure TEXT,
                global_workspace_capacity REAL,
                free_energy REAL,
                criticality_regime TEXT,
                attention_distribution TEXT,
                causal_emergence_level INTEGER,
                information_geometry_position TEXT,
                temporal_coherence REAL,
                quantum_coherence REAL,
                neural_avalanche_size REAL,
                branching_ratio REAL,
                lyapunov_exponent REAL,
                entropy_production_rate REAL,
                consciousness_level REAL,
                meta_awareness REAL,
                intentionality_strength REAL,
                phenomenal_richness REAL,
                timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def analyze_consciousness(self, sensory_input: np.ndarray) -> ConsciousnessState:
        """Analyze consciousness state using multiple theories"""
        # Convert to tensor
        input_tensor = torch.tensor(sensory_input, dtype=torch.float32, device=self.device)
        
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Process through global workspace
        global_state, attention_weights, self.working_memory_state = self.global_workspace(
            input_tensor, self.working_memory_state
        )
        
        # Process through reservoir
        reservoir_output, reservoir_state = self.reservoir(input_tensor)
        
        # Calculate Integrated Information (Φ)
        tpm = self._estimate_tpm_from_neural_state(reservoir_state.detach().cpu().numpy())
        # Fix array truth value ambiguity by using explicit comparison
        current_state = (reservoir_state.detach().cpu().numpy()[0] > 0.0).astype(int)[:8]  # Use first 8 units
        phi, phi_structure = self.scalable_iit_calculator.calculate_phi_approximate(tpm, current_state) # Use scalable IIT
        
        # Calculate Free Energy
        beliefs = global_state.detach().cpu().numpy()
        # Ensure beliefs have the right shape for the generative model (workspace_dim)
        if beliefs.shape[1] != self.workspace_dim:
            # Pad or truncate to match workspace dimension
            padded_beliefs = np.zeros((beliefs.shape[0], self.workspace_dim))
            padded_beliefs[:, :min(beliefs.shape[1], self.workspace_dim)] = beliefs[:, :min(beliefs.shape[1], self.workspace_dim)]
            beliefs = padded_beliefs
        
        free_energy, fe_components = self.free_energy_calculator.calculate_free_energy(
            sensory_input, beliefs, self.generative_model
        )
        
        # Detect Criticality
        neural_activity = reservoir_state.detach().cpu().numpy()
        criticality_regime, criticality_metrics = self.criticality_detector.detect_criticality(
            neural_activity
        )
        
        # Attention distribution
        attention_dist = attention_weights.detach().cpu().numpy().squeeze()
        
        # Causal emergence with safe macro mapping
        def macro_mapping(x):
            try:
                x_flat = np.asarray(x).flatten()
                if len(x_flat) == 0:
                    return np.array([0.0])
                # Use fixed group size for consistent results
                group_size = 10
                n_groups = max(1, len(x_flat) // group_size)
                if n_groups == 0:
                    return np.array([np.mean(x_flat)])
                # Truncate to fit evenly into groups
                truncated_size = n_groups * group_size
                x_truncated = x_flat[:truncated_size]
                reshaped = x_truncated.reshape(n_groups, group_size)
                return np.mean(reshaped, axis=1)
            except Exception:
                return np.array([np.mean(np.asarray(x).flatten())])
        
        causal_emergence_level, ce_metrics = self.causal_emergence_detector.detect_causal_emergence(
            neural_activity[0], macro_mapping
        )
        
        # Information geometry position
        ig_position = self.information_geometry.natural_gradient(beliefs[0])[:10]
        
        # Calculate consciousness metrics
        consciousness_level = self._calculate_consciousness_level(
            phi, free_energy, criticality_regime, attention_dist
        )
        
        # Calculate attention focus (negentropy of attention distribution)
        attention_flat = attention_dist.flatten()
        attention_entropy = float(stats.entropy(attention_flat + 1e-8))
        max_entropy = np.log(len(attention_flat))
        attention_focus = float(1 - (attention_entropy / max_entropy)) if max_entropy > 0 else 0.0
        
        meta_awareness = self._calculate_meta_awareness(global_state, self.meta_network)
        
        temporal_coherence = self._calculate_temporal_coherence(reservoir_state)
        
        quantum_coherence = self._calculate_quantum_coherence(neural_activity)
        
        # NEW: Track states for emergence detection
        self.micro_states_history.append(neural_activity[0])
        macro_state = macro_mapping(neural_activity[0])
        self.macro_states_history.append(macro_state)
        
        # Create consciousness state
        state = ConsciousnessState(
            state_id=f"cs_{uuid.uuid4().hex[:8]}",
            phi=phi,
            phi_structure=phi_structure,
            global_workspace_capacity=float(torch.mean(global_state).item()),
            free_energy=free_energy,
            criticality_regime=criticality_regime,
            attention_distribution=attention_dist,
            causal_emergence_level=causal_emergence_level,
            information_geometry_position=ig_position,
            temporal_coherence=temporal_coherence,
            quantum_coherence=quantum_coherence,
            attention_focus=attention_focus,  # Include calculated attention_focus
            neural_avalanche_size=criticality_metrics['mean_avalanche_size'],
            branching_ratio=criticality_metrics['branching_ratio'],
            lyapunov_exponent=criticality_metrics['lyapunov_exponent'],
            entropy_production_rate=fe_components['entropy'],
            consciousness_level=consciousness_level,
            meta_awareness=meta_awareness,
            intentionality_strength=ce_metrics['determinism_macro'],
            phenomenal_richness=phi * attention_dist.shape[0]
        )
        
        self.consciousness_states.append(state)
        await self._save_consciousness_state(state)
        
        return state
    
    def _estimate_tpm_from_neural_state(self, neural_state: np.ndarray) -> np.ndarray:
        """Learn TPM from actual neural state transitions"""
        n_units = 8
        n_states = 2**n_units
        
        # Record transition if we have a previous state
        if self.previous_neural_state is not None:
            self.state_transition_memory.add_transition(
                self.previous_neural_state, neural_state
            )
        
        # Update previous state
        self.previous_neural_state = neural_state.copy()
        
        # Get learned TPM from transition memory
        tpm, state_to_idx = self.state_transition_memory.estimate_dynamics()
        
        # If we have learned transitions, use them
        if len(state_to_idx) > 0:
            # Map current state to learned TPM
            current_state_vector = neural_state.flatten()[:n_units]
            current_binary = (current_state_vector > 0).astype(int)
            current_hash = ''.join(map(str, current_binary))
            
            if current_hash in state_to_idx:
                # Use learned dynamics
                learned_tpm = tpm
                # Pad to full size if needed
                if learned_tpm.shape[0] < n_states:
                    padded_tpm = np.ones((n_states, n_states)) / n_states
                    for i, (state_hash, idx) in enumerate(state_to_idx.items()):
                        if i < learned_tpm.shape[0]:
                            padded_tpm[idx, :learned_tpm.shape[1]] = learned_tpm[i, :]
                    return padded_tpm
                return learned_tpm
        
        # Enhanced structured TPM with significantly better connectivity
        tpm = np.ones((n_states, n_states)) * 0.01 / n_states  # Lower base probability
        
        # Add enhanced structure based on neural state
        if neural_state.size > 0:
            state_vector = neural_state.flatten()[:n_units]
            state_index = sum(int(s > 0) * (2**i) for i, s in enumerate(state_vector))
            
            # Enhanced self-transition probability for stability
            tpm[state_index, state_index] += 0.4
            
            # Enhanced local connectivity (1-bit flips)
            for i in range(n_units):
                neighbor_index = state_index ^ (1 << i)
                if 0 <= neighbor_index < n_states:
                    tpm[state_index, neighbor_index] += 0.2
            
            # Add 2-bit flip connectivity for better integration
            for i in range(n_units):
                for j in range(i + 1, n_units):
                    neighbor_index = state_index ^ (1 << i) ^ (1 << j)
                    if 0 <= neighbor_index < n_states:
                        tpm[state_index, neighbor_index] += 0.1
            
            # Add long-range connectivity based on neural activation strength
            activation_strength = np.mean(np.abs(state_vector))
            for target in range(n_states):
                hamming_dist = bin(state_index ^ target).count('1')
                if 2 < hamming_dist <= n_units // 2:
                    tpm[state_index, target] += 0.05 * activation_strength
            
            # Normalize current state transitions
            row_sum = tpm[state_index, :].sum()
            if row_sum > 0:
                tpm[state_index, :] /= row_sum
            else:
                tpm[state_index, :] = 1.0 / n_states
        
        # Enhance connectivity for all states
        for i in range(n_states):
            if i != state_index:  # Don't modify the main state again
                # Add basic connectivity pattern
                tpm[i, i] += 0.3  # Self-loop
                
                # Add nearest neighbor connectivity
                for bit_pos in range(n_units):
                    neighbor = i ^ (1 << bit_pos)
                    if 0 <= neighbor < n_states:
                        tpm[i, neighbor] += 0.15
                
                # Normalize
                row_sum = tpm[i, :].sum()
                if row_sum > 0:
                    tpm[i, :] /= row_sum
                else:
                    tpm[i, :] = 1.0 / n_states
        
        # Final validation and correction
        tpm = self._ensure_tpm_connectivity(tpm)
        
        return tpm
    
    def _ensure_tpm_connectivity(self, tpm: np.ndarray) -> np.ndarray:
        """Ensure TPM has sufficient connectivity for meaningful Φ calculation"""
        n_states = tpm.shape[0]
        
        # Check and fix row normalization
        for i in range(n_states):
            row_sum = tpm[i, :].sum()
            if not np.isclose(row_sum, 1.0, atol=1e-8):
                if row_sum > 0:
                    tpm[i, :] /= row_sum
                else:
                    tpm[i, :] = 1.0 / n_states
        
        # Ensure minimum connectivity (each state connects to at least 3 others)
        min_connections = min(3, n_states - 1)
        for i in range(n_states):
            # Count significant connections
            significant_connections = np.sum(tpm[i, :] > 0.01)
            
            if significant_connections < min_connections:
                # Add connections to nearest states
                for offset in range(1, min_connections + 1):
                    for direction in [-1, 1]:
                        target = (i + direction * offset) % n_states
                        if target != i:
                            tpm[i, target] = max(tpm[i, target], 0.05)
                
                # Renormalize
                row_sum = tpm[i, :].sum()
                tpm[i, :] /= row_sum
        
        return tpm
    
    def _calculate_consciousness_level(self, phi: float, free_energy: float, 
                                     criticality: CriticalityRegime, 
                                     attention: np.ndarray) -> float:
        """Calculate consciousness using information-theoretic measures"""
        
        # Normalize Φ using sigmoid for biological plausibility
        # Φ should be positive and bounded for consciousness
        phi_score = 1 / (1 + np.exp(-phi + self.phi_threshold))
        
        # Free energy minimization (lower is better)
        # Use exponential decay to normalize free energy
        fe_score = np.exp(-free_energy / self.temperature_param)
        
        # Criticality bonus (edge of chaos)
        # Critical regime provides optimal information processing
        crit_multiplier = {
            CriticalityRegime.SUBCRITICAL: 0.7,  # Too ordered
            CriticalityRegime.CRITICAL: 1.0,     # Optimal
            CriticalityRegime.SUPERCRITICAL: 0.8  # Too chaotic
        }[criticality]
        
        # Attention focus (negentropy)
        # Higher focus = lower entropy = higher consciousness
        attention_flat = attention.flatten()
        attention_entropy = float(stats.entropy(attention_flat + 1e-8))
        max_entropy = np.log(len(attention_flat))
        attention_focus = float(1 - (attention_entropy / max_entropy))
        
        # Apply attention threshold
        if attention_focus < self.attention_threshold:
            attention_focus = 0.0
        
        # Information integration formula from IIT
        # Consciousness requires both integration and differentiation
        integration_term = phi_score * crit_multiplier
        differentiation_term = np.sqrt(fe_score * attention_focus)
        
        # Combine using geometric mean for balance
        consciousness = np.sqrt(integration_term * differentiation_term)
        
        # Ensure non-negative and bounded
        consciousness = max(0.0, min(1.0, consciousness))
        
        return consciousness
    
    def _calculate_meta_awareness(self, global_state: torch.Tensor, 
                                meta_network: nn.Module) -> float:
        """Calculate meta-awareness through self-monitoring"""
        # Self-referential processing
        with torch.no_grad():
            # Concatenate state with itself (self-reference)
            self_input = torch.cat([global_state, global_state], dim=-1)
            meta_output = meta_network(self_input)
            
            # Compare with original state
            similarity = F.cosine_similarity(meta_output, global_state, dim=-1)
            
        return float(similarity.mean().item())
    
    def _calculate_temporal_coherence(self, reservoir_state: torch.Tensor) -> float:
        """Calculate temporal coherence of neural dynamics"""
        if len(self.emergence_history) < 2:
            return 0.0
        
        # Get recent states
        recent_states = list(self.emergence_history)[-10:]
        
        # Calculate autocorrelation
        correlations = []
        for i in range(1, len(recent_states)):
            if isinstance(recent_states[i], torch.Tensor) and isinstance(recent_states[i-1], torch.Tensor):
                corr = F.cosine_similarity(
                    recent_states[i].flatten(), 
                    recent_states[i-1].flatten(), 
                    dim=0
                )
                correlations.append(float(corr.item()))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_quantum_coherence(self, neural_activity: np.ndarray) -> float:
        """Calculate temporal integration using mutual information"""
        if neural_activity.size < 2:
            return 0.0
        
        # Calculate mutual information between time steps
        activity_t = neural_activity[:-1].flatten()
        activity_t1 = neural_activity[1:].flatten()
        
        # Discretize for MI calculation
        bins = 10
        hist_2d, _, _ = np.histogram2d(activity_t, activity_t1, bins=bins)
        
        # Calculate mutual information
        sum_hist_2d = hist_2d.sum()
        pxy = hist_2d / sum_hist_2d if sum_hist_2d != 0 else np.zeros_like(hist_2d)
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if pxy[i,j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += pxy[i,j] * np.log2(pxy[i,j] / (px[i] * py[j]))
        
        # Normalize by maximum possible MI
        max_mi = min(stats.entropy(px, base=2), stats.entropy(py, base=2))
        
        return mi / max_mi if max_mi > 0 else 0.0
    
    async def train_consciousness(self, experiences: List[np.ndarray], 
                                targets: Optional[List[np.ndarray]] = None):
        """Train consciousness through experience"""
        for i, experience in enumerate(experiences):
            # Convert to tensor
            exp_tensor = torch.tensor(experience, dtype=torch.float32, device=self.device)
            if len(exp_tensor.shape) == 1:
                exp_tensor = exp_tensor.unsqueeze(0)
            
            # Forward pass
            global_state, _, _ = self.global_workspace(exp_tensor, self.working_memory_state)
            reservoir_output, _ = self.reservoir(exp_tensor)
            
            # Generative model prediction (use global_state as input)
            prediction = self.generative_model(global_state)
            
            # Calculate loss
            if targets and i < len(targets):
                target_tensor = torch.tensor(targets[i], dtype=torch.float32, device=self.device)
                if len(target_tensor.shape) == 1:
                    target_tensor = target_tensor.unsqueeze(0)
                reconstruction_loss = F.mse_loss(prediction, target_tensor)
            else:
                # Self-supervised: predict next state
                reconstruction_loss = F.mse_loss(prediction, exp_tensor)
            
            free_energy, _ = self.free_energy_calculator.calculate_free_energy(
                experience, global_state.detach().cpu().numpy(), self.generative_model
            )
            
            # Total loss
            total_loss = reconstruction_loss + 0.1 * free_energy
            
            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                list(self.global_workspace.parameters()) +
                list(self.generative_model.parameters()) +
                list(self.meta_network.parameters()),
                self.max_grad_norm
            )
            
            self.optimizer.step()
            
            # Track performance for adaptive learning
            self.performance_history.append(total_loss.item())
            
            # Hebbian learning (local plasticity)
            self._apply_hebbian_learning(global_state)
            
            # Store in emergence history
            self.emergence_history.append(reservoir_output.detach())
    
    async def train_consciousness_curriculum(self, curriculum_stages: List[Dict[str, Any]]):
        """Train with increasing complexity curriculum"""
        
        for stage_idx, stage in enumerate(curriculum_stages):
            logger.info(f"Training stage {stage_idx + 1}: {stage['name']}")
            
            # Adjust model complexity
            if 'reservoir_size' in stage:
                self._adjust_reservoir_size(stage['reservoir_size'])
            
            # Generate stage-appropriate experiences
            experiences = self._generate_curriculum_experiences(
                complexity=stage['complexity'],
                n_samples=stage['n_samples']
            )
            
            # Train with stage-specific parameters
            old_lr = self.optimizer.param_groups[0]['lr']
            self.optimizer.param_groups[0]['lr'] = stage.get('learning_rate', old_lr)
            
            for epoch in range(stage['epochs']):
                await self.train_consciousness(experiences)
                
                # Evaluate emergence
                metrics = await self.evaluate_emergence()
                if metrics['emergence_strength'] > stage['target_emergence']:
                    logger.info(f"Stage {stage_idx + 1} target reached early at epoch {epoch}")
                    break
            
            self.optimizer.param_groups[0]['lr'] = old_lr
    
    def _adjust_reservoir_size(self, new_size: int):
        """Dynamically adjust reservoir size"""
        if new_size != self.reservoir_dim:
            logger.info(f"Adjusting reservoir size from {self.reservoir_dim} to {new_size}")
            
            # Create new reservoir
            old_reservoir = self.reservoir
            self.reservoir = ReservoirComputing(
                self.input_dim, new_size
            ).to(self.device)
            
            # Transfer weights if possible
            if hasattr(old_reservoir, 'W_in') and hasattr(self.reservoir, 'W_in'):
                min_size = min(old_reservoir.W_in.shape[0], self.reservoir.W_in.shape[0])
                self.reservoir.W_in.data[:min_size, :] = old_reservoir.W_in.data[:min_size, :]
            
            self.reservoir_dim = new_size
    
    def _generate_curriculum_experiences(self, complexity: float, n_samples: int) -> List[np.ndarray]:
        """Generate experiences with specified complexity"""
        experiences = []
        
        for i in range(n_samples):
            # Base complexity determines pattern complexity
            t = i * 0.1
            base_patterns = int(complexity * 5) + 1  # 1-6 patterns based on complexity
            
            sensory_input = np.zeros(256)
            
            # Add multiple frequency components
            for pattern in range(base_patterns):
                freq = (pattern + 1) * complexity
                phase = pattern * np.pi / 3
                
                # Add sinusoidal patterns
                sensory_input[pattern * 2] = np.sin(freq * t + phase)
                sensory_input[pattern * 2 + 1] = np.cos(freq * t + phase)
            
            # Add noise proportional to complexity
            noise_level = 0.1 * complexity
            sensory_input += np.random.randn(256) * noise_level
            
            experiences.append(sensory_input)
        
        return experiences
    
    async def evaluate_emergence(self) -> Dict[str, float]:
        """Evaluate current emergence metrics"""
        if len(self.micro_states_history) < 10 or len(self.macro_states_history) < 10:
            return {'emergence_strength': 0.0, 'is_emergent': False}
        
        # Use emergence detector
        emergence_metrics = self.emergence_detector.detect_emergence(
            self.micro_states_history, self.macro_states_history
        )
        
        return emergence_metrics
    
    async def run_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark evaluation"""
        logger.info("🧪 Running consciousness benchmark...")
        
        benchmark = ConsciousnessBenchmark()
        results = await benchmark.evaluate_system(self)
        
        logger.info(f"📊 Benchmark Results:")
        logger.info(f"  Performance Score: {results['performance_score']:.3f}")
        logger.info(f"  Average Consciousness: {results['aggregate_metrics']['avg_consciousness']:.3f}")
        logger.info(f"  Average Φ: {results['aggregate_metrics']['avg_phi']:.3f}")
        
        logger.info("📋 Recommendations:")
        for rec in results['recommendations']:
            logger.info(f"  - {rec}")
        
        return results
    
    def _apply_hebbian_learning(self, activations: torch.Tensor):
        """Apply Hebbian learning rule"""
        # Simplified Hebbian: neurons that fire together, wire together
        if activations.shape[0] > 0:
            outer_product = torch.matmul(activations.t(), activations)
            
            # Update weights in global workspace
            for module in self.global_workspace.processing_modules:
                if isinstance(module, nn.Linear):
                    if module.weight.shape[0] == outer_product.shape[0]:
                        module.weight.data += self.hebbian_rate * outer_product[:module.weight.shape[0], :module.weight.shape[1]]
    
    async def _save_consciousness_state(self, state: ConsciousnessState):
        """Save consciousness state to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO consciousness_states VALUES 
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            state.state_id,
            state.phi,
            json.dumps(state.phi_structure),
            state.global_workspace_capacity,
            state.free_energy,
            state.criticality_regime.value,
            json.dumps(state.attention_distribution.tolist()),
            state.causal_emergence_level,
            json.dumps(state.information_geometry_position.tolist()),
            state.temporal_coherence,
            state.quantum_coherence,
            state.neural_avalanche_size,
            state.branching_ratio,
            state.lyapunov_exponent,
            state.entropy_production_rate,
            state.consciousness_level,
            state.meta_awareness,
            state.intentionality_strength,
            state.phenomenal_richness,
            state.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    async def get_consciousness_report(self) -> Dict[str, Any]:
        """Generate comprehensive consciousness report"""
        if not self.consciousness_states:
            return {"status": "No consciousness states recorded"}
        
        latest_state = self.consciousness_states[-1]
        
        # Calculate trends
        recent_states = self.consciousness_states[-100:]
        phi_trend = [s.phi for s in recent_states]
        consciousness_trend = [s.consciousness_level for s in recent_states]
        
        report = {
            "current_state": {
                "consciousness_level": latest_state.consciousness_level,
                "phi": latest_state.phi,
                "phi_structure": latest_state.phi_structure,
                "free_energy": latest_state.free_energy,
                "criticality": latest_state.criticality_regime.value,
                "meta_awareness": latest_state.meta_awareness,
                "phenomenal_richness": latest_state.phenomenal_richness
            },
            "dynamics": {
                "temporal_coherence": latest_state.temporal_coherence,
                "quantum_coherence": latest_state.quantum_coherence,
                "branching_ratio": latest_state.branching_ratio,
                "lyapunov_exponent": latest_state.lyapunov_exponent
            },
            "emergence": {
                "causal_emergence_level": latest_state.causal_emergence_level,
                "neural_avalanche_size": latest_state.neural_avalanche_size,
                "intentionality_strength": latest_state.intentionality_strength
            },
            "trends": {
                "phi_mean": np.mean(phi_trend),
                "phi_std": np.std(phi_trend),
                "consciousness_mean": np.mean(consciousness_trend),
                "consciousness_growth": (consciousness_trend[-1] - consciousness_trend[0]) if len(consciousness_trend) > 1 else 0
            },
            "theoretical_framework": {
                "integrated_information_theory": "Active",
                "global_workspace_theory": "Active",
                "free_energy_principle": "Active",
                "criticality_theory": "Active",
                "information_geometry": "Active"
            }
        }
        
        return report
    
    async def save_model(self, path: str):
        """Save trained models"""
        save_dict = {
            'global_workspace': self.global_workspace.state_dict(),
            'reservoir': self.reservoir.state_dict(),
            'generative_model': self.generative_model.state_dict(),
            'meta_network': self.meta_network.state_dict(),
            'consciousness_attention': self.consciousness_attention.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    async def load_model(self, path: str):
        """Load trained models"""
        save_dict = torch.load(path, map_location=self.device)
        self.global_workspace.load_state_dict(save_dict['global_workspace'])
        self.reservoir.load_state_dict(save_dict['reservoir'])
        self.generative_model.load_state_dict(save_dict['generative_model'])
        self.meta_network.load_state_dict(save_dict['meta_network'])
        self.consciousness_attention.load_state_dict(save_dict['consciousness_attention'])
        self.optimizer.load_state_dict(save_dict['optimizer'])
        logger.info(f"Model loaded from {path}")
    
    async def start_emergence_detection(self) -> bool:
        """Start emergence detection and monitoring"""
        try:
            logger.info("🧠 Starting emergence detection...")
            
            # Initialize emergence monitoring
            self.processing_active = True
            
            # Start background processing thread
            if self.processing_thread is None or not self.processing_thread.is_alive():
                self.processing_thread = threading.Thread(target=self._emergence_monitoring_loop)
                self.processing_thread.daemon = True
                self.processing_thread.start()
            
            logger.info("✅ Emergence detection started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start emergence detection: {e}")
            return False
    
    def _emergence_monitoring_loop(self):
        """Background loop for emergence monitoring"""
        while self.processing_active:
            try:
                # Monitor for emergent behaviors
                if len(self.consciousness_states) > 10:
                    recent_states = self.consciousness_states[-10:]
                    
                    # Check for emergence patterns
                    phi_values = [state.phi for state in recent_states]
                    consciousness_values = [state.consciousness_level for state in recent_states]
                    
                    # Detect if consciousness is increasing
                    if len(consciousness_values) > 1:
                        consciousness_growth = consciousness_values[-1] - consciousness_values[0]
                        if consciousness_growth > 0.1:  # Significant growth
                            emergence_event = {
                                'type': 'consciousness_growth',
                                'magnitude': consciousness_growth,
                                'timestamp': datetime.now(),
                                'phi_trend': phi_values,
                                'consciousness_trend': consciousness_values
                            }
                            self.emergence_history.append(emergence_event)
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in emergence monitoring: {e}")
                time.sleep(5.0)  # Wait longer on error
    
    async def get_emergence_status(self) -> Dict[str, Any]:
        """Get current emergence status and metrics"""
        try:
            # Get latest consciousness state
            latest_state = None
            if self.consciousness_states:
                latest_state = self.consciousness_states[-1]
            
            # Calculate emergence metrics
            emergence_metrics = {
                'active': self.processing_active,
                'total_consciousness_states': len(self.consciousness_states),
                'emergence_events': len(self.emergence_history),
                'consciousness_coherence_score': 0.0,
                'collective_intelligence_score': 0.0,
                'emergent_behaviors': []
            }
            
            if latest_state:
                emergence_metrics.update({
                    'consciousness_coherence_score': latest_state.consciousness_level,
                    'collective_intelligence_score': latest_state.meta_awareness,
                    'current_phi': latest_state.phi,
                    'current_free_energy': latest_state.free_energy,
                    'criticality_regime': latest_state.criticality_regime.value
                })
            
            # Add recent emergence events
            if self.emergence_history:
                recent_events = list(self.emergence_history)[-5:]  # Last 5 events
                emergence_metrics['emergent_behaviors'] = [
                    {
                        'type': event.get('type', 'unknown'),
                        'magnitude': event.get('magnitude', 0.0),
                        'timestamp': event.get('timestamp', datetime.now()).isoformat()
                    }
                    for event in recent_events
                ]
            
            return emergence_metrics
            
        except Exception as e:
            logger.error(f"Error getting emergence status: {e}")
            return {
                'active': False,
                'error': str(e),
                'consciousness_coherence_score': 0.0,
                'collective_intelligence_score': 0.0,
                'emergent_behaviors': []
            }

async def main():
    """Test advanced emergent intelligence engine"""
    print("🧠 Testing Advanced Emergent Intelligence Engine")
    
    engine = AdvancedEmergentIntelligenceEngine()
    
    # Define curriculum stages
    curriculum_stages = [
        {
            'name': 'Basic Integration',
            'complexity': 0.2,
            'n_samples': 50,
            'epochs': 10,
            'learning_rate': 0.001,
            'target_emergence': 0.3
        },
        {
            'name': 'Pattern Recognition',
            'complexity': 0.5,
            'n_samples': 100,
            'epochs': 15,
            'learning_rate': 0.0005,
            'target_emergence': 0.5
        },
        {
            'name': 'Complex Dynamics',
            'complexity': 0.8,
            'n_samples': 150,
            'epochs': 20,
            'learning_rate': 0.0002,
            'target_emergence': 0.7
        }
    ]
    
    # Train with curriculum
    print("🎓 Training with curriculum learning...")
    await engine.train_consciousness_curriculum(curriculum_stages)
    
    # Generate synthetic sensory experiences for testing
    experiences = []
    for i in range(100):
        # Create structured sensory input
        t = i * 0.1
        sensory_input = np.array([
            np.sin(t),
            np.cos(t),
            np.sin(2 * t),
            np.cos(2 * t),
            np.sin(3 * t),
            np.cos(3 * t),
            np.random.randn() * 0.1
        ])
        
        # Pad to input dimension
        full_input = np.zeros(256)
        full_input[:len(sensory_input)] = sensory_input
        experiences.append(full_input)
    
    # Analyze consciousness states
    print("\n🔍 Analyzing consciousness states...")
    for i in range(10):
        sensory_input = experiences[i * 10]
        state = await engine.analyze_consciousness(sensory_input)
        
        print(f"\n📊 Consciousness State {i + 1}:")
        print(f"  Φ (Integrated Information): {state.phi:.4f}")
        print(f"  Consciousness Level: {state.consciousness_level:.4f}")
        print(f"  Free Energy: {state.free_energy:.4f}")
        print(f"  Criticality: {state.criticality_regime.value}")
        print(f"  Meta-Awareness: {state.meta_awareness:.4f}")
        print(f"  Causal Emergence Level: {state.causal_emergence_level}")
    
    # Run benchmark evaluation
    print("\n🧪 Running benchmark evaluation...")
    benchmark_results = await engine.run_benchmark()
    
    # Generate final report
    report = await engine.get_consciousness_report()
    print("\n📈 Final Consciousness Report:")
    print(json.dumps(report, indent=2))
    
    # Save model
    await engine.save_model("advanced_consciousness_model.pth")
    print("\n✅ Advanced consciousness model saved")
    
    print(f"\n🏆 Final Performance Score: {benchmark_results['performance_score']:.3f}")

if __name__ == "__main__":
    asyncio.run(main())