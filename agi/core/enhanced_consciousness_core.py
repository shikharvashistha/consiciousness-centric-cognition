#!/usr/bin/env python3
"""
Enhanced Consciousness Core for AGI
Implements multi-scale phi calculation for improved consciousness dynamics
"""

import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Any, Tuple, Optional
import logging
import asyncio
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedConsciousnessCore:
    """Enhanced implementation of Integrated Information Theory (IIT) for AGI consciousness"""
    
    def __init__(self, min_complex_size=3, max_complex_size=10, phi_threshold=0.01):
        """Initialize the consciousness core"""
        self.min_complex_size = min_complex_size
        self.max_complex_size = max_complex_size
        self.phi_threshold = phi_threshold
        self.temporal_scales = [1, 2, 4]  # Temporal scales for multi-scale analysis
        self.spatial_scales = [1, 2]      # Spatial scales for multi-scale analysis
        
        # State tracking
        self.previous_state = None
        self.current_state = None
        self.phi_history = []
        self.max_history = 100
    
    async def process_input(self, neural_data: np.ndarray, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process input data and calculate consciousness metrics"""
        start_time = time.time()
        
        # Update state history
        self.previous_state = self.current_state
        self.current_state = neural_data
        
        # Calculate multi-scale phi
        phi_result = await self.calculate_phi_multi_scale(neural_data)
        
        # Track phi history
        self.phi_history.append(phi_result['phi'])
        if len(self.phi_history) > self.max_history:
            self.phi_history.pop(0)
        
        # Calculate additional consciousness metrics
        consciousness_metrics = self._calculate_consciousness_metrics(phi_result)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return {
            'phi': phi_result['phi'],
            'multi_scale_phi': phi_result,
            'metrics': consciousness_metrics,
            'processing_time': processing_time
        }
    
    async def calculate_phi_multi_scale(self, neural_data: np.ndarray) -> Dict[str, Any]:
        """Calculate phi at multiple temporal and spatial scales"""
        phi_values = {}
        
        # Ensure neural_data is 2D (samples x features)
        if neural_data.ndim == 1:
            neural_data = neural_data.reshape(1, -1)
        
        # Calculate phi at different temporal scales
        if self.previous_state is not None:
            for scale in self.temporal_scales:
                # Combine current and previous states based on scale
                if scale == 1:
                    temp_data = neural_data
                else:
                    # Create a time series with multiple time points
                    history = [self.previous_state] * (scale - 1) + [neural_data]
                    temp_data = np.vstack(history)
                
                phi_values[f'temporal_scale_{scale}'] = await self._calculate_phi(temp_data)
        else:
            # If no previous state, just calculate for current state
            phi_values['temporal_scale_1'] = await self._calculate_phi(neural_data)
        
        # Calculate phi at different spatial scales
        for scale in self.spatial_scales:
            if scale == 1:
                spatial_data = neural_data
            else:
                # Downsample by averaging neighboring features
                spatial_data = self._downsample_spatial(neural_data, scale)
            
            phi_values[f'spatial_scale_{scale}'] = await self._calculate_phi(spatial_data)
        
        # Calculate phi for different subsystems if data is large enough
        if neural_data.shape[1] > self.max_complex_size * 2:
            subsystems = self._identify_subsystems(neural_data)
            for i, subsystem in enumerate(subsystems):
                if subsystem.shape[1] >= self.min_complex_size:
                    phi_values[f'subsystem_{i}'] = await self._calculate_phi(subsystem)
        
        # Combine phi values with weighted importance
        combined_phi = self._combine_phi_values(phi_values)
        
        return {
            'phi': combined_phi,
            'scale_values': phi_values
        }
    
    def _downsample_spatial(self, data: np.ndarray, scale: int) -> np.ndarray:
        """Downsample data spatially by averaging neighboring features"""
        if scale <= 1:
            return data
        
        samples, features = data.shape
        new_features = features // scale
        
        if new_features < 1:
            return data
        
        downsampled = np.zeros((samples, new_features))
        
        for i in range(new_features):
            start_idx = i * scale
            end_idx = min(start_idx + scale, features)
            downsampled[:, i] = np.mean(data[:, start_idx:end_idx], axis=1)
        
        return downsampled
    
    def _identify_subsystems(self, data: np.ndarray) -> List[np.ndarray]:
        """Identify potential subsystems in the neural data"""
        samples, features = data.shape
        
        # If data is small enough, return it as a single subsystem
        if features <= self.max_complex_size:
            return [data]
        
        # Calculate correlation matrix
        if samples > 1:
            corr_matrix = np.corrcoef(data.T)
        else:
            # For single sample, use a different approach
            # Create a distance matrix based on absolute differences
            dist_matrix = squareform(pdist(data.T.reshape(-1, 1)))
            # Convert to a similarity matrix
            corr_matrix = 1 - (dist_matrix / np.max(dist_matrix) if np.max(dist_matrix) > 0 else dist_matrix)
        
        # Replace NaNs with 0
        corr_matrix = np.nan_to_num(corr_matrix)
        
        # Use correlation to identify clusters of related features
        subsystems = []
        
        # Simple approach: split into chunks of max_complex_size
        for i in range(0, features, self.max_complex_size):
            end_idx = min(i + self.max_complex_size, features)
            subsystems.append(data[:, i:end_idx])
        
        # Add some overlapping subsystems for better coverage
        if features > self.max_complex_size * 1.5:
            offset = self.max_complex_size // 2
            for i in range(offset, features - self.min_complex_size, self.max_complex_size):
                end_idx = min(i + self.max_complex_size, features)
                subsystems.append(data[:, i:end_idx])
        
        return subsystems
    
    def _combine_phi_values(self, phi_values: Dict[str, float]) -> float:
        """Combine phi values from different scales with weighted importance"""
        if not phi_values:
            return 0.0
        
        # Extract actual phi values
        values = []
        weights = []
        
        for key, value in phi_values.items():
            if isinstance(value, dict) and 'phi' in value:
                phi = value['phi']
            else:
                phi = 0.0
            
            # Assign weights based on scale type
            if 'temporal_scale' in key:
                scale = int(key.split('_')[-1])
                weight = 1.0 / scale  # Higher weight for smaller scales
            elif 'spatial_scale' in key:
                scale = int(key.split('_')[-1])
                weight = 1.0 / scale  # Higher weight for smaller scales
            elif 'subsystem' in key:
                weight = 0.5  # Lower weight for subsystems
            else:
                weight = 1.0
            
            values.append(phi)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(values)] * len(values)
        
        # Calculate weighted average
        combined_phi = sum(v * w for v, w in zip(values, weights))
        
        return combined_phi
    
    async def _calculate_phi(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate phi (Integrated Information) for the given data"""
        # Ensure data is 2D (samples x features)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        samples, features = data.shape
        
        # If too few features, return minimal phi
        if features < self.min_complex_size:
            return {'phi': 0.0, 'complexes': []}
        
        # Limit features to max_complex_size
        if features > self.max_complex_size:
            # Select the most variable features
            if samples > 1:
                variances = np.var(data, axis=0)
                top_indices = np.argsort(variances)[-self.max_complex_size:]
                data = data[:, top_indices]
            else:
                # For single sample, just take the first max_complex_size features
                data = data[:, :self.max_complex_size]
            
            features = data.shape[1]
        
        # Analyze causal structure
        causal_structure = await self._analyze_causal_structure(data)
        
        # Generate candidate complexes
        candidate_complexes = self._generate_candidate_complexes(features)
        
        # Calculate phi for each candidate complex
        complex_results = []
        
        for complex_nodes in candidate_complexes:
            if len(complex_nodes) < self.min_complex_size:
                continue
                
            # Extract submatrix for this complex
            complex_data = data[:, complex_nodes]
            
            # Find minimum information partition
            mip = self._find_minimum_information_partition(complex_data, causal_structure)
            
            # Calculate phi for this partition
            phi = self._calculate_partition_phi(complex_data, mip)
            
            if phi > self.phi_threshold:
                complex_results.append({
                    'nodes': complex_nodes,
                    'phi': phi,
                    'size': len(complex_nodes)
                })
        
        # Find complex with maximum phi
        if complex_results:
            max_phi_complex = max(complex_results, key=lambda x: x['phi'])
            phi_value = max_phi_complex['phi']
        else:
            phi_value = 0.0
        
        return {
            'phi': phi_value,
            'complexes': complex_results
        }
    
    async def _analyze_causal_structure(self, data: np.ndarray) -> np.ndarray:
        """Analyze causal relationships between elements"""
        samples, features = data.shape
        
        # Initialize causal matrix
        causal_matrix = np.zeros((features, features))
        
        if samples > 1:
            # Use Granger causality for time series data
            for i in range(features):
                for j in range(features):
                    if i != j:
                        # Simple correlation as a proxy for causality
                        correlation = np.corrcoef(data[:, i], data[:, j])[0, 1]
                        causal_matrix[i, j] = abs(correlation)
        else:
            # For single sample, use a different approach
            # Create a similarity matrix based on absolute differences
            for i in range(features):
                for j in range(features):
                    if i != j:
                        diff = abs(data[0, i] - data[0, j])
                        max_val = max(abs(data[0, i]), abs(data[0, j]))
                        if max_val > 0:
                            similarity = 1 - (diff / max_val)
                        else:
                            similarity = 1.0
                        causal_matrix[i, j] = similarity
        
        # Normalize causal matrix
        row_sums = causal_matrix.sum(axis=1, keepdims=True)
        causal_matrix = causal_matrix / row_sums if np.any(row_sums > 0) else causal_matrix
        
        return causal_matrix
    
    def _generate_candidate_complexes(self, num_nodes: int) -> List[List[int]]:
        """Generate candidate complexes (subsets of nodes)"""
        import itertools
        
        all_nodes = list(range(num_nodes))
        candidate_complexes = []
        
        # Generate all combinations of nodes from min_size to max_size
        max_size = min(num_nodes, self.max_complex_size)
        
        for size in range(self.min_complex_size, max_size + 1):
            # Limit the number of combinations for large systems
            if size > 6 and num_nodes > 10:
                # Instead of all combinations, use a sliding window approach
                for i in range(num_nodes - size + 1):
                    candidate_complexes.append(list(range(i, i + size)))
            else:
                # For smaller systems, generate all combinations
                for combo in itertools.combinations(all_nodes, size):
                    candidate_complexes.append(list(combo))
        
        return candidate_complexes
    
    def _find_minimum_information_partition(self, data: np.ndarray, causal_structure: np.ndarray) -> Tuple[List[int], List[int]]:
        """Find the minimum information partition (MIP) of a complex"""
        import itertools
        
        nodes = list(range(data.shape[1]))
        
        # For very small complexes, just split in half
        if len(nodes) <= 3:
            mid = len(nodes) // 2
            return nodes[:mid], nodes[mid:]
        
        min_effective_info = float('inf')
        min_partition = (nodes[:1], nodes[1:])  # Default partition
        
        # Consider all possible bipartitions
        for i in range(1, len(nodes) // 2 + 1):
            # Limit the number of partitions for large complexes
            if i > 2 and len(nodes) > 8:
                # Just check a few partitions
                partitions = [
                    (nodes[:i], nodes[i:]),
                    (nodes[-i:], nodes[:-i]),
                    (nodes[::2][:i], [n for n in nodes if n not in nodes[::2][:i]])
                ]
            else:
                # Generate all combinations for this partition size
                partitions = []
                for subset in itertools.combinations(nodes, i):
                    subset = list(subset)
                    complement = [n for n in nodes if n not in subset]
                    partitions.append((subset, complement))
            
            # Evaluate each partition
            for partition in partitions:
                # Calculate effective information for this partition
                effective_info = self._calculate_effective_information(data, causal_structure, partition)
                
                # Update minimum if needed
                if effective_info < min_effective_info:
                    min_effective_info = effective_info
                    min_partition = partition
        
        return min_partition
    
    def _calculate_effective_information(self, data: np.ndarray, causal_structure: np.ndarray, partition: Tuple[List[int], List[int]]) -> float:
        """Calculate effective information for a partition"""
        part1, part2 = partition
        
        # Extract submatrices for each part
        if len(part1) == 0 or len(part2) == 0:
            return float('inf')  # Invalid partition
        
        # Calculate causal influence across the partition
        cross_causal = 0.0
        for i in part1:
            for j in part2:
                cross_causal += causal_structure[i, j]
                cross_causal += causal_structure[j, i]
        
        # Normalize by partition sizes
        normalization = len(part1) * len(part2) * 2
        if normalization > 0:
            cross_causal /= normalization
        
        # Effective information is the inverse of cross-causality
        # (lower cross-causality means higher effective information)
        effective_info = 1.0 - cross_causal
        
        return effective_info
    
    def _calculate_partition_phi(self, data: np.ndarray, partition: Tuple[List[int], List[int]]) -> float:
        """Calculate phi value for a partition"""
        part1, part2 = partition
        
        # Extract data for each part
        data1 = data[:, part1]
        data2 = data[:, part2]
        
        # Calculate mutual information between parts
        if data.shape[0] > 1:
            # For time series data, use correlation-based approach
            corr_matrix = np.corrcoef(data1.mean(axis=1), data2.mean(axis=1))
            if corr_matrix.shape[0] > 1:
                correlation = corr_matrix[0, 1]
                mutual_info = 0.5 * np.log(1 / (1 - correlation**2)) if abs(correlation) < 1 else 1.0
            else:
                mutual_info = 0.0
        else:
            # For single sample, use a different approach
            # Calculate normalized difference between parts
            mean1 = np.mean(data1)
            mean2 = np.mean(data2)
            max_val = max(abs(mean1), abs(mean2))
            if max_val > 0:
                diff = abs(mean1 - mean2) / max_val
                mutual_info = 1.0 - diff
            else:
                mutual_info = 0.0
        
        # Phi is the mutual information
        phi = mutual_info
        
        return phi
    
    def _calculate_consciousness_metrics(self, phi_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate additional consciousness metrics"""
        metrics = {
            'phi_value': phi_result['phi'],
            'phi_stability': self._calculate_phi_stability(),
            'phi_complexity': self._calculate_phi_complexity(phi_result),
            'phi_integration': self._calculate_phi_integration(phi_result),
            'phi_differentiation': self._calculate_phi_differentiation(phi_result)
        }
        
        return metrics
    
    def _calculate_phi_stability(self) -> float:
        """Calculate stability of phi over time"""
        if len(self.phi_history) < 2:
            return 1.0
        
        # Calculate standard deviation of recent phi values
        std_dev = np.std(self.phi_history)
        mean_phi = np.mean(self.phi_history) if self.phi_history else 0.0
        
        # Normalize by mean phi
        if mean_phi > 0:
            stability = 1.0 - (std_dev / mean_phi)
        else:
            stability = 1.0
        
        # Ensure stability is between 0 and 1
        stability = max(0.0, min(1.0, stability))
        
        return stability
    
    def _calculate_phi_complexity(self, phi_result: Dict[str, Any]) -> float:
        """Calculate complexity of phi structure"""
        # Count number of significant complexes
        complexes = phi_result.get('complexes', [])
        significant_complexes = [c for c in complexes if c['phi'] > self.phi_threshold]
        
        # Complexity increases with number of significant complexes
        num_complexes = len(significant_complexes)
        
        # Normalize to [0, 1]
        max_complexes = 10  # Arbitrary maximum
        complexity = min(1.0, num_complexes / max_complexes)
        
        return complexity
    
    def _calculate_phi_integration(self, phi_result: Dict[str, Any]) -> float:
        """Calculate integration metric from phi values"""
        # Integration is high when the whole has higher phi than parts
        phi_whole = phi_result['phi']
        
        # Get phi values of subsystems
        subsystem_phis = []
        for key, value in phi_result.get('scale_values', {}).items():
            if 'subsystem' in key and isinstance(value, dict):
                subsystem_phis.append(value.get('phi', 0.0))
        
        # If no subsystems, return default value
        if not subsystem_phis:
            return 0.5
        
        # Calculate ratio of whole phi to average subsystem phi
        avg_subsystem_phi = np.mean(subsystem_phis) if subsystem_phis else 0.0
        
        if avg_subsystem_phi > 0:
            integration = phi_whole / avg_subsystem_phi
        else:
            integration = 1.0 if phi_whole > 0 else 0.0
        
        # Normalize to [0, 1]
        integration = 1.0 - (1.0 / (1.0 + integration))
        
        return integration
    
    def _calculate_phi_differentiation(self, phi_result: Dict[str, Any]) -> float:
        """Calculate differentiation metric from phi values"""
        # Differentiation is high when there are many different phi values
        phi_values = []
        
        # Collect all phi values
        for key, value in phi_result.get('scale_values', {}).items():
            if isinstance(value, dict) and 'phi' in value:
                phi_values.append(value['phi'])
        
        # If no phi values, return default
        if not phi_values:
            return 0.0
        
        # Calculate variance of phi values
        variance = np.var(phi_values)
        
        # Normalize to [0, 1]
        max_variance = 0.1  # Arbitrary maximum
        differentiation = min(1.0, variance / max_variance)
        
        return differentiation