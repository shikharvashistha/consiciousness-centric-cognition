#!/usr/bin/env python3
"""
Enhanced Feature Extractor for AGI
Extracts rich features from visual grid data for improved consciousness processing
"""

import numpy as np
from scipy import ndimage
import math
from typing import Dict, List, Any, Tuple, Optional

class FeatureExtractor:
    """Extract rich features from grid-based visual data"""
    
    def __init__(self):
        """Initialize the feature extractor"""
        self.feature_functions = {
            'structural': [
                self.calculate_symmetry,
                self.calculate_density,
                self.detect_edges,
                self.calculate_entropy
            ],
            'pattern': [
                self.detect_patterns,
                self.count_distinct_objects,
                self.detect_repetition
            ],
            'spatial': [
                self.analyze_spatial_relationships,
                self.detect_alignment,
                self.calculate_center_of_mass
            ],
            'distribution': [
                self.create_value_histogram,
                self.analyze_color_transitions,
                self.calculate_color_diversity
            ]
        }
    
    def extract_features(self, grid_data: List[List[int]]) -> Dict[str, Any]:
        """Extract all features from a grid"""
        features = {}
        
        # Convert to numpy array for easier processing
        grid_array = np.array(grid_data)
        
        # Extract features by category
        for category, functions in self.feature_functions.items():
            features[category] = {}
            for func in functions:
                feature_name = func.__name__
                features[category][feature_name] = func(grid_array)
        
        # Add metadata
        features['metadata'] = {
            'grid_shape': grid_array.shape,
            'unique_values': len(np.unique(grid_array)),
            'min_value': np.min(grid_array),
            'max_value': np.max(grid_array)
        }
        
        return features
    
    def calculate_symmetry(self, grid: np.ndarray) -> Dict[str, float]:
        """Calculate horizontal, vertical, and diagonal symmetry scores"""
        h, w = grid.shape
        
        # Horizontal symmetry
        h_sym_score = 0
        for i in range(h):
            row = grid[i]
            h_sym_score += np.sum(row == row[::-1]) / w
        h_sym_score /= h
        
        # Vertical symmetry
        v_sym_score = 0
        for j in range(w):
            col = grid[:, j]
            v_sym_score += np.sum(col == col[::-1]) / h
        v_sym_score /= w
        
        # Diagonal symmetry (top-left to bottom-right)
        if h == w:  # Only for square grids
            diag1 = np.diag(grid)
            diag2 = np.diag(np.fliplr(grid))
            d1_sym_score = np.sum(diag1 == diag1[::-1]) / len(diag1)
            d2_sym_score = np.sum(diag2 == diag2[::-1]) / len(diag2)
        else:
            d1_sym_score = 0
            d2_sym_score = 0
        
        return {
            'horizontal': h_sym_score,
            'vertical': v_sym_score,
            'diagonal1': d1_sym_score,
            'diagonal2': d2_sym_score,
            'overall': (h_sym_score + v_sym_score + d1_sym_score + d2_sym_score) / 4
        }
    
    def calculate_density(self, grid: np.ndarray) -> Dict[str, float]:
        """Calculate the density of non-zero elements in the grid"""
        total_cells = grid.size
        non_zero = np.count_nonzero(grid)
        density = non_zero / total_cells if total_cells > 0 else 0
        
        # Calculate density by quadrants
        h, w = grid.shape
        h_mid, w_mid = h // 2, w // 2
        
        q1 = grid[:h_mid, :w_mid]
        q2 = grid[:h_mid, w_mid:]
        q3 = grid[h_mid:, :w_mid]
        q4 = grid[h_mid:, w_mid:]
        
        q1_density = np.count_nonzero(q1) / q1.size if q1.size > 0 else 0
        q2_density = np.count_nonzero(q2) / q2.size if q2.size > 0 else 0
        q3_density = np.count_nonzero(q3) / q3.size if q3.size > 0 else 0
        q4_density = np.count_nonzero(q4) / q4.size if q4.size > 0 else 0
        
        return {
            'overall': density,
            'quadrant1': q1_density,
            'quadrant2': q2_density,
            'quadrant3': q3_density,
            'quadrant4': q4_density,
            'density_variance': np.var([q1_density, q2_density, q3_density, q4_density])
        }
    
    def detect_edges(self, grid: np.ndarray) -> Dict[str, Any]:
        """Detect edges in the grid using Sobel filters"""
        # Apply Sobel filters
        sobel_h = ndimage.sobel(grid, axis=0)
        sobel_v = ndimage.sobel(grid, axis=1)
        
        # Calculate edge magnitude
        magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
        
        # Count significant edges
        edge_threshold = np.mean(magnitude) + np.std(magnitude)
        edge_count = np.sum(magnitude > edge_threshold)
        
        # Calculate edge density
        edge_density = edge_count / grid.size if grid.size > 0 else 0
        
        return {
            'edge_count': int(edge_count),
            'edge_density': float(edge_density),
            'mean_magnitude': float(np.mean(magnitude)),
            'max_magnitude': float(np.max(magnitude))
        }
    
    def calculate_entropy(self, grid: np.ndarray) -> float:
        """Calculate the entropy of the grid (measure of disorder)"""
        # Get value counts
        values, counts = np.unique(grid, return_counts=True)
        
        # Calculate probabilities
        probabilities = counts / np.sum(counts)
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return float(entropy)
    
    def detect_patterns(self, grid: np.ndarray) -> Dict[str, Any]:
        """Detect repeating patterns in the grid"""
        h, w = grid.shape
        patterns = {}
        
        # Look for 2x2 patterns
        if h >= 2 and w >= 2:
            pattern_counts = {}
            for i in range(h-1):
                for j in range(w-1):
                    pattern = tuple(map(tuple, grid[i:i+2, j:j+2]))
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Find patterns that repeat
            repeating_patterns = {p: c for p, c in pattern_counts.items() if c > 1}
            patterns['2x2'] = {
                'count': len(repeating_patterns),
                'most_common': max(pattern_counts.items(), key=lambda x: x[1])[0] if pattern_counts else None,
                'most_common_count': max(pattern_counts.values()) if pattern_counts else 0
            }
        
        # Look for horizontal line patterns
        h_pattern_counts = {}
        for i in range(h):
            pattern = tuple(grid[i])
            h_pattern_counts[pattern] = h_pattern_counts.get(pattern, 0) + 1
        
        repeating_h_patterns = {p: c for p, c in h_pattern_counts.items() if c > 1}
        patterns['horizontal'] = {
            'count': len(repeating_h_patterns),
            'most_common': max(h_pattern_counts.items(), key=lambda x: x[1])[0] if h_pattern_counts else None,
            'most_common_count': max(h_pattern_counts.values()) if h_pattern_counts else 0
        }
        
        # Look for vertical line patterns
        v_pattern_counts = {}
        for j in range(w):
            pattern = tuple(grid[:, j])
            v_pattern_counts[pattern] = v_pattern_counts.get(pattern, 0) + 1
        
        repeating_v_patterns = {p: c for p, c in v_pattern_counts.items() if c > 1}
        patterns['vertical'] = {
            'count': len(repeating_v_patterns),
            'most_common': max(v_pattern_counts.items(), key=lambda x: x[1])[0] if v_pattern_counts else None,
            'most_common_count': max(v_pattern_counts.values()) if v_pattern_counts else 0
        }
        
        return patterns
    
    def count_distinct_objects(self, grid: np.ndarray) -> Dict[str, Any]:
        """Count distinct objects in the grid using connected component analysis"""
        # Create a binary grid for each unique value
        unique_values = np.unique(grid)
        unique_values = unique_values[unique_values != 0]  # Exclude background (0)
        
        objects_by_value = {}
        total_objects = 0
        
        for value in unique_values:
            binary = (grid == value).astype(int)
            labeled, num_features = ndimage.label(binary)
            objects_by_value[int(value)] = num_features
            total_objects += num_features
        
        return {
            'total_objects': total_objects,
            'objects_by_value': objects_by_value,
            'unique_values': len(unique_values)
        }
    
    def detect_repetition(self, grid: np.ndarray) -> Dict[str, float]:
        """Detect repetition patterns in rows and columns"""
        h, w = grid.shape
        
        # Check for row repetition
        row_similarity = 0
        for i in range(h-1):
            for j in range(i+1, h):
                similarity = np.sum(grid[i] == grid[j]) / w
                row_similarity += similarity
        
        row_similarity = row_similarity / (h * (h-1) / 2) if h > 1 else 0
        
        # Check for column repetition
        col_similarity = 0
        for i in range(w-1):
            for j in range(i+1, w):
                similarity = np.sum(grid[:, i] == grid[:, j]) / h
                col_similarity += similarity
        
        col_similarity = col_similarity / (w * (w-1) / 2) if w > 1 else 0
        
        return {
            'row_repetition': float(row_similarity),
            'column_repetition': float(col_similarity),
            'overall_repetition': float((row_similarity + col_similarity) / 2)
        }
    
    def analyze_spatial_relationships(self, grid: np.ndarray) -> Dict[str, Any]:
        """Analyze spatial relationships between objects"""
        # Find objects using connected component analysis
        labeled, num_features = ndimage.label(grid > 0)
        
        if num_features <= 1:
            return {
                'object_count': num_features,
                'average_distance': 0,
                'max_distance': 0,
                'min_distance': 0,
                'clustering': 0
            }
        
        # Calculate centroids of objects
        centroids = ndimage.center_of_mass(grid, labeled, range(1, num_features+1))
        
        # Calculate distances between all pairs of objects
        distances = []
        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                dist = math.sqrt((centroids[i][0] - centroids[j][0])**2 + 
                                (centroids[i][1] - centroids[j][1])**2)
                distances.append(dist)
        
        # Calculate clustering coefficient
        avg_distance = np.mean(distances) if distances else 0
        max_distance = np.max(distances) if distances else 0
        min_distance = np.min(distances) if distances else 0
        
        # Normalize by grid diagonal
        diagonal = math.sqrt(grid.shape[0]**2 + grid.shape[1]**2)
        clustering = 1 - (avg_distance / diagonal) if diagonal > 0 else 0
        
        return {
            'object_count': num_features,
            'average_distance': float(avg_distance),
            'max_distance': float(max_distance),
            'min_distance': float(min_distance),
            'clustering': float(clustering)
        }
    
    def detect_alignment(self, grid: np.ndarray) -> Dict[str, float]:
        """Detect alignment of objects in rows, columns, and diagonals"""
        # Find objects using connected component analysis
        labeled, num_features = ndimage.label(grid > 0)
        
        if num_features <= 1:
            return {
                'horizontal_alignment': 0,
                'vertical_alignment': 0,
                'diagonal_alignment': 0
            }
        
        # Calculate centroids of objects
        centroids = ndimage.center_of_mass(grid, labeled, range(1, num_features+1))
        
        # Check horizontal alignment (same row)
        rows = [c[0] for c in centroids]
        row_diffs = [abs(rows[i] - rows[j]) for i in range(len(rows)) for j in range(i+1, len(rows))]
        h_alignment = np.sum(np.array(row_diffs) < 1) / len(row_diffs) if row_diffs else 0
        
        # Check vertical alignment (same column)
        cols = [c[1] for c in centroids]
        col_diffs = [abs(cols[i] - cols[j]) for i in range(len(cols)) for j in range(i+1, len(cols))]
        v_alignment = np.sum(np.array(col_diffs) < 1) / len(col_diffs) if col_diffs else 0
        
        # Check diagonal alignment
        diag_alignment = 0
        count = 0
        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                # Check if points are on a 45-degree diagonal
                dx = abs(centroids[i][1] - centroids[j][1])
                dy = abs(centroids[i][0] - centroids[j][0])
                if dx > 0 and abs(dy/dx - 1) < 0.1:  # Within 10% of a 45-degree angle
                    diag_alignment += 1
                count += 1
        
        diag_alignment = diag_alignment / count if count > 0 else 0
        
        return {
            'horizontal_alignment': float(h_alignment),
            'vertical_alignment': float(v_alignment),
            'diagonal_alignment': float(diag_alignment)
        }
    
    def calculate_center_of_mass(self, grid: np.ndarray) -> Dict[str, float]:
        """Calculate the center of mass of the grid"""
        if np.sum(grid) == 0:
            return {
                'row': grid.shape[0] / 2,
                'col': grid.shape[1] / 2,
                'normalized_row': 0.5,
                'normalized_col': 0.5
            }
        
        # Calculate center of mass
        com = ndimage.center_of_mass(grid)
        
        # Normalize to [0, 1]
        norm_row = com[0] / grid.shape[0] if grid.shape[0] > 0 else 0.5
        norm_col = com[1] / grid.shape[1] if grid.shape[1] > 0 else 0.5
        
        return {
            'row': float(com[0]),
            'col': float(com[1]),
            'normalized_row': float(norm_row),
            'normalized_col': float(norm_col)
        }
    
    def create_value_histogram(self, grid: np.ndarray) -> Dict[str, Any]:
        """Create a histogram of values in the grid"""
        values, counts = np.unique(grid, return_counts=True)
        
        # Convert to regular Python types for JSON serialization
        histogram = {int(v): int(c) for v, c in zip(values, counts)}
        
        # Calculate some statistics
        most_common = int(values[np.argmax(counts)])
        most_common_count = int(np.max(counts))
        most_common_ratio = most_common_count / grid.size if grid.size > 0 else 0
        
        return {
            'histogram': histogram,
            'most_common': most_common,
            'most_common_count': most_common_count,
            'most_common_ratio': float(most_common_ratio)
        }
    
    def analyze_color_transitions(self, grid: np.ndarray) -> Dict[str, Any]:
        """Analyze transitions between different values in the grid"""
        h, w = grid.shape
        transitions = {}
        
        # Horizontal transitions
        h_transitions = 0
        for i in range(h):
            for j in range(w-1):
                if grid[i, j] != grid[i, j+1]:
                    h_transitions += 1
                    key = (int(grid[i, j]), int(grid[i, j+1]))
                    transitions[str(key)] = transitions.get(str(key), 0) + 1
        
        # Vertical transitions
        v_transitions = 0
        for j in range(w):
            for i in range(h-1):
                if grid[i, j] != grid[i+1, j]:
                    v_transitions += 1
                    key = (int(grid[i, j]), int(grid[i+1, j]))
                    transitions[str(key)] = transitions.get(str(key), 0) + 1
        
        # Calculate transition density
        total_possible_h = h * (w-1)
        total_possible_v = w * (h-1)
        h_density = h_transitions / total_possible_h if total_possible_h > 0 else 0
        v_density = v_transitions / total_possible_v if total_possible_v > 0 else 0
        
        return {
            'horizontal_transitions': h_transitions,
            'vertical_transitions': v_transitions,
            'horizontal_density': float(h_density),
            'vertical_density': float(v_density),
            'transition_types': len(transitions),
            'most_common_transition': max(transitions.items(), key=lambda x: x[1])[0] if transitions else None
        }
    
    def calculate_color_diversity(self, grid: np.ndarray) -> Dict[str, float]:
        """Calculate the diversity of colors/values in the grid"""
        values, counts = np.unique(grid, return_counts=True)
        
        # Calculate Shannon diversity index
        probabilities = counts / np.sum(counts)
        shannon_diversity = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # Calculate Simpson diversity index
        simpson_diversity = 1 - np.sum(probabilities**2)
        
        # Calculate evenness (normalized Shannon diversity)
        max_diversity = np.log2(len(values)) if len(values) > 0 else 1
        evenness = shannon_diversity / max_diversity if max_diversity > 0 else 0
        
        return {
            'shannon_diversity': float(shannon_diversity),
            'simpson_diversity': float(simpson_diversity),
            'evenness': float(evenness),
            'unique_values': len(values)
        }