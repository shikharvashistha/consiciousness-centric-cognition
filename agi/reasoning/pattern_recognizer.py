#!/usr/bin/env python3
"""
Pattern Recognition System for ARC Tasks
Identifies transformation patterns in ARC tasks and applies them to new inputs
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ARCPatternRecognizer:
    """Recognize patterns in ARC tasks and map them to transformations"""
    
    def __init__(self):
        """Initialize the pattern recognizer"""
        self.transformation_detectors = [
            self._detect_identity,
            self._detect_horizontal_flip,
            self._detect_vertical_flip,
            self._detect_rotation_90,
            self._detect_rotation_180,
            self._detect_rotation_270,
            self._detect_color_mapping,
            self._detect_object_movement,
            self._detect_pattern_completion,
            self._detect_cellular_automata
        ]
        
        self.transformation_names = [
            "identity",
            "horizontal_flip",
            "vertical_flip",
            "rotation_90",
            "rotation_180",
            "rotation_270",
            "color_mapping",
            "object_movement",
            "pattern_completion",
            "cellular_automata"
        ]
    
    def analyze_examples(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze examples to determine the transformation pattern"""
        transformation_scores = {}
        transformation_params = {}
        
        # Check if examples are valid
        if not examples or not all(('input' in ex and 'output' in ex) for ex in examples):
            logger.warning("Invalid examples provided")
            return {
                'transformation': 'identity',
                'confidence': 0.0,
                'all_scores': {'identity': 0.0},
                'params': {}
            }
        
        # Analyze each transformation
        for detector, name in zip(self.transformation_detectors, self.transformation_names):
            score, params = detector(examples)
            transformation_scores[name] = score
            transformation_params[name] = params
        
        # Find the most likely transformation
        best_transformation = max(transformation_scores.items(), key=lambda x: x[1])
        
        return {
            'transformation': best_transformation[0],
            'confidence': best_transformation[1],
            'all_scores': transformation_scores,
            'params': transformation_params[best_transformation[0]]
        }
    
    def apply_transformation(self, input_grid: List[List[int]], transformation: str, params: Dict = None) -> List[List[int]]:
        """Apply the identified transformation to the input grid"""
        # Convert to numpy array for easier processing
        grid = np.array(input_grid)
        
        # Apply the transformation
        if transformation == "identity":
            result = self._apply_identity(grid, params)
        elif transformation == "horizontal_flip":
            result = self._apply_horizontal_flip(grid, params)
        elif transformation == "vertical_flip":
            result = self._apply_vertical_flip(grid, params)
        elif transformation == "rotation_90":
            result = self._apply_rotation_90(grid, params)
        elif transformation == "rotation_180":
            result = self._apply_rotation_180(grid, params)
        elif transformation == "rotation_270":
            result = self._apply_rotation_270(grid, params)
        elif transformation == "color_mapping":
            result = self._apply_color_mapping(grid, params)
        elif transformation == "object_movement":
            result = self._apply_object_movement(grid, params)
        elif transformation == "pattern_completion":
            result = self._apply_pattern_completion(grid, params)
        elif transformation == "cellular_automata":
            result = self._apply_cellular_automata(grid, params)
        else:
            # Default to identity
            result = grid.copy()
        
        # Convert back to list
        return result.tolist()
    
    def _detect_identity(self, examples: List[Dict[str, Any]]) -> Tuple[float, Dict]:
        """Detect if the transformation is identity (no change)"""
        score = 0.0
        count = 0
        
        for example in examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Check if input and output have the same shape
            if input_grid.shape != output_grid.shape:
                continue
            
            # Calculate similarity
            total_cells = input_grid.size
            matching_cells = np.sum(input_grid == output_grid)
            similarity = matching_cells / total_cells if total_cells > 0 else 0
            
            score += similarity
            count += 1
        
        # Calculate average score
        avg_score = score / count if count > 0 else 0.0
        
        return avg_score, {}
    
    def _apply_identity(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        """Apply identity transformation (no change)"""
        return grid.copy()
    
    def _detect_horizontal_flip(self, examples: List[Dict[str, Any]]) -> Tuple[float, Dict]:
        """Detect if the transformation is a horizontal flip"""
        score = 0.0
        count = 0
        
        for example in examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Check if input and output have the same shape
            if input_grid.shape != output_grid.shape:
                continue
            
            # Apply horizontal flip to input
            flipped = np.fliplr(input_grid)
            
            # Calculate similarity
            total_cells = output_grid.size
            matching_cells = np.sum(flipped == output_grid)
            similarity = matching_cells / total_cells if total_cells > 0 else 0
            
            score += similarity
            count += 1
        
        # Calculate average score
        avg_score = score / count if count > 0 else 0.0
        
        return avg_score, {}
    
    def _apply_horizontal_flip(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        """Apply horizontal flip transformation"""
        return np.fliplr(grid)
    
    def _detect_vertical_flip(self, examples: List[Dict[str, Any]]) -> Tuple[float, Dict]:
        """Detect if the transformation is a vertical flip"""
        score = 0.0
        count = 0
        
        for example in examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Check if input and output have the same shape
            if input_grid.shape != output_grid.shape:
                continue
            
            # Apply vertical flip to input
            flipped = np.flipud(input_grid)
            
            # Calculate similarity
            total_cells = output_grid.size
            matching_cells = np.sum(flipped == output_grid)
            similarity = matching_cells / total_cells if total_cells > 0 else 0
            
            score += similarity
            count += 1
        
        # Calculate average score
        avg_score = score / count if count > 0 else 0.0
        
        return avg_score, {}
    
    def _apply_vertical_flip(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        """Apply vertical flip transformation"""
        return np.flipud(grid)
    
    def _detect_rotation_90(self, examples: List[Dict[str, Any]]) -> Tuple[float, Dict]:
        """Detect if the transformation is a 90-degree rotation"""
        score = 0.0
        count = 0
        
        for example in examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Check if dimensions are compatible with 90-degree rotation
            if input_grid.shape[0] != output_grid.shape[1] or input_grid.shape[1] != output_grid.shape[0]:
                continue
            
            # Apply 90-degree rotation to input
            rotated = np.rot90(input_grid, k=1)
            
            # Calculate similarity
            total_cells = output_grid.size
            matching_cells = np.sum(rotated == output_grid)
            similarity = matching_cells / total_cells if total_cells > 0 else 0
            
            score += similarity
            count += 1
        
        # Calculate average score
        avg_score = score / count if count > 0 else 0.0
        
        return avg_score, {}
    
    def _apply_rotation_90(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        """Apply 90-degree rotation transformation"""
        return np.rot90(grid, k=1)
    
    def _detect_rotation_180(self, examples: List[Dict[str, Any]]) -> Tuple[float, Dict]:
        """Detect if the transformation is a 180-degree rotation"""
        score = 0.0
        count = 0
        
        for example in examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Check if input and output have the same shape
            if input_grid.shape != output_grid.shape:
                continue
            
            # Apply 180-degree rotation to input
            rotated = np.rot90(input_grid, k=2)
            
            # Calculate similarity
            total_cells = output_grid.size
            matching_cells = np.sum(rotated == output_grid)
            similarity = matching_cells / total_cells if total_cells > 0 else 0
            
            score += similarity
            count += 1
        
        # Calculate average score
        avg_score = score / count if count > 0 else 0.0
        
        return avg_score, {}
    
    def _apply_rotation_180(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        """Apply 180-degree rotation transformation"""
        return np.rot90(grid, k=2)
    
    def _detect_rotation_270(self, examples: List[Dict[str, Any]]) -> Tuple[float, Dict]:
        """Detect if the transformation is a 270-degree rotation"""
        score = 0.0
        count = 0
        
        for example in examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Check if dimensions are compatible with 270-degree rotation
            if input_grid.shape[0] != output_grid.shape[1] or input_grid.shape[1] != output_grid.shape[0]:
                continue
            
            # Apply 270-degree rotation to input
            rotated = np.rot90(input_grid, k=3)
            
            # Calculate similarity
            total_cells = output_grid.size
            matching_cells = np.sum(rotated == output_grid)
            similarity = matching_cells / total_cells if total_cells > 0 else 0
            
            score += similarity
            count += 1
        
        # Calculate average score
        avg_score = score / count if count > 0 else 0.0
        
        return avg_score, {}
    
    def _apply_rotation_270(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        """Apply 270-degree rotation transformation"""
        return np.rot90(grid, k=3)
    
    def _detect_color_mapping(self, examples: List[Dict[str, Any]]) -> Tuple[float, Dict]:
        """Detect if the transformation is a color mapping"""
        score = 0.0
        count = 0
        color_maps = []
        
        for example in examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Check if input and output have the same shape
            if input_grid.shape != output_grid.shape:
                continue
            
            # Try to find a color mapping
            color_map = {}
            is_valid_mapping = True
            
            for i in range(input_grid.shape[0]):
                for j in range(input_grid.shape[1]):
                    in_color = input_grid[i, j]
                    out_color = output_grid[i, j]
                    
                    if in_color in color_map:
                        if color_map[in_color] != out_color:
                            is_valid_mapping = False
                            break
                    else:
                        color_map[in_color] = out_color
                
                if not is_valid_mapping:
                    break
            
            if is_valid_mapping:
                score += 1.0
                color_maps.append(color_map)
            
            count += 1
        
        # Calculate average score
        avg_score = score / count if count > 0 else 0.0
        
        # Find the most common color map
        final_color_map = {}
        if color_maps:
            # Combine all color maps
            for color_map in color_maps:
                for in_color, out_color in color_map.items():
                    if in_color not in final_color_map:
                        final_color_map[in_color] = []
                    final_color_map[in_color].append(out_color)
            
            # Find most common mapping for each color
            for in_color, out_colors in final_color_map.items():
                if out_colors:
                    # Count occurrences of each output color
                    color_counts = {}
                    for color in out_colors:
                        color_counts[color] = color_counts.get(color, 0) + 1
                    
                    # Find most common
                    final_color_map[in_color] = max(color_counts.items(), key=lambda x: x[1])[0]
        
        return avg_score, {'color_map': final_color_map}
    
    def _apply_color_mapping(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        """Apply color mapping transformation"""
        if not params or 'color_map' not in params:
            return grid.copy()
        
        color_map = params['color_map']
        result = grid.copy()
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] in color_map:
                    result[i, j] = color_map[grid[i, j]]
        
        return result
    
    def _detect_object_movement(self, examples: List[Dict[str, Any]]) -> Tuple[float, Dict]:
        """Detect if the transformation involves object movement"""
        score = 0.0
        count = 0
        movements = []
        
        for example in examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Check if input and output have the same shape
            if input_grid.shape != output_grid.shape:
                continue
            
            # Find objects in input and output
            input_objects = self._find_objects(input_grid)
            output_objects = self._find_objects(output_grid)
            
            # If number of objects doesn't match, this is not a simple movement
            if len(input_objects) != len(output_objects):
                continue
            
            # Try to match objects and calculate movements
            matched_objects = 0
            example_movements = []
            
            for in_obj in input_objects:
                best_match = None
                best_match_score = -1
                
                for out_obj in output_objects:
                    # Check if objects have the same color
                    if in_obj['color'] != out_obj['color']:
                        continue
                    
                    # Check if objects have the same shape
                    if in_obj['shape'] != out_obj['shape']:
                        continue
                    
                    # Calculate movement
                    dy = out_obj['center'][0] - in_obj['center'][0]
                    dx = out_obj['center'][1] - in_obj['center'][1]
                    
                    # Calculate match score (higher is better)
                    match_score = 1.0 / (1.0 + abs(dy) + abs(dx))
                    
                    if match_score > best_match_score:
                        best_match = out_obj
                        best_match_score = match_score
                
                if best_match:
                    matched_objects += 1
                    dy = best_match['center'][0] - in_obj['center'][0]
                    dx = best_match['center'][1] - in_obj['center'][1]
                    example_movements.append((dy, dx))
            
            # Calculate score based on matched objects
            if input_objects:
                example_score = matched_objects / len(input_objects)
                score += example_score
                
                if example_score > 0.5:  # Only consider movements if most objects matched
                    movements.extend(example_movements)
            
            count += 1
        
        # Calculate average score
        avg_score = score / count if count > 0 else 0.0
        
        # Calculate average movement
        avg_dy, avg_dx = 0, 0
        if movements:
            avg_dy = sum(m[0] for m in movements) / len(movements)
            avg_dx = sum(m[1] for m in movements) / len(movements)
        
        return avg_score, {'dy': int(round(avg_dy)), 'dx': int(round(avg_dx))}
    
    def _apply_object_movement(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        """Apply object movement transformation"""
        if not params or 'dy' not in params or 'dx' not in params:
            return grid.copy()
        
        dy = params['dy']
        dx = params['dx']
        
        # Create a new grid with zeros
        result = np.zeros_like(grid)
        
        # Find objects in the grid
        objects = self._find_objects(grid)
        
        # Move each object
        for obj in objects:
            color = obj['color']
            mask = obj['mask']
            
            # Calculate new positions
            new_positions = []
            for pos in obj['positions']:
                new_y = pos[0] + dy
                new_x = pos[1] + dx
                
                # Check if new position is within bounds
                if 0 <= new_y < grid.shape[0] and 0 <= new_x < grid.shape[1]:
                    new_positions.append((new_y, new_x))
            
            # Set new positions in result grid
            for y, x in new_positions:
                result[y, x] = color
        
        return result
    
    def _find_objects(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Find objects in a grid using connected component analysis"""
        from scipy import ndimage
        
        objects = []
        
        # Find unique colors (excluding 0, which is typically background)
        colors = np.unique(grid)
        colors = colors[colors != 0]
        
        for color in colors:
            # Create binary mask for this color
            mask = (grid == color).astype(int)
            
            # Find connected components
            labeled, num_features = ndimage.label(mask)
            
            for i in range(1, num_features + 1):
                # Get positions of this object
                object_mask = (labeled == i)
                positions = list(zip(*np.where(object_mask)))
                
                # Calculate center of mass
                if positions:
                    center_y = sum(p[0] for p in positions) / len(positions)
                    center_x = sum(p[1] for p in positions) / len(positions)
                    
                    # Calculate shape (simplified as the set of positions)
                    shape = frozenset((p[0] - int(center_y), p[1] - int(center_x)) for p in positions)
                    
                    objects.append({
                        'color': int(color),
                        'mask': object_mask,
                        'positions': positions,
                        'center': (center_y, center_x),
                        'shape': shape
                    })
        
        return objects
    
    def _detect_pattern_completion(self, examples: List[Dict[str, Any]]) -> Tuple[float, Dict]:
        """Detect if the transformation is pattern completion"""
        score = 0.0
        count = 0
        patterns = []
        
        for example in examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Check if output is larger than input
            if output_grid.shape[0] < input_grid.shape[0] or output_grid.shape[1] < input_grid.shape[1]:
                continue
            
            # Check if input is a subset of output
            is_subset = True
            for i in range(input_grid.shape[0]):
                for j in range(input_grid.shape[1]):
                    if input_grid[i, j] != 0 and input_grid[i, j] != output_grid[i, j]:
                        is_subset = False
                        break
                if not is_subset:
                    break
            
            if is_subset:
                # Calculate pattern
                pattern = {}
                for i in range(output_grid.shape[0]):
                    for j in range(output_grid.shape[1]):
                        if output_grid[i, j] != 0:
                            # Check if this position is outside input or input is 0
                            if (i >= input_grid.shape[0] or j >= input_grid.shape[1] or 
                                input_grid[i, j] == 0):
                                # This is part of the completion pattern
                                pattern[(i, j)] = int(output_grid[i, j])
                
                patterns.append(pattern)
                score += 1.0
            
            count += 1
        
        # Calculate average score
        avg_score = score / count if count > 0 else 0.0
        
        # Combine patterns
        combined_pattern = {}
        if patterns:
            # Count occurrences of each position and color
            position_colors = {}
            for pattern in patterns:
                for pos, color in pattern.items():
                    if pos not in position_colors:
                        position_colors[pos] = {}
                    position_colors[pos][color] = position_colors[pos].get(color, 0) + 1
            
            # Find most common color for each position
            for pos, colors in position_colors.items():
                if colors:
                    combined_pattern[pos] = max(colors.items(), key=lambda x: x[1])[0]
        
        return avg_score, {'pattern': combined_pattern}
    
    def _apply_pattern_completion(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        """Apply pattern completion transformation"""
        if not params or 'pattern' not in params:
            return grid.copy()
        
        pattern = params['pattern']
        
        # Determine size of output grid
        max_y = max([pos[0] for pos in pattern.keys()]) if pattern else grid.shape[0] - 1
        max_x = max([pos[1] for pos in pattern.keys()]) if pattern else grid.shape[1] - 1
        
        # Create output grid
        output_shape = (max(grid.shape[0], max_y + 1), max(grid.shape[1], max_x + 1))
        result = np.zeros(output_shape, dtype=grid.dtype)
        
        # Copy input grid
        result[:grid.shape[0], :grid.shape[1]] = grid
        
        # Apply pattern
        for (y, x), color in pattern.items():
            if 0 <= y < output_shape[0] and 0 <= x < output_shape[1]:
                result[y, x] = color
        
        return result
    
    def _detect_cellular_automata(self, examples: List[Dict[str, Any]]) -> Tuple[float, Dict]:
        """Detect if the transformation follows cellular automata rules"""
        score = 0.0
        count = 0
        rules = []
        
        for example in examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Check if input and output have the same shape
            if input_grid.shape != output_grid.shape:
                continue
            
            # Try to find cellular automata rules
            example_rules = {}
            rule_coverage = 0
            
            for i in range(1, input_grid.shape[0] - 1):
                for j in range(1, input_grid.shape[1] - 1):
                    # Get 3x3 neighborhood
                    neighborhood = input_grid[i-1:i+2, j-1:j+2].copy()
                    center_value = neighborhood[1, 1]
                    neighborhood[1, 1] = -1  # Mark center to exclude from neighborhood
                    
                    # Get output value
                    output_value = output_grid[i, j]
                    
                    # Create rule key
                    neighborhood_tuple = tuple(neighborhood.flatten())
                    rule_key = (center_value, neighborhood_tuple)
                    
                    # Add rule
                    if rule_key not in example_rules:
                        example_rules[rule_key] = output_value
                        rule_coverage += 1
            
            # Calculate score based on rule coverage
            total_cells = (input_grid.shape[0] - 2) * (input_grid.shape[1] - 2)
            if total_cells > 0:
                example_score = rule_coverage / total_cells
                score += example_score
                rules.append(example_rules)
            
            count += 1
        
        # Calculate average score
        avg_score = score / count if count > 0 else 0.0
        
        # Combine rules
        combined_rules = {}
        if rules:
            # Count occurrences of each rule
            rule_counts = {}
            for rule_set in rules:
                for rule_key, output_value in rule_set.items():
                    if rule_key not in rule_counts:
                        rule_counts[rule_key] = {}
                    rule_counts[rule_key][output_value] = rule_counts[rule_key].get(output_value, 0) + 1
            
            # Find most common output for each rule
            for rule_key, outputs in rule_counts.items():
                if outputs:
                    combined_rules[rule_key] = max(outputs.items(), key=lambda x: x[1])[0]
        
        return avg_score, {'rules': combined_rules}
    
    def _apply_cellular_automata(self, grid: np.ndarray, params: Dict = None) -> np.ndarray:
        """Apply cellular automata transformation"""
        if not params or 'rules' not in params:
            return grid.copy()
        
        rules = params['rules']
        result = grid.copy()
        
        # Apply rules to each cell
        for i in range(1, grid.shape[0] - 1):
            for j in range(1, grid.shape[1] - 1):
                # Get 3x3 neighborhood
                neighborhood = grid[i-1:i+2, j-1:j+2].copy()
                center_value = neighborhood[1, 1]
                neighborhood[1, 1] = -1  # Mark center to exclude from neighborhood
                
                # Create rule key
                neighborhood_tuple = tuple(neighborhood.flatten())
                rule_key = (center_value, neighborhood_tuple)
                
                # Apply rule if it exists
                if rule_key in rules:
                    result[i, j] = rules[rule_key]
        
        return result