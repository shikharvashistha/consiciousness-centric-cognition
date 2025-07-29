#!/usr/bin/env python3
"""
Test script for the phi calculation in real_data_consciousness_system.py
"""

import numpy as np
import sys
from pathlib import Path

# Add the Temp Reference File directory to the path
temp_ref_path = Path(__file__).parent / "Temp Reference File"
sys.path.append(str(temp_ref_path))

from real_data_consciousness_system import RealDataConsciousnessSystem

def test_phi_calculation():
    """Test phi calculation with sample data"""
    print("🧠 Testing Phi Calculation in RealDataConsciousnessSystem")
    
    # Initialize consciousness system
    system = RealDataConsciousnessSystem()
    
    # Create test neural data
    n_samples = 10
    n_features = 20
    test_data = np.random.rand(n_samples, n_features)
    print(f"📊 Test data shape: {test_data.shape}")
    
    # Calculate phi
    phi = system.calculate_phi(test_data)
    print(f"🧠 Calculated Φ: {phi:.6f}")
    
    # Test with different data sizes
    for size in [2, 4, 8, 16]:
        test_data_small = np.random.rand(n_samples, size)
        phi_small = system.calculate_phi(test_data_small)
        print(f"🧠 Φ for {size} features: {phi_small:.6f}")
    
    # Test with different sample sizes
    for size in [2, 5, 20, 50]:
        test_data_samples = np.random.rand(size, n_features)
        phi_samples = system.calculate_phi(test_data_samples)
        print(f"🧠 Φ for {size} samples: {phi_samples:.6f}")
    
    # Test with structured data (correlated features)
    structured_data = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        base = np.random.rand()
        for j in range(n_features):
            # Create correlated features
            structured_data[i, j] = base + 0.1 * np.random.rand()
    
    phi_structured = system.calculate_phi(structured_data)
    print(f"🧠 Φ for structured data: {phi_structured:.6f}")
    
    # Test with random data (uncorrelated features)
    random_data = np.random.rand(n_samples, n_features)
    phi_random = system.calculate_phi(random_data)
    print(f"🧠 Φ for random data: {phi_random:.6f}")
    
    # Test with highly integrated data
    integrated_data = np.zeros((n_samples, n_features))
    base_pattern = np.random.rand(n_features)
    for i in range(n_samples):
        # Small variations of the same pattern
        integrated_data[i] = base_pattern + 0.05 * np.random.rand(n_features)
    
    phi_integrated = system.calculate_phi(integrated_data)
    print(f"🧠 Φ for integrated data: {phi_integrated:.6f}")
    
    # Test with segregated data (independent subsystems)
    segregated_data = np.zeros((n_samples, n_features))
    mid_point = n_features // 2
    for i in range(n_samples):
        # Independent subsystems
        segregated_data[i, :mid_point] = np.random.rand(mid_point)
        segregated_data[i, mid_point:] = np.random.rand(n_features - mid_point)
    
    phi_segregated = system.calculate_phi(segregated_data)
    print(f"🧠 Φ for segregated data: {phi_segregated:.6f}")
    
    # Compare results
    print("\n📊 Comparison of Φ values:")
    print(f"  Structured data: {phi_structured:.6f}")
    print(f"  Random data: {phi_random:.6f}")
    print(f"  Integrated data: {phi_integrated:.6f}")
    print(f"  Segregated data: {phi_segregated:.6f}")

if __name__ == "__main__":
    test_phi_calculation()