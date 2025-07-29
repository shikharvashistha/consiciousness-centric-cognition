#!/usr/bin/env python3
"""
Simple test of phi calculation in the real data system
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
    
    # Create test neural data with high mutual information
    n_samples = 20
    n_features = 10
    
    # Create data with high mutual information between subsystems
    print("\n📊 Testing with highly integrated data:")
    base_values = np.random.rand(n_samples)
    
    # Create two subsystems with high mutual information
    subsystem_a = np.zeros((n_samples, n_features // 2))
    subsystem_b = np.zeros((n_samples, n_features // 2))
    
    for i in range(n_samples):
        # Both subsystems depend on the same base value
        for j in range(n_features // 2):
            subsystem_a[i, j] = base_values[i] + 0.1 * np.random.rand()
            subsystem_b[i, j] = base_values[i] + 0.1 * np.random.rand()
    
    # Combine subsystems
    integrated_data = np.hstack((subsystem_a, subsystem_b))
    print(f"📊 Integrated data shape: {integrated_data.shape}")
    
    # Calculate phi
    phi = system.calculate_phi(integrated_data)
    print(f"🧠 Calculated Φ for integrated data: {phi:.6f}")
    
    # Create data with low mutual information between subsystems
    print("\n📊 Testing with segregated data:")
    
    # Create two independent subsystems
    subsystem_a = np.random.rand(n_samples, n_features // 2)
    subsystem_b = np.random.rand(n_samples, n_features // 2)
    
    # Combine subsystems
    segregated_data = np.hstack((subsystem_a, subsystem_b))
    print(f"📊 Segregated data shape: {segregated_data.shape}")
    
    # Calculate phi
    phi = system.calculate_phi(segregated_data)
    print(f"🧠 Calculated Φ for segregated data: {phi:.6f}")
    
    # Test mutual information calculation directly
    print("\n📊 Testing mutual information calculation:")
    
    # Calculate mutual information between subsystems
    mi = system.mutual_information(subsystem_a, subsystem_b)
    print(f"🧠 Mutual information between independent subsystems: {mi:.6f}")
    
    # Calculate mutual information between correlated subsystems
    base_values = np.random.rand(n_samples)
    correlated_a = np.zeros((n_samples, n_features // 2))
    correlated_b = np.zeros((n_samples, n_features // 2))
    
    for i in range(n_samples):
        # Both subsystems depend on the same base value
        for j in range(n_features // 2):
            correlated_a[i, j] = base_values[i] + 0.1 * np.random.rand()
            correlated_b[i, j] = base_values[i] + 0.1 * np.random.rand()
    
    mi = system.mutual_information(correlated_a, correlated_b)
    print(f"🧠 Mutual information between correlated subsystems: {mi:.6f}")
    
    # Test entropy calculation
    print("\n📊 Testing entropy calculation:")
    
    # Calculate entropy of random data
    random_data = np.random.rand(n_samples, n_features)
    entropy = system.entropy(random_data)
    print(f"🧠 Entropy of random data: {entropy:.6f}")
    
    # Calculate entropy of uniform data
    uniform_data = np.ones((n_samples, n_features))
    entropy = system.entropy(uniform_data)
    print(f"🧠 Entropy of uniform data: {entropy:.6f}")
    
    # Calculate entropy of structured data
    structured_data = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        structured_data[i] = np.linspace(0, 1, n_features)
    entropy = system.entropy(structured_data)
    print(f"🧠 Entropy of structured data: {entropy:.6f}")

if __name__ == "__main__":
    test_phi_calculation()