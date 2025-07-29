#!/usr/bin/env python3
"""
Test phi calculation directly
"""

import asyncio
import numpy as np
from agi.core.consciousness_core import EnhancedConsciousnessCore

async def test_phi():
    """Test phi calculation with sample data"""
    print("🧠 Testing Phi Calculation")
    
    # Initialize consciousness core
    core = EnhancedConsciousnessCore()
    
    # Test with sample neural data
    test_data = np.random.rand(64) * 2 - 1  # Random data between -1 and 1
    print(f"📊 Test data length: {len(test_data)}")
    print(f"📊 Test data range: [{test_data.min():.3f}, {test_data.max():.3f}]")
    
    # Calculate phi
    phi = core._calculate_phi(test_data)
    print(f"🧠 Calculated Φ: {phi:.6f}")
    
    # Test with different data sizes
    for size in [2, 4, 8, 16]:
        test_data_small = np.random.rand(size) * 2 - 1
        phi_small = core._calculate_phi(test_data_small)
        print(f"🧠 Φ for {size} elements: {phi_small:.6f}")
    
    await core.shutdown()

if __name__ == "__main__":
    asyncio.run(test_phi())