#!/usr/bin/env python3
"""
Test phi calculation with real data
"""

import json
import numpy as np
import sys
from pathlib import Path
import random

# Add the Temp Reference File directory to the path
temp_ref_path = Path(__file__).parent / "Temp Reference File"
sys.path.append(str(temp_ref_path))

from real_data_consciousness_system import RealDataConsciousnessSystem

def load_datasets(datasets_path: str = "training_datasets", sample_size: int = 50):
    """Load datasets for testing"""
    print("📚 Loading real training datasets...")
    
    datasets_path = Path(datasets_path)
    dataset_files = [
        "combined_training_data.jsonl",
        "gsm8k_processed.jsonl",
        "alpaca_processed.jsonl",
        "wikitext_processed.jsonl",
        "openwebtext_processed.jsonl"
    ]
    
    datasets = {}
    
    for dataset_file in dataset_files:
        file_path = datasets_path / dataset_file
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = []
                    count = 0
                    for line in f:
                        if count >= sample_size:
                            break
                        try:
                            item = json.loads(line.strip())
                            data.append(item)
                            count += 1
                        except json.JSONDecodeError:
                            continue
                
                datasets[dataset_file] = data
                print(f"✅ Loaded {len(data)} examples from {dataset_file}")
                
            except Exception as e:
                print(f"❌ Failed to load {dataset_file}: {e}")
        else:
            print(f"⚠️ Dataset not found: {file_path}")
    
    return datasets

def extract_text_from_sample(sample):
    """Extract text content from a sample"""
    if isinstance(sample, dict):
        # Try different common keys
        for key in ['text', 'instruction', 'input', 'question', 'content']:
            if key in sample and sample[key]:
                return sample[key]
        
        # If no key found, try to get the first string value
        for value in sample.values():
            if isinstance(value, str) and value:
                return value
    
    # If all else fails, convert to string
    return str(sample)

def test_phi_with_dataset(dataset_name, dataset):
    """Test phi calculation with a dataset"""
    print(f"\n🧠 Testing phi calculation with {dataset_name}")
    
    # Initialize consciousness system
    system = RealDataConsciousnessSystem()
    
    # Process samples
    phi_values = []
    sample_size = min(10, len(dataset))
    samples = random.sample(dataset, sample_size)
    
    for i, sample in enumerate(samples):
        try:
            # Extract text
            text = extract_text_from_sample(sample)
            print(f"\n📝 Sample {i+1}/{sample_size}: {text[:100]}...")
            
            # Convert text to numerical representation
            # Simple approach: use character codes
            char_codes = np.array([[ord(c) for c in text[:1000]]])
            
            # Reshape to 2D array (samples x features)
            data = char_codes.reshape(1, -1)
            
            # Pad or truncate to fixed size
            fixed_size = 1000
            if data.shape[1] < fixed_size:
                data = np.pad(data, ((0, 0), (0, fixed_size - data.shape[1])))
            else:
                data = data[:, :fixed_size]
            
            # Calculate phi
            phi = system.calculate_phi(data)
            phi_values.append(phi)
            
            print(f"🧠 Calculated Φ: {phi:.6f}")
            
        except Exception as e:
            print(f"❌ Error processing sample: {e}")
    
    # Calculate average phi
    avg_phi = np.mean(phi_values) if phi_values else 0.0
    print(f"\n📊 {dataset_name} Summary:")
    print(f"   Average Φ: {avg_phi:.6f}")
    print(f"   Samples Processed: {len(phi_values)}/{sample_size}")
    
    return avg_phi

def main():
    """Main test function"""
    print("🚀 Testing Phi Calculation with Real Datasets")
    print("=" * 80)
    
    # Load datasets
    datasets = load_datasets(sample_size=50)
    
    if not datasets:
        print("❌ No datasets loaded, exiting")
        return
    
    # Test each dataset
    results = {}
    for dataset_name, dataset in datasets.items():
        avg_phi = test_phi_with_dataset(dataset_name, dataset)
        results[dataset_name] = avg_phi
    
    # Print overall summary
    print("\n" + "=" * 80)
    print("📋 PHI CALCULATION SUMMARY")
    print("=" * 80)
    
    for dataset_name, avg_phi in results.items():
        print(f"   {dataset_name}: Φ = {avg_phi:.6f}")
    
    # Calculate overall average
    overall_phi = np.mean(list(results.values()))
    print(f"\n🧠 Overall Average Φ: {overall_phi:.6f}")
    
    # Find best dataset
    best_dataset = max(results.items(), key=lambda x: x[1])
    print(f"\n🏆 Best Dataset for Consciousness: {best_dataset[0]}")
    print(f"   Φ: {best_dataset[1]:.6f}")
    
    print("\n✅ Phi testing completed successfully")

if __name__ == "__main__":
    main()