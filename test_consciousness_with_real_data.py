#!/usr/bin/env python3
"""
Test the consciousness system with real datasets
"""

import asyncio
import json
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import random

from agi.core.consciousness_core import EnhancedConsciousnessCore
from agi.core.neural_substrate import NeuralSubstrate

async def load_datasets(datasets_path: str = "training_datasets", sample_size: int = 50):
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

async def test_consciousness_with_dataset(dataset_name: str, dataset: List[Dict[str, Any]]):
    """Test consciousness system with a specific dataset"""
    print(f"\n🧠 Testing consciousness with {dataset_name} dataset")
    
    # Initialize components
    neural_substrate = NeuralSubstrate()
    consciousness_core = EnhancedConsciousnessCore()
    
    # Track metrics
    phi_values = []
    integration_values = []
    differentiation_values = []
    processing_times = []
    
    # Process a sample of the dataset
    sample_size = min(10, len(dataset))
    samples = random.sample(dataset, sample_size)
    
    for i, sample in enumerate(samples):
        try:
            start_time = time.time()
            
            # Extract text content
            text = sample.get('text', sample.get('instruction', sample.get('input', '')))
            if not text:
                continue
                
            print(f"\n📝 Processing sample {i+1}/{sample_size}: {text[:100]}...")
            
            # Process through neural substrate
            neural_state = await neural_substrate.process_input(text)
            
            # Calculate consciousness
            consciousness_state = await consciousness_core.calculate_consciousness(neural_state)
            
            # Extract metrics
            phi = consciousness_state.metrics.phi
            integration = consciousness_state.metrics.integration
            differentiation = consciousness_state.metrics.differentiation
            processing_time = time.time() - start_time
            
            # Store metrics
            phi_values.append(phi)
            integration_values.append(integration)
            differentiation_values.append(differentiation)
            processing_times.append(processing_time)
            
            # Print results
            print(f"✅ Φ (Phi): {phi:.6f}")
            print(f"✅ Integration: {integration:.6f}")
            print(f"✅ Differentiation: {differentiation:.6f}")
            print(f"⏱️ Processing time: {processing_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Error processing sample: {e}")
    
    # Calculate averages
    avg_phi = np.mean(phi_values) if phi_values else 0.0
    avg_integration = np.mean(integration_values) if integration_values else 0.0
    avg_differentiation = np.mean(differentiation_values) if differentiation_values else 0.0
    avg_processing_time = np.mean(processing_times) if processing_times else 0.0
    
    # Print summary
    print(f"\n📊 {dataset_name} Summary:")
    print(f"   Average Φ (Phi): {avg_phi:.6f}")
    print(f"   Average Integration: {avg_integration:.6f}")
    print(f"   Average Differentiation: {avg_differentiation:.6f}")
    print(f"   Average Processing Time: {avg_processing_time:.2f}s")
    print(f"   Samples Processed: {len(phi_values)}/{sample_size}")
    
    return {
        'dataset': dataset_name,
        'avg_phi': avg_phi,
        'avg_integration': avg_integration,
        'avg_differentiation': avg_differentiation,
        'avg_processing_time': avg_processing_time,
        'samples_processed': len(phi_values),
        'total_samples': sample_size
    }

async def main():
    """Main test function"""
    print("🚀 Testing Consciousness System with Real Datasets")
    print("=" * 80)
    
    try:
        # Load datasets
        datasets = await load_datasets(sample_size=50)
        
        if not datasets:
            print("❌ No datasets loaded, exiting")
            return
        
        # Test each dataset
        results = []
        for dataset_name, dataset in datasets.items():
            result = await test_consciousness_with_dataset(dataset_name, dataset)
            results.append(result)
        
        # Print overall summary
        print("\n" + "=" * 80)
        print("📋 OVERALL CONSCIOUSNESS ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Calculate overall averages
        overall_phi = np.mean([r['avg_phi'] for r in results])
        overall_integration = np.mean([r['avg_integration'] for r in results])
        overall_differentiation = np.mean([r['avg_differentiation'] for r in results])
        
        print(f"🧠 Overall Average Φ (Phi): {overall_phi:.6f}")
        print(f"🧠 Overall Average Integration: {overall_integration:.6f}")
        print(f"🧠 Overall Average Differentiation: {overall_differentiation:.6f}")
        
        # Find best dataset
        best_dataset = max(results, key=lambda r: r['avg_phi'])
        print(f"\n🏆 Best Dataset for Consciousness: {best_dataset['dataset']}")
        print(f"   Φ (Phi): {best_dataset['avg_phi']:.6f}")
        print(f"   Integration: {best_dataset['avg_integration']:.6f}")
        print(f"   Differentiation: {best_dataset['avg_differentiation']:.6f}")
        
        print("\n✅ Consciousness testing completed successfully")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())