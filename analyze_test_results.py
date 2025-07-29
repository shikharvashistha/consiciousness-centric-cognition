#!/usr/bin/env python3
"""
Analyze AGI test results and display summary table
"""

import json

def analyze_results():
    with open('agi_test_report.json', 'r') as f:
        data = json.load(f)
    
    print("🧠 AGI FULL COGNITIVE CYCLE TEST RESULTS")
    print("=" * 60)
    
    # Test Summary
    summary = data["test_summary"]
    print(f"📊 Total Test Cycles: {summary['total_cycles']}")
    print(f"✅ Success Rate: {summary['success_rate']:.1%}")
    print(f"⏱️ Total Test Time: {summary['total_test_time']:.2f}s")
    print(f"🔄 Average Cycle Time: {summary['average_cycle_time']:.2f}s")
    
    # Cognitive Analysis
    cognitive = data["cognitive_analysis"]
    print(f"🎨 Average Creativity Score: {cognitive['average_creativity']:.3f}")
    print(f"⚖️ Ethical Rejections: {cognitive['ethical_rejections']}")
    
    # Step Performance
    print("\n⚡ Step Performance (Average Times):")
    step_times = data["performance_metrics"]["step_averages"]
    for step, time in step_times.items():
        print(f"  {step.capitalize()}: {time:.3f}s")
    
    # System Status
    print(f"\n🎯 System Status:")
    if summary['success_rate'] >= 0.8:
        print("   ✅ EXCELLENT - System performing at high level")
    elif summary['success_rate'] >= 0.6:
        print("   ⚠️ GOOD - System performing adequately")
    else:
        print("   ❌ NEEDS IMPROVEMENT - System requires optimization")
    
    # Issues Found
    print(f"\n🔍 Issues Identified:")
    issues = []
    if cognitive['average_phi'] == 0.0:
        issues.append("Consciousness calculation (Φ) returning 0.0")
    if cognitive['average_quality'] == 0.0:
        issues.append("Quality assessment returning 0.0")
    if cognitive['ethical_rejections'] == summary['total_cycles']:
        issues.append("All plans rejected by ethical review")
    
    if issues:
        for issue in issues:
            print(f"   ⚠️ {issue}")
    else:
        print("   ✅ No critical issues detected")

if __name__ == "__main__":
    analyze_results() 