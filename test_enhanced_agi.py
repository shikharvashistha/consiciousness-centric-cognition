#!/usr/bin/env python3
"""
Enhanced AGI Test - Demonstrating Next-Phase Capabilities
"""

import os
import asyncio
import numpy as np
from test_arc_prize_integration import ARCAgent

async def test_enhanced_features():
    """Test the enhanced AGI features"""
    print("🚀 Enhanced AGI - Next Phase Capabilities Test")
    print("=" * 80)
    
    # Initialize the agent
    api_key = os.getenv("ARC_API_KEY")
    if not api_key:
        print("❌ ARC_API_KEY environment variable not set")
        return
    
    agent = ARCAgent(api_key)
    
    # Get available games
    print("🎮 Getting available games...")
    games = agent.client.get_available_games()
    
    if not games:
        print("❌ No games available or API connection failed")
        return
    
    print(f"✅ Found {len(games)} available games:")
    for game in games:
        print(f"   - {game.get('name', 'Unknown')} ({game.get('game_id', 'No ID')})")
    
    # Test enhanced features on first game
    test_game = games[0]
    print(f"\n🧠 Testing enhanced features with: {test_game.get('name', 'Unknown')}")
    
    # Play game with enhanced system
    results = await agent.play_game(test_game['game_id'], max_actions=15)
    
    if "error" in results:
        print(f"❌ Game failed: {results['error']}")
        return
    
    # Analyze enhanced capabilities
    print(f"\n{'='*80}")
    print("🎉 ENHANCED CAPABILITIES ANALYSIS")
    print(f"{'='*80}")
    
    # 1. Dynamic Consciousness Analysis
    phi_values = [c.get('phi_value', 0.0) for c in results['consciousness_log']]
    phi_range = max(phi_values) - min(phi_values)
    phi_variance = np.var(phi_values)
    
    print(f"🧠 Dynamic Consciousness:")
    print(f"   Average Φ: {results['average_phi']:.4f}")
    print(f"   Φ Range: {phi_range:.4f} ({'DYNAMIC' if phi_range > 0.05 else 'STATIC'})")
    print(f"   Φ Variance: {phi_variance:.6f}")
    
    # 2. Enhanced Visual Processing
    print(f"\n👁️ Enhanced Visual Processing:")
    if hasattr(agent, 'pattern_history') and agent.pattern_history:
        pattern_types = {}
        for pattern_info in agent.pattern_history:
            pattern_type = pattern_info["dominant_pattern"]
            if pattern_type not in pattern_types:
                pattern_types[pattern_type] = 0
            pattern_types[pattern_type] += 1
        
        print(f"   Pattern Types Detected:")
        for pattern, count in pattern_types.items():
            print(f"     - {pattern}: {count} instances")
    
    # 3. Learning System Analysis
    learning_insights = agent._get_learning_insights()
    print(f"\n🎓 Learning System:")
    print(f"   Experiences Recorded: {len(agent.action_history)}")
    
    if "optimal_phi_range" in learning_insights:
        print(f"   Optimal Φ Range: {learning_insights['optimal_phi_range']}")
        print(f"   Optimal Performance: {learning_insights['optimal_phi_performance']:.4f}")
    
    if "top_strategies" in learning_insights:
        print(f"   Top Strategies:")
        for strategy, score, count in learning_insights["top_strategies"][:2]:
            print(f"     - {strategy}: {score:.4f} (used {count}x)")
    
    # 4. ARC Strategy Integration
    print(f"\n🎯 ARC Strategy Integration:")
    print(f"   Strategy Performance Tracking: {len(agent.strategy_performance)} strategies")
    print(f"   Phi-Performance Mapping: {len(agent.phi_performance_map)} phi ranges")
    
    # 5. Action Diversity Analysis
    actions_used = set()
    for phi, action, score in agent.action_history:
        actions_used.add(action)
    
    print(f"\n⚡ Action Diversity:")
    print(f"   Unique Actions Used: {len(actions_used)}/6 possible")
    print(f"   Actions: {', '.join(sorted(actions_used))}")
    
    # 6. Consciousness-Performance Correlation
    if len(agent.action_history) > 5:
        phi_scores = [(phi, score) for phi, action, score in agent.action_history]
        phi_values_hist = [phi for phi, score in phi_scores]
        scores_hist = [score for phi, score in phi_scores]
        
        correlation = np.corrcoef(phi_values_hist, scores_hist)[0, 1] if len(phi_values_hist) > 1 else 0
        
        print(f"\n📊 Consciousness-Performance Correlation:")
        print(f"   Φ-Score Correlation: {correlation:.4f}")
        if abs(correlation) > 0.3:
            print(f"   Status: {'STRONG' if abs(correlation) > 0.7 else 'MODERATE'} correlation detected")
        else:
            print(f"   Status: WEAK correlation (more data needed)")
    
    # Summary
    print(f"\n{'='*80}")
    print("✅ ENHANCED CAPABILITIES SUMMARY")
    print(f"{'='*80}")
    
    enhancements = []
    if phi_range > 0.05:
        enhancements.append("✅ Dynamic Consciousness")
    else:
        enhancements.append("⚠️ Static Consciousness")
    
    if len(actions_used) > 2:
        enhancements.append("✅ Action Diversity")
    else:
        enhancements.append("⚠️ Limited Action Diversity")
    
    if len(agent.action_history) > 10:
        enhancements.append("✅ Active Learning")
    else:
        enhancements.append("⚠️ Limited Learning Data")
    
    if hasattr(agent, 'pattern_history') and len(agent.pattern_history) > 5:
        enhancements.append("✅ Pattern Recognition")
    else:
        enhancements.append("⚠️ Limited Pattern Data")
    
    for enhancement in enhancements:
        print(f"   {enhancement}")
    
    print(f"\n🔗 View detailed results:")
    print(f"   https://three.arcprize.org/scorecards/{results['scorecard_id']}")
    
    print(f"\n🎉 Enhanced AGI test completed!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_features())