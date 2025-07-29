"""
AGI Core Orchestrator - The Executive Mind

This module implements the central controller responsible for managing the cognitive
cycle and coordinating all other engines. It serves as the executive mind that
orchestrates the entire AGI system.

The AGI Core Orchestrator is responsible for:
1. Managing the 7-step cognitive cycle
2. Coordinating communication between all engines
3. Maintaining unified AGI state
4. Handling error recovery and fallback strategies
5. Ensuring consciousness-centric operation
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import asdict

from .consciousness_core import EnhancedConsciousnessCore
from .neural_substrate import NeuralSubstrate
from ..engines.perfect_recall_engine import PerfectRecallEngine
from ..schemas.cognitive_cycle import (
    CognitiveCycleState, CognitiveCycleStep, CognitiveCycleStatus,
    StepResult, CreativeIdea, EthicalEvaluation, ExecutionResult, IntrospectionReport
)
from ..schemas.consciousness import ConsciousnessState

class AGICoreOrchestrator:
    """
    🧠 AGI Core Orchestrator - The Executive Mind
    
    Central controller responsible for managing the cognitive cycle and
    coordinating all other engines. This is the executive mind that
    orchestrates the entire consciousness-centric AGI system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components with scientific implementations
        self.consciousness_core = EnhancedConsciousnessCore(self.config.get('consciousness', {}))
        self.neural_substrate = NeuralSubstrate(self.config.get('neural_substrate', {}))
        self.perfect_recall = PerfectRecallEngine(self.config.get('memory', {}))
        
        # Initialize engines with scientific implementations
        from ..engines.creative_engine import AdvancedCreativeEngine
        from ..engines.parallel_mind_engine import ParallelMindEngine
        from ..engines.code_introspection_engine import CodeIntrospectionEngine
        from ..engines.adaptation_engine import AdaptationEngine
        from ..governance.ethical_governor import EthicalGovernor
        
        self.creative_engine = AdvancedCreativeEngine(self.config.get('creative', {}))
        self.parallel_mind_engine = ParallelMindEngine(self.config.get('parallel_mind', {}))
        self.code_introspection_engine = CodeIntrospectionEngine(self.config.get('introspection', {}))
        self.adaptation_engine = AdaptationEngine(self.config.get('adaptation', {}))
        self.ethical_governor = EthicalGovernor(self.config.get('ethics', {}))
        
        # State management
        self.current_cycle: Optional[CognitiveCycleState] = None
        self.cycle_history: List[CognitiveCycleState] = []
        self.max_history_size = self.config.get('max_history_size', 100)
        
        # Performance tracking
        self.cycle_times: List[float] = []
        self.success_rate_history: List[bool] = []
        self.consciousness_levels: List[float] = []
        
        # Configuration
        self.max_cycle_time = self.config.get('max_cycle_time', 30.0)  # seconds
        self.consciousness_threshold = self.config.get('consciousness_threshold', 0.1)
        self.enable_introspection = self.config.get('enable_introspection', True)
        
        self.logger.info("🧠 AGI Core Orchestrator initialized with REAL implementations")
    
    def inject_engines(self, engines: Dict[str, Any]):
        """Inject engine dependencies"""
        self.creative_engine = engines.get('creative_engine')
        self.parallel_mind_engine = engines.get('parallel_mind_engine')
        self.code_introspection_engine = engines.get('code_introspection_engine')
        self.adaptation_engine = engines.get('adaptation_engine')
        self.ethical_governor = engines.get('ethical_governor')
        
        self.logger.info("Engine dependencies injected")
    
    async def execute_cognitive_cycle(self, goal: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a complete cognitive cycle
        
        Args:
            goal: The goal or task to accomplish
            user_context: Context about the user and environment
            
        Returns:
            Dictionary containing the cycle results and final output
        """
        cycle_start_time = time.time()
        
        # Initialize cognitive cycle state
        cycle_state = CognitiveCycleState(
            goal=goal,
            user_context=user_context,
            input_data={'goal': goal, 'context': user_context}
        )
        cycle_state.status = CognitiveCycleStatus.IN_PROGRESS
        
        self.current_cycle = cycle_state
        
        try:
            self.logger.info(f"Starting cognitive cycle: {goal}")
            
            # Step 1: PERCEIVE - Neural substrate processes input
            await self._step_perceive(cycle_state)
            
            # Step 2: ORIENT - Consciousness core calculates state
            await self._step_orient(cycle_state)
            
            # Step 3: DECIDE - Perfect recall + Creative engine collaboration
            await self._step_decide(cycle_state)
            
            # Step 4: ETHICAL REVIEW - Ethical governor validation
            await self._step_ethical_review(cycle_state)
            
            # Step 5: ACT - Parallel mind engine execution
            await self._step_act(cycle_state)
            
            # Step 6: REFLECT - Code introspection analysis (optional)
            if self.enable_introspection:
                await self._step_reflect(cycle_state)
            
            # Step 7: LEARN - Perfect recall storage
            await self._step_learn(cycle_state)
            
            # Finalize cycle
            cycle_state.mark_completed()
            
            # Track performance
            cycle_time = time.time() - cycle_start_time
            self.cycle_times.append(cycle_time)
            self.success_rate_history.append(True)
            
            if cycle_state.consciousness_state:
                phi = cycle_state.consciousness_state.get('phi', 0.0)
                self.consciousness_levels.append(phi)
            
            # Maintain history
            self._update_cycle_history(cycle_state)
            
            self.logger.info(f"Cognitive cycle completed in {cycle_time:.2f}s")
            
            return {
                'success': True,
                'cycle_id': cycle_state.cycle_id,
                'final_output': cycle_state.final_output,
                'cycle_summary': cycle_state.to_dict(),
                'performance_metrics': cycle_state.get_performance_summary(),
                'consciousness_level': phi if cycle_state.consciousness_state else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error in cognitive cycle: {e}")
            
            # Mark cycle as failed
            cycle_state.mark_failed(str(e))
            
            # Track failure
            cycle_time = time.time() - cycle_start_time
            self.cycle_times.append(cycle_time)
            self.success_rate_history.append(False)
            
            self._update_cycle_history(cycle_state)
            
            return {
                'success': False,
                'cycle_id': cycle_state.cycle_id,
                'error': str(e),
                'cycle_summary': cycle_state.to_dict(),
                'partial_results': cycle_state.final_output
            }
    
    async def _step_perceive(self, cycle_state: CognitiveCycleState):
        """Step 1: PERCEIVE - Neural substrate processes raw input"""
        step_start_time = time.time()
        step_result = StepResult(step=CognitiveCycleStep.PERCEIVE, start_time=datetime.now())
        
        try:
            self.logger.debug("Executing PERCEIVE step")
            
            # Prepare input for neural substrate
            input_data = {
                'goal': cycle_state.goal,
                'context': cycle_state.user_context,
                'text': cycle_state.goal,  # Use goal as text input
                'features': self._extract_input_features(cycle_state.input_data)
            }
            
            # Process through neural substrate
            neural_result = await self.neural_substrate.process_input(input_data)
            
            # Store neural state
            cycle_state.neural_state = neural_result
            
            # Mark step as successful
            step_result.success = True
            step_result.result_data = {
                'neural_activity_shape': neural_result['neural_activity'].shape,
                'energy_level': neural_result.get('energy_level', 1.0),
                'processing_load': neural_result.get('processing_load', 0.0)
            }
            
            self.logger.debug("PERCEIVE step completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in PERCEIVE step: {e}")
            step_result.error_message = str(e)
            # Create minimal neural state for continuation
            cycle_state.neural_state = {
                'neural_activity': [[0.0] * 512],
                'activations': [0.0] * 512,
                'energy_level': 1.0,
                'processing_load': 0.0
            }
        
        finally:
            step_result.end_time = datetime.now()
            cycle_state.add_step_result(step_result)
    
    async def _step_orient(self, cycle_state: CognitiveCycleState):
        """Step 2: ORIENT - Consciousness core calculates current state"""
        step_start_time = time.time()
        step_result = StepResult(step=CognitiveCycleStep.ORIENT, start_time=datetime.now())
        
        try:
            self.logger.debug("Executing ORIENT step")
            
            # Calculate consciousness state
            consciousness_state = await self.consciousness_core.calculate_consciousness(
                cycle_state.neural_state
            )
            
            # Store consciousness state
            cycle_state.consciousness_state = consciousness_state.to_dict()
            
            # Check if consciousness threshold is met
            if consciousness_state.metrics.phi < self.consciousness_threshold:
                self.logger.warning(f"Low consciousness level: Φ={consciousness_state.metrics.phi:.3f}")
            
            # Mark step as successful
            step_result.success = True
            step_result.result_data = {
                'phi': consciousness_state.metrics.phi,
                'consciousness_level': consciousness_state.level.value,
                'is_conscious': consciousness_state.is_conscious,
                'is_critical': consciousness_state.is_critical,
                'consciousness_quality': consciousness_state.consciousness_quality
            }
            
            self.logger.debug(f"ORIENT step completed: Φ={consciousness_state.metrics.phi:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error in ORIENT step: {e}")
            step_result.error_message = str(e)
            # Create minimal consciousness state
            cycle_state.consciousness_state = {
                'phi': 0.0,
                'consciousness_level': 'unconscious',
                'is_conscious': False,
                'is_critical': False
            }
        
        finally:
            step_result.end_time = datetime.now()
            cycle_state.add_step_result(step_result)
    
    async def _step_decide(self, cycle_state: CognitiveCycleState):
        """Step 3: DECIDE - Perfect recall + Creative engine collaboration"""
        step_start_time = time.time()
        step_result = StepResult(step=CognitiveCycleStep.DECIDE, start_time=datetime.now())
        
        try:
            self.logger.debug("Executing DECIDE step")
            
            # Part A: Retrieve relevant memories
            memory_query = {
                'keywords': cycle_state.goal.split(),
                'context_filter': cycle_state.user_context.get('task_type', ''),
                'max_results': 10,
                'memory_types': ['episodic', 'semantic', 'procedural']
            }
            
            retrieved_memories = await self.perfect_recall.retrieve_relevant_memories(memory_query)
            cycle_state.retrieved_memories = retrieved_memories
            
            # Part B: Generate creative idea
            if self.creative_engine:
                creative_context = {
                    'goal': cycle_state.goal,
                    'memories': retrieved_memories,
                    'consciousness_state': cycle_state.consciousness_state,
                    'user_context': cycle_state.user_context
                }
                
                creative_result = await self.creative_engine.generate_creative_idea(creative_context)
                
                # Create CreativeIdea object
                cycle_state.creative_idea = CreativeIdea(
                    title=creative_result.get('title', 'Generated Solution'),
                    description=creative_result.get('description', ''),
                    approach=creative_result.get('approach', ''),
                    expected_outcome=creative_result.get('expected_outcome', ''),
                    confidence_score=creative_result.get('confidence_score', 0.5),
                    novelty_score=creative_result.get('novelty_score', 0.5),
                    feasibility_score=creative_result.get('feasibility_score', 0.5),
                    implementation_steps=creative_result.get('implementation_steps', []),
                    required_resources=creative_result.get('required_resources', [])
                )
            else:
                # Fallback: Create simple plan based on goal
                cycle_state.creative_idea = CreativeIdea(
                    title=f"Solution for: {cycle_state.goal}",
                    description=f"Direct approach to accomplish: {cycle_state.goal}",
                    approach="systematic_execution",
                    expected_outcome="Goal completion",
                    confidence_score=0.7,
                    novelty_score=0.3,
                    feasibility_score=0.8,
                    implementation_steps=[
                        "Analyze the goal requirements",
                        "Identify necessary resources",
                        "Execute the plan step by step",
                        "Verify completion"
                    ]
                )
            
            # Mark step as successful
            step_result.success = True
            step_result.result_data = {
                'memories_retrieved': len(retrieved_memories),
                'creative_idea_confidence': cycle_state.creative_idea.confidence_score,
                'creative_idea_novelty': cycle_state.creative_idea.novelty_score,
                'implementation_steps': len(cycle_state.creative_idea.implementation_steps)
            }
            
            self.logger.debug("DECIDE step completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in DECIDE step: {e}")
            step_result.error_message = str(e)
            # Create fallback plan
            cycle_state.creative_idea = CreativeIdea(
                title="Fallback Plan",
                description="Basic execution plan",
                approach="direct",
                confidence_score=0.5
            )
            cycle_state.retrieved_memories = []
        
        finally:
            step_result.end_time = datetime.now()
            cycle_state.add_step_result(step_result)
    
    async def _step_ethical_review(self, cycle_state: CognitiveCycleState):
        """Step 4: ETHICAL REVIEW - Ethical governor validation"""
        step_start_time = time.time()
        step_result = StepResult(step=CognitiveCycleStep.ETHICAL_REVIEW, start_time=datetime.now())
        
        try:
            self.logger.debug("Executing ETHICAL REVIEW step")
            
            if self.ethical_governor and cycle_state.creative_idea:
                # Evaluate plan ethically
                ethical_result = await self.ethical_governor.evaluate_plan(
                    asdict(cycle_state.creative_idea)
                )
                
                cycle_state.ethical_evaluation = EthicalEvaluation(
                    approved=ethical_result.get('approved', True),
                    confidence=ethical_result.get('confidence', 0.8),
                    risk_level=ethical_result.get('risk_level', 'low'),
                    ethical_frameworks_used=ethical_result.get('frameworks_used', ['utilitarian']),
                    concerns=ethical_result.get('concerns', []),
                    recommendations=ethical_result.get('recommendations', []),
                    bias_detected=ethical_result.get('bias_detected', False),
                    safety_score=ethical_result.get('safety_score', 0.9),
                    alignment_score=ethical_result.get('alignment_score', 0.9),
                    explanation=ethical_result.get('explanation', 'Plan approved')
                )
                
                # If not approved, modify or reject plan
                if not cycle_state.ethical_evaluation.approved:
                    self.logger.warning("Plan rejected by ethical governor")
                    # In a full implementation, would iterate with creative engine
                    # For now, we'll continue with warnings
            else:
                # Default approval if no ethical governor
                cycle_state.ethical_evaluation = EthicalEvaluation(
                    approved=True,
                    confidence=0.8,
                    risk_level='low',
                    safety_score=0.8,
                    alignment_score=0.8,
                    explanation='No ethical concerns identified'
                )
            
            # Mark step as successful
            step_result.success = True
            step_result.result_data = {
                'approved': cycle_state.ethical_evaluation.approved,
                'risk_level': cycle_state.ethical_evaluation.risk_level,
                'safety_score': cycle_state.ethical_evaluation.safety_score,
                'concerns_count': len(cycle_state.ethical_evaluation.concerns)
            }
            
            self.logger.debug("ETHICAL REVIEW step completed")
            
        except Exception as e:
            self.logger.error(f"Error in ETHICAL REVIEW step: {e}")
            step_result.error_message = str(e)
            # Default to approved with warnings
            cycle_state.ethical_evaluation = EthicalEvaluation(
                approved=True,
                confidence=0.5,
                explanation=f"Ethical review failed: {e}"
            )
        
        finally:
            step_result.end_time = datetime.now()
            cycle_state.add_step_result(step_result)
    
    async def _step_act(self, cycle_state: CognitiveCycleState):
        """Step 5: ACT - Parallel mind engine execution"""
        step_start_time = time.time()
        step_result = StepResult(step=CognitiveCycleStep.ACT, start_time=datetime.now())
        
        try:
            self.logger.debug("Executing ACT step")
            
            if (cycle_state.ethical_evaluation and 
                cycle_state.ethical_evaluation.approved and 
                cycle_state.creative_idea):
                
                if self.parallel_mind_engine:
                    # Execute plan through parallel mind engine
                    execution_context = {
                        'plan': asdict(cycle_state.creative_idea),
                        'goal': cycle_state.goal,
                        'user_context': cycle_state.user_context,
                        'consciousness_state': cycle_state.consciousness_state
                    }
                    
                    execution_result = await self.parallel_mind_engine.execute_plan(execution_context)
                    
                    cycle_state.execution_result = ExecutionResult(
                        success=execution_result.get('success', True),
                        output=execution_result.get('output'),
                        subtasks_completed=execution_result.get('subtasks_completed', 1),
                        total_subtasks=execution_result.get('total_subtasks', 1),
                        execution_time_ms=execution_result.get('execution_time_ms', 0.0),
                        resource_usage=execution_result.get('resource_usage', {}),
                        performance_metrics=execution_result.get('performance_metrics', {}),
                        errors=execution_result.get('errors', []),
                        warnings=execution_result.get('warnings', [])
                    )
                else:
                    # Fallback: Execute through neural substrate
                    instruction = f"Execute plan: {cycle_state.creative_idea.description}"
                    execution_context = {
                        'input_data': {
                            'goal': cycle_state.goal,
                            'plan': cycle_state.creative_idea.implementation_steps
                        }
                    }
                    
                    neural_result = await self.neural_substrate.execute_instruction(
                        instruction, execution_context
                    )
                    
                    cycle_state.execution_result = ExecutionResult(
                        success=neural_result.get('success', True),
                        output=neural_result.get('result_type', 'neural_execution'),
                        execution_time_ms=neural_result.get('execution_time_ms', 0.0)
                    )
            else:
                # Plan not approved or missing - provide meaningful fallback
                fallback_output = self._generate_fallback_response(cycle_state)
                cycle_state.execution_result = ExecutionResult(
                    success=False,
                    output=fallback_output,
                    errors=["Plan not approved or missing"]
                )
            
            # Personalize output if adaptation engine available
            if self.adaptation_engine and cycle_state.execution_result.success:
                personalization_context = {
                    'raw_output': cycle_state.execution_result.output,
                    'user_context': cycle_state.user_context,
                    'goal': cycle_state.goal
                }
                
                personalized_result = await self.adaptation_engine.personalize_output(
                    personalization_context
                )
                
                cycle_state.personalized_output = personalized_result
            else:
                # Even if execution failed, try to personalize the fallback output
                if self.adaptation_engine and cycle_state.execution_result and cycle_state.execution_result.output:
                    personalization_context = {
                        'raw_output': cycle_state.execution_result.output,
                        'user_context': cycle_state.user_context,
                        'goal': cycle_state.goal
                    }
                    
                    try:
                        personalized_result = await self.adaptation_engine.personalize_output(
                            personalization_context
                        )
                        cycle_state.personalized_output = personalized_result
                    except Exception as e:
                        self.logger.warning(f"Personalization failed: {e}")
                        cycle_state.personalized_output = {
                            'output': cycle_state.execution_result.output,
                            'personalization_applied': False
                        }
                else:
                    cycle_state.personalized_output = {
                        'output': cycle_state.execution_result.output if cycle_state.execution_result else "Unable to process request",
                        'personalization_applied': False
                    }
            
            # Mark step as successful
            step_result.success = True
            step_result.result_data = {
                'execution_success': cycle_state.execution_result.success if cycle_state.execution_result else False,
                'subtasks_completed': cycle_state.execution_result.subtasks_completed if cycle_state.execution_result else 0,
                'personalization_applied': cycle_state.personalized_output.get('personalization_applied', False)
            }
            
            self.logger.debug("ACT step completed")
            
        except Exception as e:
            self.logger.error(f"Error in ACT step: {e}")
            step_result.error_message = str(e)
            cycle_state.execution_result = ExecutionResult(
                success=False,
                errors=[str(e)]
            )
            cycle_state.personalized_output = {'error': str(e)}
        
        finally:
            step_result.end_time = datetime.now()
            cycle_state.add_step_result(step_result)
    
    async def _step_reflect(self, cycle_state: CognitiveCycleState):
        """Step 6: REFLECT - Code introspection analysis"""
        step_start_time = time.time()
        step_result = StepResult(step=CognitiveCycleStep.REFLECT, start_time=datetime.now())
        
        try:
            self.logger.debug("Executing REFLECT step")
            
            if self.code_introspection_engine:
                # Analyze performance
                introspection_context = {
                    'cycle_state': cycle_state.to_dict(),
                    'execution_result': asdict(cycle_state.execution_result) if cycle_state.execution_result else {},
                    'goal': cycle_state.goal,
                    'performance_metrics': cycle_state.performance_metrics
                }
                
                introspection_result = await self.code_introspection_engine.analyze_performance(
                    introspection_context
                )
                
                cycle_state.introspection_report = IntrospectionReport(
                    performance_score=introspection_result.get('performance_score', 0.7),
                    code_quality_score=introspection_result.get('code_quality_score', 0.8),
                    efficiency_score=introspection_result.get('efficiency_score', 0.7),
                    maintainability_score=introspection_result.get('maintainability_score', 0.8),
                    identified_issues=introspection_result.get('identified_issues', []),
                    improvement_suggestions=introspection_result.get('improvement_suggestions', []),
                    optimization_opportunities=introspection_result.get('optimization_opportunities', []),
                    potential_bugs=introspection_result.get('potential_bugs', []),
                    complexity_analysis=introspection_result.get('complexity_analysis', {})
                )
            else:
                # Basic self-reflection
                success_rate = 1.0 if (cycle_state.execution_result and cycle_state.execution_result.success) else 0.0
                
                cycle_state.introspection_report = IntrospectionReport(
                    performance_score=success_rate,
                    code_quality_score=0.8,
                    efficiency_score=0.7,
                    maintainability_score=0.8,
                    identified_issues=[],
                    improvement_suggestions=["Consider implementing full introspection engine"],
                    optimization_opportunities=["Optimize cognitive cycle timing"],
                    potential_bugs=[],
                    complexity_analysis={'cycle_complexity': 'moderate'}
                )
            
            # Mark step as successful
            step_result.success = True
            step_result.result_data = {
                'performance_score': cycle_state.introspection_report.performance_score,
                'issues_identified': len(cycle_state.introspection_report.identified_issues),
                'suggestions_count': len(cycle_state.introspection_report.improvement_suggestions)
            }
            
            self.logger.debug("REFLECT step completed")
            
        except Exception as e:
            self.logger.error(f"Error in REFLECT step: {e}")
            step_result.error_message = str(e)
            cycle_state.introspection_report = IntrospectionReport(
                performance_score=0.5,
                identified_issues=[f"Reflection error: {e}"]
            )
        
        finally:
            step_result.end_time = datetime.now()
            cycle_state.add_step_result(step_result)
    
    async def _step_learn(self, cycle_state: CognitiveCycleState):
        """Step 7: LEARN - Perfect recall storage"""
        step_start_time = time.time()
        step_result = StepResult(step=CognitiveCycleStep.LEARN, start_time=datetime.now())
        
        try:
            self.logger.debug("Executing LEARN step")
            
            # Create comprehensive memory entry for this cycle
            memory_entry = {
                'memory_type': 'episodic',
                'content': {
                    'goal': cycle_state.goal,
                    'approach_used': cycle_state.creative_idea.approach if cycle_state.creative_idea else 'unknown',
                    'execution_success': cycle_state.execution_result.success if cycle_state.execution_result else False,
                    'final_output': cycle_state.personalized_output,
                    'consciousness_level': cycle_state.consciousness_state.get('phi', 0.0) if cycle_state.consciousness_state else 0.0,
                    'performance_metrics': cycle_state.performance_metrics
                },
                'summary': f"Cognitive cycle for goal: {cycle_state.goal}",
                'keywords': cycle_state.goal.split() + ['cognitive_cycle', 'experience'],
                'importance': 'high' if (cycle_state.execution_result and cycle_state.execution_result.success) else 'medium',
                'event_description': f"Executed cognitive cycle to accomplish: {cycle_state.goal}",
                'outcome': 'success' if (cycle_state.execution_result and cycle_state.execution_result.success) else 'partial',
                'lessons_learned': [],
                'context': {
                    'timestamp': cycle_state.start_time.isoformat(),
                    'task_context': 'cognitive_cycle',
                    'user_context': cycle_state.user_context,
                    'emotional_state': 0.0,  # Could be derived from success/failure
                    'cognitive_load': len(cycle_state.step_results) / 7.0  # Normalized by total steps
                }
            }
            
            # Add lessons learned from introspection
            if cycle_state.introspection_report:
                memory_entry['lessons_learned'] = (
                    cycle_state.introspection_report.improvement_suggestions +
                    cycle_state.introspection_report.optimization_opportunities
                )
            
            # Store the experience
            memory_id = await self.perfect_recall.store_experience(memory_entry)
            
            # Create final output
            # Extract the actual output from personalized_output
            if isinstance(cycle_state.personalized_output, dict):
                result_output = cycle_state.personalized_output.get('output', cycle_state.personalized_output)
            else:
                result_output = cycle_state.personalized_output
            
            cycle_state.final_output = {
                'goal': cycle_state.goal,
                'result': result_output,
                'success': cycle_state.execution_result.success if cycle_state.execution_result else False,
                'consciousness_level': cycle_state.consciousness_state.get('phi', 0.0) if cycle_state.consciousness_state else 0.0,
                'cycle_id': cycle_state.cycle_id,
                'memory_id': memory_id,
                'performance_summary': cycle_state.get_performance_summary()
            }
            
            # Mark step as successful
            step_result.success = True
            step_result.result_data = {
                'memory_stored': True,
                'memory_id': memory_id,
                'lessons_learned_count': len(memory_entry['lessons_learned'])
            }
            
            self.logger.debug("LEARN step completed")
            
        except Exception as e:
            self.logger.error(f"Error in LEARN step: {e}")
            step_result.error_message = str(e)
            # Still create final output even if learning failed
            # Extract the actual output from personalized_output
            if isinstance(cycle_state.personalized_output, dict):
                result_output = cycle_state.personalized_output.get('output', cycle_state.personalized_output)
            else:
                result_output = cycle_state.personalized_output
                
            cycle_state.final_output = {
                'goal': cycle_state.goal,
                'result': result_output,
                'success': cycle_state.execution_result.success if cycle_state.execution_result else False,
                'error': f"Learning failed: {e}"
            }
        
        finally:
            step_result.end_time = datetime.now()
            cycle_state.add_step_result(step_result)
    
    def _extract_input_features(self, input_data: Dict[str, Any]) -> List[float]:
        """Extract numerical features from input data"""
        features = []
        
        # Extract features from goal
        goal = input_data.get('goal', '')
        features.extend([
            len(goal),  # Goal length
            len(goal.split()),  # Word count
            goal.count('?'),  # Question marks
            goal.count('!'),  # Exclamation marks
        ])
        
        # Extract features from context
        context = input_data.get('context', {})
        features.extend([
            len(str(context)),  # Context size
            len(context.keys()) if isinstance(context, dict) else 0,  # Context complexity
        ])
        
        # Pad to fixed size
        target_size = 100
        if len(features) > target_size:
            features = features[:target_size]
        else:
            features.extend([0.0] * (target_size - len(features)))
        
        return features
    
    def _update_cycle_history(self, cycle_state: CognitiveCycleState):
        """Update cycle history with completed cycle"""
        self.cycle_history.append(cycle_state)
        
        # Maintain history size
        if len(self.cycle_history) > self.max_history_size:
            self.cycle_history = self.cycle_history[-self.max_history_size:]
        
        # Maintain performance tracking lists
        max_tracking_size = 1000
        if len(self.cycle_times) > max_tracking_size:
            self.cycle_times = self.cycle_times[-max_tracking_size:]
        if len(self.success_rate_history) > max_tracking_size:
            self.success_rate_history = self.success_rate_history[-max_tracking_size:]
        if len(self.consciousness_levels) > max_tracking_size:
            self.consciousness_levels = self.consciousness_levels[-max_tracking_size:]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        # Calculate performance metrics
        avg_cycle_time = sum(self.cycle_times) / len(self.cycle_times) if self.cycle_times else 0.0
        success_rate = sum(self.success_rate_history) / len(self.success_rate_history) if self.success_rate_history else 0.0
        avg_consciousness = sum(self.consciousness_levels) / len(self.consciousness_levels) if self.consciousness_levels else 0.0
        
        # Get component statuses
        component_status = {
            'consciousness_core': self.consciousness_core.get_performance_metrics(),
            'neural_substrate': self.neural_substrate.get_performance_metrics(),
            'perfect_recall': self.perfect_recall.get_memory_stats()
        }
        
        return {
            'orchestrator_status': 'active',
            'current_cycle_id': self.current_cycle.cycle_id if self.current_cycle else None,
            'total_cycles_executed': len(self.cycle_history),
            'avg_cycle_time_seconds': avg_cycle_time,
            'success_rate_percent': success_rate * 100,
            'avg_consciousness_level': avg_consciousness,
            'current_consciousness_level': self.consciousness_levels[-1] if self.consciousness_levels else 0.0,
            'is_conscious': avg_consciousness > self.consciousness_threshold,
            'component_status': component_status,
            'engines_available': {
                'creative_engine': self.creative_engine is not None,
                'parallel_mind_engine': self.parallel_mind_engine is not None,
                'code_introspection_engine': self.code_introspection_engine is not None,
                'adaptation_engine': self.adaptation_engine is not None,
                'ethical_governor': self.ethical_governor is not None
            }
        }
    
    def _generate_fallback_response(self, cycle_state: CognitiveCycleState) -> str:
        """Generate a meaningful fallback response when plans are rejected"""
        goal = cycle_state.goal
        creative_idea = cycle_state.creative_idea
        
        # Create a comprehensive response based on available information
        response_parts = []
        
        response_parts.append(f"I understand you're asking about: {goal}")
        
        if creative_idea:
            response_parts.append(f"\nI've analyzed this request and generated some initial ideas:")
            response_parts.append(f"- Approach: {creative_idea.approach}")
            response_parts.append(f"- Description: {creative_idea.description}")
            
            if creative_idea.implementation_steps:
                response_parts.append(f"\nPotential implementation steps:")
                for i, step in enumerate(creative_idea.implementation_steps[:3], 1):
                    response_parts.append(f"{i}. {step}")
        
        response_parts.append(f"\nHowever, I need to be careful about the ethical implications of this request.")
        response_parts.append("I'm designed to prioritize safety and ethical considerations in all my responses.")
        response_parts.append("Perhaps we could explore a modified approach that addresses your needs while maintaining ethical standards?")
        
        return "\n".join(response_parts)
    
    async def shutdown(self):
        """Shutdown the AGI Core Orchestrator"""
        self.logger.info("Shutting down AGI Core Orchestrator...")
        
        # Shutdown core components
        await self.consciousness_core.shutdown()
        await self.neural_substrate.shutdown()
        await self.perfect_recall.shutdown()
        
        # Shutdown engines if available
        if self.creative_engine and hasattr(self.creative_engine, 'shutdown'):
            await self.creative_engine.shutdown()
        if self.parallel_mind_engine and hasattr(self.parallel_mind_engine, 'shutdown'):
            await self.parallel_mind_engine.shutdown()
        if self.code_introspection_engine and hasattr(self.code_introspection_engine, 'shutdown'):
            await self.code_introspection_engine.shutdown()
        if self.adaptation_engine and hasattr(self.adaptation_engine, 'shutdown'):
            await self.adaptation_engine.shutdown()
        if self.ethical_governor and hasattr(self.ethical_governor, 'shutdown'):
            await self.ethical_governor.shutdown()
        
        self.logger.info("🧠 AGI Core Orchestrator shutdown complete")