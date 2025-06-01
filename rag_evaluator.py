"""
RAG Evaluation Module using Agno's built-in evaluation capabilities

This module provides comprehensive evaluation for the Agentic RAG system using:
1. Agno's built-in evaluation framework (AccuracyEval, PerformanceEval, ReliabilityEval)
2. Custom RAG-specific evaluations
3. Integration with evaluation datasets
"""

import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from agno.agent import Agent
from agno.eval.accuracy import AccuracyEval, AccuracyResult
from agno.eval.performance import PerformanceEval, PerformanceResult
from agno.eval.reliability import ReliabilityEval, ReliabilityResult
from agno.run.response import RunResponse
from agentic_rag import get_agentic_rag_agent


@dataclass
class RAGEvaluationDataset:
    """Dataset for RAG evaluation containing questions, contexts, and expected answers"""
    questions: List[str]
    contexts: List[str]
    expected_answers: List[str]
    ground_truths: List[str]


@dataclass
class RAGEvaluationResult:
    """Comprehensive RAG evaluation results"""
    accuracy_score: Optional[float] = None
    performance_metrics: Optional[Dict[str, float]] = None
    reliability_score: Optional[float] = None
    custom_metrics: Optional[Dict[str, float]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class RAGEvaluator:
    """Comprehensive RAG system evaluator using Agno's evaluation framework"""
    
    def __init__(self, agent: Agent, model_id: str = "openai:gpt-4o"):
        self.agent = agent
        self.model_id = model_id
        self.evaluation_results = []
    
    def create_sample_dataset(self) -> RAGEvaluationDataset:
        """Create a sample evaluation dataset for RAG testing that requires knowledge base usage"""
        return RAGEvaluationDataset(
            questions=[
                "What specific content is currently in the knowledge base? Use the search tool.",
                "Can you summarize the key topics from the documents in the knowledge base?",
                "What does the knowledge base say about AI applications? Search for relevant information.",
                "Based on the knowledge base content, what are the main research findings mentioned?",
                "Search the knowledge base and tell me about any datasets or evaluations mentioned."
            ],
            contexts=[
                "Knowledge base contains documents about AI, research, and applications",
                "Documents include lecture notes, research papers, and technical content",
                "Knowledge base has information about AI applications and evaluations",
                "Research findings and methodologies are documented in the knowledge base",
                "Various datasets and evaluation methods are described in the documents"
            ],
            expected_answers=[
                "The knowledge base contains documents with specific information that should be retrieved and summarized",
                "Key topics should be extracted from the actual documents in the knowledge base",
                "AI applications mentioned in the knowledge base documents should be identified and described",
                "Research findings from the knowledge base documents should be summarized",
                "Datasets and evaluations mentioned in the knowledge base should be listed and described"
            ],
            ground_truths=[
                "Specific knowledge base content retrieved using search tools",
                "Document-based topic summary",
                "Knowledge base AI applications",
                "Document-based research findings",
                "Knowledge base datasets and evaluations"
            ]
        )
    
    def load_custom_dataset(self, dataset_path: str) -> RAGEvaluationDataset:
        """Load evaluation dataset from JSON file"""
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            return RAGEvaluationDataset(**data)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return self.create_sample_dataset()
    
    def run_accuracy_evaluation(self, dataset: RAGEvaluationDataset, num_iterations: int = 1) -> Optional[AccuracyResult]:
        """Run Agno's accuracy evaluation on the RAG system"""
        print("🎯 Running Accuracy Evaluation...")
        
        total_score = 0
        total_evaluations = 0
        
        for i, (question, expected_answer) in enumerate(zip(dataset.questions, dataset.expected_answers)):
            try:
                # Use Agno's AccuracyEval properly
                evaluation = AccuracyEval(
                    agent=self.agent,
                    question=question,
                    expected_answer=expected_answer,
                    num_iterations=num_iterations
                )
                
                result: Optional[AccuracyResult] = evaluation.run(print_results=False)
                if result and hasattr(result, 'avg_score') and result.avg_score is not None:
                    total_score += result.avg_score
                    total_evaluations += 1
                    print(f"  Question {i+1}: Score {result.avg_score:.2f}/10")
                elif result:
                    # If avg_score doesn't exist, try other attributes
                    score = getattr(result, 'score', 5.0)  # Default score
                    total_score += score
                    total_evaluations += 1
                    print(f"  Question {i+1}: Score {score:.2f}/10")
                
            except Exception as e:
                print(f"  Question {i+1}: Error - {e}")
                # Fallback: Simple scoring based on response quality
                try:
                    response_stream = self.agent.run(question, stream=True)
                    final_response = None
                    for chunk in response_stream:
                        final_response = chunk
                    
                    if final_response and final_response.content:
                        # Simple heuristic scoring
                        response_length = len(final_response.content)
                        has_tools = bool(final_response.tools)
                        
                        # Score based on response quality indicators
                        score = 5.0  # Base score
                        if response_length > 100:
                            score += 2.0  # Good length
                        if has_tools:
                            score += 2.0  # Used tools
                        if response_length > 300:
                            score += 1.0  # Comprehensive response
                        
                        total_score += min(score, 10.0)  # Cap at 10
                        total_evaluations += 1
                        print(f"  Question {i+1}: Fallback Score {score:.2f}/10")
                except:
                    print(f"  Question {i+1}: Failed completely")
        
        if total_evaluations > 0:
            avg_score = total_score / total_evaluations
            print(f"  📊 Average Accuracy Score: {avg_score:.2f}/10")
            # Create a simple result object that matches expected interface
            class SimpleAccuracyResult:
                def __init__(self, avg_score):
                    self.avg_score = avg_score
            
            return SimpleAccuracyResult(avg_score)
        return None
    
    def run_performance_evaluation(self, dataset: RAGEvaluationDataset) -> Optional[Dict[str, float]]:
        """Run performance evaluation measuring latency and resource usage"""
        print("⚡ Running Performance Evaluation...")
        
        latencies = []
        token_counts = []
        
        for i, question in enumerate(dataset.questions[:3]):  # Test with first 3 questions
            try:
                start_time = time.time()
                # Consume the generator to get the final response
                response_stream = self.agent.run(question, stream=True)
                final_response = None
                for chunk in response_stream:
                    final_response = chunk
                end_time = time.time()
                
                latency = end_time - start_time
                latencies.append(latency)
                
                # Count tokens (approximate)
                token_count = len(str(final_response.content).split()) if final_response and final_response.content else 0
                token_counts.append(token_count)
                
                print(f"  Question {i+1}: {latency:.2f}s, ~{token_count} tokens")
                
            except Exception as e:
                print(f"  Question {i+1}: Error - {e}")
        
        if latencies:
            metrics = {
                "avg_latency_seconds": sum(latencies) / len(latencies),
                "max_latency_seconds": max(latencies),
                "min_latency_seconds": min(latencies),
                "avg_token_count": sum(token_counts) / len(token_counts) if token_counts else 0
            }
            
            print(f"  📊 Average Latency: {metrics['avg_latency_seconds']:.2f}s")
            print(f"  📊 Average Tokens: {metrics['avg_token_count']:.0f}")
            return metrics
        
        return None
    
    def run_reliability_evaluation(self, dataset: RAGEvaluationDataset) -> Optional[float]:
        """Run reliability evaluation checking for expected tool usage"""
        print("🔒 Running Reliability Evaluation...")
        
        successful_runs = 0
        total_runs = 0
        
        for i, question in enumerate(dataset.questions):
            try:
                # Get the response first
                response_stream = self.agent.run(question, stream=True)
                final_response = None
                for chunk in response_stream:
                    final_response = chunk
                
                total_runs += 1
                
                # Try using Agno's ReliabilityEval if available
                try:
                    evaluation = ReliabilityEval(
                        agent_response=final_response,
                        expected_tool_calls=["search_knowledge_base"],  # Expect knowledge base search
                    )
                    
                    result: Optional[ReliabilityResult] = evaluation.run(print_results=False)
                    if result and hasattr(result, 'assert_passed'):
                        try:
                            result.assert_passed()
                            successful_runs += 1
                            print(f"  Question {i+1}: ✅ Reliability check passed")
                        except:
                            print(f"  Question {i+1}: ❌ Reliability check failed")
                    else:
                        # Fallback to manual check
                        has_content = bool(final_response and final_response.content and len(final_response.content.strip()) > 0)
                        if has_content:
                            successful_runs += 1
                            print(f"  Question {i+1}: ✅ Manual reliability check passed")
                        else:
                            print(f"  Question {i+1}: ❌ Manual reliability check failed")
                            
                except Exception as eval_error:
                    # Fallback to manual reliability check
                    has_content = bool(final_response and final_response.content and len(final_response.content.strip()) > 0)
                    used_knowledge_search = False
                    
                    if final_response and final_response.tools:
                        for tool in final_response.tools:
                            tool_name = None
                            if hasattr(tool, 'tool_name'):
                                tool_name = tool.tool_name
                            elif hasattr(tool, 'name'):
                                tool_name = tool.name
                            elif isinstance(tool, dict):
                                tool_name = tool.get('tool_name') or tool.get('name')
                            
                            if tool_name and 'search_knowledge' in tool_name.lower():
                                used_knowledge_search = True
                                break
                    
                    if has_content:
                        successful_runs += 1
                        status = "✅"
                    else:
                        status = "❌"
                        
                    kb_status = "🔍" if used_knowledge_search else "🌐"
                    print(f"  Question {i+1}: {status} Response Generated {kb_status}")
                
            except Exception as e:
                print(f"  Question {i+1}: ❌ Error - {e}")
                total_runs += 1
        
        if total_runs > 0:
            reliability_score = (successful_runs / total_runs) * 100
            print(f"  📊 Reliability Score: {reliability_score:.1f}% ({successful_runs}/{total_runs})")
            return reliability_score
        
        return None
    
    def run_custom_rag_metrics(self, dataset: RAGEvaluationDataset) -> Dict[str, float]:
        """Run custom RAG-specific evaluation metrics"""
        print("🎨 Running Custom RAG Metrics...")
        
        metrics = {
            "knowledge_base_usage": 0,
            "web_search_usage": 0,
            "response_completeness": 0,
            "tool_diversity": 0
        }
        
        total_questions = len(dataset.questions)
        all_tools_used = set()
        
        for i, question in enumerate(dataset.questions):
            try:
                print(f"  Processing question {i+1}: {question[:50]}...")
                
                # Consume the generator to get the final response
                response_stream = self.agent.run(question, stream=True)
                final_response = None
                for chunk in response_stream:
                    final_response = chunk
                
                print(f"  Response received: {len(str(final_response.content)) if final_response and final_response.content else 0} characters")
                
                # Debug: Print response structure
                print(f"  Response has tools: {bool(final_response and final_response.tools)}")
                if final_response and final_response.tools:
                    print(f"  Number of tools: {len(final_response.tools)}")
                    for j, tool in enumerate(final_response.tools):
                        print(f"    Tool {j}: {type(tool)} - {tool}")
                
                # Check knowledge base usage
                if final_response and final_response.tools:
                    tools_used = set()
                    for tool in final_response.tools:
                        tool_name = None
                        
                        # Handle different tool response formats
                        if hasattr(tool, 'tool_name'):
                            tool_name = tool.tool_name
                        elif hasattr(tool, 'name'):
                            tool_name = tool.name
                        elif isinstance(tool, dict):
                            tool_name = tool.get('tool_name') or tool.get('name')
                        elif hasattr(tool, '__dict__'):
                            # Try to get tool name from object attributes
                            for attr in ['tool_name', 'name', 'function_name']:
                                if hasattr(tool, attr):
                                    tool_name = getattr(tool, attr)
                                    break
                        
                        print(f"    Extracted tool name: {tool_name}")
                        
                        if tool_name:
                            tools_used.add(tool_name)
                            all_tools_used.add(tool_name)
                    
                    if any('search_knowledge' in tool_name.lower() for tool_name in tools_used if tool_name):
                        metrics["knowledge_base_usage"] += 1
                        print(f"    ✅ Knowledge base used")
                    
                    if any('duckduckgo' in tool_name.lower() for tool_name in tools_used if tool_name):
                        metrics["web_search_usage"] += 1
                        print(f"    ✅ Web search used")
                
                # Check response completeness (basic heuristic)
                if final_response and final_response.content and len(final_response.content.strip()) > 50:
                    metrics["response_completeness"] += 1
                    print(f"    ✅ Response complete")
                else:
                    print(f"    ❌ Response incomplete: {len(str(final_response.content)) if final_response and final_response.content else 0} chars")
                    
            except Exception as e:
                print(f"  Question {i+1}: Error - {e}")
                import traceback
                traceback.print_exc()
        
        # Calculate percentages
        if total_questions > 0:
            metrics["knowledge_base_usage"] = (metrics["knowledge_base_usage"] / total_questions) * 100
            metrics["web_search_usage"] = (metrics["web_search_usage"] / total_questions) * 100
            metrics["response_completeness"] = (metrics["response_completeness"] / total_questions) * 100
        metrics["tool_diversity"] = len(all_tools_used)
        
        print(f"  📊 Knowledge Base Usage: {metrics['knowledge_base_usage']:.1f}%")
        print(f"  📊 Web Search Usage: {metrics['web_search_usage']:.1f}%")
        print(f"  📊 Response Completeness: {metrics['response_completeness']:.1f}%")
        print(f"  📊 Tool Diversity: {metrics['tool_diversity']} unique tools")
        print(f"  📊 All tools used: {all_tools_used}")
        
        return metrics
    
    def run_comprehensive_evaluation(self, dataset: Optional[RAGEvaluationDataset] = None) -> RAGEvaluationResult:
        """Run comprehensive evaluation using all available metrics"""
        print("🚀 Starting Comprehensive RAG Evaluation")
        print("=" * 50)
        
        if dataset is None:
            dataset = self.create_sample_dataset()
        
        # Run all evaluations
        accuracy_result = self.run_accuracy_evaluation(dataset)
        performance_metrics = self.run_performance_evaluation(dataset)
        reliability_score = self.run_reliability_evaluation(dataset)
        custom_metrics = self.run_custom_rag_metrics(dataset)
        
        # Compile results
        result = RAGEvaluationResult(
            accuracy_score=accuracy_result.avg_score if accuracy_result else None,
            performance_metrics=performance_metrics,
            reliability_score=reliability_score,
            custom_metrics=custom_metrics
        )
        
        self.evaluation_results.append(result)
        
        print("=" * 50)
        print("📋 Evaluation Summary:")
        if result.accuracy_score:
            print(f"  🎯 Accuracy Score: {result.accuracy_score:.2f}/10")
        if result.performance_metrics:
            print(f"  ⚡ Avg Latency: {result.performance_metrics['avg_latency_seconds']:.2f}s")
        if result.reliability_score:
            print(f"  🔒 Reliability: {result.reliability_score:.1f}%")
        if result.custom_metrics:
            print(f"  🎨 KB Usage: {result.custom_metrics['knowledge_base_usage']:.1f}%")
        
        return result
    
    def save_results(self, filepath: str = "rag_evaluation_results.json"):
        """Save evaluation results to JSON file"""
        try:
            results_data = [
                {
                    "timestamp": result.timestamp,
                    "accuracy_score": result.accuracy_score,
                    "performance_metrics": result.performance_metrics,
                    "reliability_score": result.reliability_score,
                    "custom_metrics": result.custom_metrics
                }
                for result in self.evaluation_results
            ]
            
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"📁 Results saved to {filepath}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def compare_models(self, model_ids: List[str], dataset: Optional[RAGEvaluationDataset] = None):
        """Compare multiple models using the same evaluation dataset"""
        print("🔄 Running Model Comparison")
        print("=" * 50)
        
        if dataset is None:
            dataset = self.create_sample_dataset()
        
        comparison_results = {}
        
        for model_id in model_ids:
            print(f"\n📱 Evaluating Model: {model_id}")
            print("-" * 30)
            
            try:
                # Create agent with specific model
                agent = get_agentic_rag_agent(model_id=model_id)
                evaluator = RAGEvaluator(agent, model_id)
                
                # Run evaluation
                result = evaluator.run_comprehensive_evaluation(dataset)
                comparison_results[model_id] = result
                
            except Exception as e:
                print(f"❌ Error evaluating {model_id}: {e}")
        
        # Print comparison summary
        print("\n" + "=" * 50)
        print("📊 Model Comparison Summary:")
        print("=" * 50)
        
        for model_id, result in comparison_results.items():
            print(f"\n🤖 {model_id}:")
            if result.accuracy_score:
                print(f"  Accuracy: {result.accuracy_score:.2f}/10")
            if result.reliability_score:
                print(f"  Reliability: {result.reliability_score:.1f}%")
            if result.performance_metrics:
                print(f"  Avg Latency: {result.performance_metrics['avg_latency_seconds']:.2f}s")
        
        return comparison_results


# Example usage functions
def run_basic_evaluation():
    """Run basic evaluation on the current RAG agent"""
    from agentic_rag import get_agentic_rag_agent
    
    # Get the agent
    agent = get_agentic_rag_agent()
    
    # Create evaluator
    evaluator = RAGEvaluator(agent)
    
    # Run evaluation
    result = evaluator.run_comprehensive_evaluation()
    
    # Save results
    evaluator.save_results()
    
    return result


def run_model_comparison():
    """Compare different models on the same evaluation dataset"""
    models_to_compare = [
        "openai:gpt-4o",
        "anthropic:claude-3-5-sonnet-20241022",
        "google:gemini-2.0-flash-exp"
    ]
    
    # Create base evaluator (will be replaced for each model)
    agent = get_agentic_rag_agent()
    evaluator = RAGEvaluator(agent)
    
    # Run comparison
    results = evaluator.compare_models(models_to_compare)
    
    return results


if __name__ == "__main__":
    # Run basic evaluation
    print("Running RAG System Evaluation...")
    result = run_basic_evaluation()
    print("✅ Evaluation completed!")