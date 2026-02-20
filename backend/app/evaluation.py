"""
Evaluation framework for RAG system.
Tracks metrics and logs for quality assessment and monitoring.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import os


class RAGEvaluator:
    """Comprehensive evaluation and logging system for RAG queries"""
    
    def __init__(self, log_dir: str = None):
        """
        Initialize evaluator with logging directory
        
        Args:
            log_dir: Directory to store evaluation logs (default: backend/logs/)
        """
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        
        self.log_dir = log_dir
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_latency": 0.0,
            "total_cost": 0.0,
            "hallucination_detections": 0,
        }
        
        # Per-query logs
        self.query_logs = []
        
        # Precision/recall tracking
        self.relevance_scores = []
        self.retrieval_quality_scores = []
        
        print(f"✓ RAG Evaluator initialized (logs: {self.log_dir})")
    
    def calculate_retrieval_recall(self, retrieved_docs: int, relevant_docs: int) -> float:
        """
        Calculate retrieval recall: retrieved relevant / total relevant
        
        Args:
            retrieved_docs: Number of relevant documents retrieved
            relevant_docs: Total number of relevant documents
        
        Returns:
            Recall score (0.0-1.0)
        """
        if relevant_docs == 0:
            return 1.0  # Perfect recall if no relevant docs exist
        
        recall = retrieved_docs / relevant_docs
        return min(1.0, recall)  # Cap at 1.0
    
    def calculate_precision(self, relevant_retrieved: int, total_retrieved: int) -> float:
        """
        Calculate precision: relevant retrieved / total retrieved
        
        Args:
            relevant_retrieved: Number of relevant docs in retrieved set
            total_retrieved: Total number of retrieved documents
        
        Returns:
            Precision score (0.0-1.0)
        """
        if total_retrieved == 0:
            return 0.0
        
        return relevant_retrieved / total_retrieved
    
    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """
        Calculate F1 score: harmonic mean of precision and recall
        
        Args:
            precision: Precision score
            recall: Recall score
        
        Returns:
            F1 score (0.0-1.0)
        """
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def estimate_hallucination_rate(self, reflection_data: Optional[Dict]) -> float:
        """
        Estimate hallucination rate based on reflection verification
        
        Args:
            reflection_data: Reflection metadata from answer generation
        
        Returns:
            Hallucination rate (0.0-1.0)
        """
        if not reflection_data:
            return 0.5  # Unknown risk if no reflection data
        
        # Calculate based on verification confidence
        confidence = reflection_data.get("confidence", 0.5)
        
        # Higher confidence = lower hallucination risk
        hallucination_rate = 1.0 - confidence
        
        # Adjust based on halluci_risk level
        risk_level = reflection_data.get("hallucination_risk", "medium")
        if risk_level == "high":
            hallucination_rate = max(0.7, hallucination_rate)
        elif risk_level == "low":
            hallucination_rate = min(0.1, hallucination_rate)
        
        return hallucination_rate
    
    def estimate_cost_per_query(self, 
                               num_retrievals: int = 1,
                               num_llm_calls: int = 1,
                               num_regenerations: int = 0) -> float:
        """
        Estimate cost per query based on API calls.
        
        Groq pricing (approximate):
        - Vector search: ~$0.0001 per search
        - LLM calls: ~$0.0005 per call
        
        Args:
            num_retrievals: Number of retrieval operations
            num_llm_calls: Number of LLM chat completions
            num_regenerations: Number of answer regenerations
        
        Returns:
            Estimated cost in USD
        """
        # Cost estimates (in USD)
        cost_per_vector_search = 0.0001
        cost_per_llm_call = 0.0005
        
        total_cost = (
            num_retrievals * cost_per_vector_search +
            num_llm_calls * cost_per_llm_call +
            num_regenerations * cost_per_llm_call
        )
        
        return total_cost
    
    def log_query(self,
                  query: str,
                  retrieved_docs: List[str],
                  final_answer: str,
                  latency_ms: float,
                  cost: float,
                  metadata: Optional[Dict] = None) -> Dict:
        """
        Log a complete query with all details
        
        Args:
            query: The user's question
            retrieved_docs: List of retrieved document sources
            final_answer: The generated answer
            latency_ms: Query execution time in milliseconds
            cost: Estimated cost in USD
            metadata: Additional metadata (reflection, reasoning_method, etc.)
        
        Returns:
            Log entry dictionary
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "retrieved_docs": retrieved_docs,
            "num_docs_retrieved": len(retrieved_docs),
            "final_answer": final_answer[:500],  # First 500 chars
            "answer_length": len(final_answer),
            "latency_ms": latency_ms,
            "cost_usd": cost,
            "metadata": metadata or {}
        }
        
        # Add to logs
        self.query_logs.append(log_entry)
        
        # Update metrics
        self.metrics["total_queries"] += 1
        self.metrics["total_latency"] += latency_ms
        self.metrics["total_cost"] += cost
        
        if metadata and metadata.get("reflection"):
            if not metadata["reflection"].get("verified", True):
                self.metrics["hallucination_detections"] += 1
        
        self.metrics["successful_queries"] += 1
        
        return log_entry
    
    def get_metrics_summary(self) -> Dict:
        """
        Get summary of all metrics
        
        Returns:
            Dictionary with computed metrics
        """
        total = self.metrics["total_queries"]
        if total == 0:
            return self.metrics
        
        avg_latency = self.metrics["total_latency"] / total
        avg_cost = self.metrics["total_cost"] / total
        hallucination_rate = self.metrics["hallucination_detections"] / total
        success_rate = self.metrics["successful_queries"] / total
        
        # Calculate retrieval metrics if available
        avg_retrieval_precision = 0.0
        avg_retrieval_recall = 0.0
        if self.retrieval_quality_scores:
            avg_retrieval_precision = sum(s.get("precision", 0) 
                                         for s in self.retrieval_quality_scores) / len(self.retrieval_quality_scores)
            avg_retrieval_recall = sum(s.get("recall", 0) 
                                      for s in self.retrieval_quality_scores) / len(self.retrieval_quality_scores)
        
        return {
            **self.metrics,
            "avg_latency_ms": round(avg_latency, 2),
            "avg_cost_usd": round(avg_cost, 4),
            "hallucination_rate": round(hallucination_rate, 3),
            "success_rate": round(success_rate, 3),
            "avg_retrieval_precision": round(avg_retrieval_precision, 3),
            "avg_retrieval_recall": round(avg_retrieval_recall, 3),
            "num_logged_queries": len(self.query_logs)
        }
    
    def save_logs(self, filename: str = None) -> str:
        """
        Save all logs to file
        
        Args:
            filename: Custom filename (default: rag_logs_<timestamp>.json)
        
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_logs_{timestamp}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        logs_data = {
            "exported_at": datetime.now().isoformat(),
            "summary": self.get_metrics_summary(),
            "query_logs": self.query_logs,
            "retrieval_quality_scores": self.retrieval_quality_scores
        }
        
        with open(filepath, 'w') as f:
            json.dump(logs_data, f, indent=2)
        
        print(f"✓ Logs saved to {filepath}")
        return filepath
    
    def load_logs(self, filename: str) -> Dict:
        """
        Load logs from file
        
        Args:
            filename: Filename to load
        
        Returns:
            Loaded logs data
        """
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return data
    
    def print_summary(self):
        """Print metrics summary to console"""
        summary = self.get_metrics_summary()
        
        print("\n" + "=" * 70)
        print("RAG SYSTEM EVALUATION SUMMARY")
        print("=" * 70)
        print(f"\n📊 BASIC METRICS:")
        print(f"  Total Queries: {summary.get('total_queries', 0)}")
        print(f"  Successful: {summary.get('successful_queries', 0)}")
        print(f"  Failed: {summary.get('failed_queries', 0)}")
        
        print(f"\n⏱️  PERFORMANCE METRICS:")
        print(f"  Avg Latency: {summary.get('avg_latency_ms', 0):.0f} ms")
        print(f"  Total Latency: {summary.get('total_latency', 0):.0f} ms")
        
        print(f"\n💰 COST METRICS:")
        print(f"  Avg Cost per Query: ${summary.get('avg_cost_usd', 0):.4f}")
        print(f"  Total Cost: ${summary.get('total_cost', 0):.4f}")
        
        print(f"\n🎯 QUALITY METRICS:")
        print(f"  Success Rate: {summary.get('success_rate', 0):.1%}")
        print(f"  Hallucination Rate: {summary.get('hallucination_rate', 0):.1%}")
        
        print(f"\n📚 RETRIEVAL METRICS:")
        print(f"  Avg Precision: {summary.get('avg_retrieval_precision', 0):.3f}")
        print(f"  Avg Recall: {summary.get('avg_retrieval_recall', 0):.3f}")
        
        print("\n" + "=" * 70)
    
    def record_retrieval_quality(self, 
                                precision: float,
                                recall: float,
                                f1: float,
                                query: str):
        """
        Record retrieval quality metrics
        
        Args:
            precision: Precision score
            recall: Recall score
            f1: F1 score
            query: Associated query
        """
        self.retrieval_quality_scores.append({
            "query": query,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "timestamp": datetime.now().isoformat()
        })


class QueryMetricsTracker:
    """Tracks metrics for a single query execution"""
    
    def __init__(self, query: str, evaluator: RAGEvaluator):
        """
        Initialize tracker for a query
        
        Args:
            query: The user's question
            evaluator: RAGEvaluator instance
        """
        self.query = query
        self.evaluator = evaluator
        self.start_time = time.time()
        
        # Track different phases
        self.phase_times = {}
        self.phase_start = None
    
    def start_phase(self, phase_name: str):
        """Start timing a phase"""
        self.phase_start = time.time()
        self.current_phase = phase_name
    
    def end_phase(self):
        """End timing a phase"""
        if self.phase_start:
            elapsed = (time.time() - self.phase_start) * 1000  # Convert to ms
            self.phase_times[self.current_phase] = elapsed
    
    def get_total_latency_ms(self) -> float:
        """Get total query latency in milliseconds"""
        return (time.time() - self.start_time) * 1000
    
    def log_result(self,
                   retrieved_docs: List[str],
                   final_answer: str,
                   metadata: Optional[Dict] = None):
        """
        Log the query result
        
        Args:
            retrieved_docs: List of retrieved document sources
            final_answer: Generated answer
            metadata: Additional metadata
        """
        latency_ms = self.get_total_latency_ms()
        
        # Estimate cost based on metadata
        num_retrievals = 1
        num_llm_calls = 1
        num_regenerations = 0
        
        if metadata:
            if "reasoning_method" in metadata:
                # Multi-hop uses more retrievals
                if metadata["reasoning_method"] == "multi_hop":
                    num_retrievals = 4  # Estimate for multi-hop
                elif metadata["reasoning_method"] == "multi_query":
                    num_retrievals = 4  # Multiple query variants
            
            if "reflection" in metadata:
                # Reflection adds verification LLM call
                num_llm_calls += 1
                num_regenerations = metadata["reflection"].get("regeneration_attempts", 0)
        
        cost = self.evaluator.estimate_cost_per_query(
            num_retrievals=num_retrievals,
            num_llm_calls=num_llm_calls,
            num_regenerations=num_regenerations
        )
        
        # Log with evaluator
        log_entry = self.evaluator.log_query(
            query=self.query,
            retrieved_docs=retrieved_docs,
            final_answer=final_answer,
            latency_ms=latency_ms,
            cost=cost,
            metadata=metadata
        )
        
        # Add phase timing info
        log_entry["phase_timing_ms"] = self.phase_times
        
        return log_entry


# Global evaluator instance
_global_evaluator = None


def get_evaluator(log_dir: str = None) -> RAGEvaluator:
    """Get or create global evaluator instance"""
    global _global_evaluator
    if _global_evaluator is None:
        _global_evaluator = RAGEvaluator(log_dir)
    return _global_evaluator
