"""Test evaluation framework and metrics collection"""

import sys
from pathlib import Path
import time
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.evaluation import RAGEvaluator, QueryMetricsTracker, get_evaluator


def test_retrieval_metrics():
    """Test retrieval precision and recall calculations"""
    print("\n" + "=" * 70)
    print("TEST 1: Retrieval Metrics (Precision & Recall)")
    print("=" * 70)
    
    evaluator = RAGEvaluator()
    
    # Test case 1: Perfect retrieval
    print(f"\n Test Case 1: Perfect Retrieval")
    precision = evaluator.calculate_precision(5, 5)  # All 5 retrieved are relevant
    recall = evaluator.calculate_recall(5, 5)        # All 5 relevant were retrieved
    f1 = evaluator.calculate_f1_score(precision, recall)
    
    print(f"  Precision (5/5): {precision:.3f}")
    print(f"  Recall (5/5): {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    assert precision == 1.0 and recall == 1.0 and f1 == 1.0
    print(f"  ✓ Perfect retrieval metrics correct")
    
    # Test case 2: Partial retrieval
    print(f"\n Test Case 2: Partial Retrieval")
    precision = evaluator.calculate_precision(3, 5)  # 3 of 5 retrieved are relevant
    recall = evaluator.calculate_recall(3, 8)        # Retrieved 3 of 8 relevant
    f1 = evaluator.calculate_f1_score(precision, recall)
    
    print(f"  Precision (3/5): {precision:.3f}")
    print(f"  Recall (3/8): {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    assert precision == 0.6 and recall == 0.375
    print(f"  ✓ Partial retrieval metrics correct")
    
    # Test case 3: No retrieval
    print(f"\n Test Case 3: No Relevant Retrieval")
    precision = evaluator.calculate_precision(0, 5)  # 0 of 5 are relevant
    recall = evaluator.calculate_recall(0, 5)        # Retrieved 0 of 5 relevant
    f1 = evaluator.calculate_f1_score(precision, recall)
    
    print(f"  Precision (0/5): {precision:.3f}")
    print(f"  Recall (0/5): {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    assert precision == 0.0 and recall == 0.0 and f1 == 0.0
    print(f"  ✓ No retrieval metrics correct")


def test_hallucination_rate():
    """Test hallucination rate estimation"""
    print("\n" + "=" * 70)
    print("TEST 2: Hallucination Rate Estimation")
    print("=" * 70)
    
    evaluator = RAGEvaluator()
    
    # Test case 1: High confidence answer
    print(f"\n Test Case 1: High Confidence (Low Hallucination Risk)")
    reflection_high = {
        "confidence": 0.95,
        "hallucination_risk": "low"
    }
    rate = evaluator.estimate_hallucination_rate(reflection_high)
    print(f"  Confidence: {reflection_high['confidence']}")
    print(f"  Risk Level: {reflection_high['hallucination_risk']}")
    print(f"  Hallucination Rate: {rate:.3f}")
    assert rate <= 0.1, "High confidence should have low hallucination rate"
    print(f"  ✓ High confidence correctly maps to low hallucination")
    
    # Test case 2: Medium confidence answer
    print(f"\n Test Case 2: Medium Confidence")
    reflection_med = {
        "confidence": 0.65,
        "hallucination_risk": "medium"
    }
    rate = evaluator.estimate_hallucination_rate(reflection_med)
    print(f"  Confidence: {reflection_med['confidence']}")
    print(f"  Risk Level: {reflection_med['hallucination_risk']}")
    print(f"  Hallucination Rate: {rate:.3f}")
    assert 0.2 < rate < 0.4, "Medium confidence should have medium hallucination rate"
    print(f"  ✓ Medium confidence correctly estimated")
    
    # Test case 3: High hallucination risk
    print(f"\n Test Case 3: High Hallucination Risk")
    reflection_high_risk = {
        "confidence": 0.4,
        "hallucination_risk": "high"
    }
    rate = evaluator.estimate_hallucination_rate(reflection_high_risk)
    print(f"  Confidence: {reflection_high_risk['confidence']}")
    print(f"  Risk Level: {reflection_high_risk['hallucination_risk']}")
    print(f"  Hallucination Rate: {rate:.3f}")
    assert rate >= 0.7, "High risk should have high hallucination rate"
    print(f"  ✓ High risk correctly identified")


def test_cost_estimation():
    """Test cost estimation per query"""
    print("\n" + "=" * 70)
    print("TEST 3: Cost Estimation")
    print("=" * 70)
    
    evaluator = RAGEvaluator()
    
    # Test case 1: Simple query
    print(f"\n Test Case 1: Simple Query (1 retrieval, 1 LLM call)")
    cost = evaluator.estimate_cost_per_query(
        num_retrievals=1,
        num_llm_calls=1,
        num_regenerations=0
    )
    print(f"  Cost: ${cost:.4f}")
    assert 0.0005 < cost < 0.001, "Simple query should cost ~$0.0006"
    print(f"  ✓ Simple query cost reasonable")
    
    # Test case 2: Multi-hop query
    print(f"\n Test Case 2: Multi-Hop Query (4 retrievals, 1 LLM call)")
    cost = evaluator.estimate_cost_per_query(
        num_retrievals=4,
        num_llm_calls=1,
        num_regenerations=0
    )
    print(f"  Cost: ${cost:.4f}")
    assert cost > 0.0006, "Multi-hop should cost more"
    print(f"  ✓ Multi-hop query cost reflects complexity")
    
    # Test case 3: Query with regenerations
    print(f"\n Test Case 3: Query with Regenerations (1 retrieval, 3 LLM calls)")
    cost = evaluator.estimate_cost_per_query(
        num_retrievals=1,
        num_llm_calls=3,
        num_regenerations=2
    )
    print(f"  Cost: ${cost:.4f}")
    assert cost > 0.0015, "Query with regenerations should cost more"
    print(f"  ✓ Regeneration cost accounted for")


def test_query_logging():
    """Test query logging functionality"""
    print("\n" + "=" * 70)
    print("TEST 4: Query Logging")
    print("=" * 70)
    
    evaluator = RAGEvaluator()
    
    # Log multiple queries
    queries = [
        {
            "query": "What is machine learning?",
            "docs": ["ml_intro.pdf", "ai_guide.pdf"],
            "answer": "Machine learning is a subset of AI...",
            "latency": 245.5,
            "cost": 0.0006
        },
        {
            "query": "How do neural networks work?",
            "docs": ["neural_nets.pdf", "deep_learning.pdf"],
            "answer": "Neural networks are inspired by biology...",
            "latency": 312.3,
            "cost": 0.0009
        },
    ]
    
    print(f"\nLogging {len(queries)} test queries...")
    for q in queries:
        evaluator.log_query(
            query=q["query"],
            retrieved_docs=q["docs"],
            final_answer=q["answer"],
            latency_ms=q["latency"],
            cost=q["cost"]
        )
    
    # Check metrics updated
    print(f"\n✓ Logged {len(queries)} queries")
    assert evaluator.metrics["total_queries"] == len(queries)
    print(f"  Total Queries: {evaluator.metrics['total_queries']}")
    print(f"  Avg Latency: {evaluator.metrics['total_latency'] / len(queries):.1f} ms")
    print(f"  Total Cost: ${evaluator.metrics['total_cost']:.4f}")


def test_metrics_summary():
    """Test metrics summary calculation"""
    print("\n" + "=" * 70)
    print("TEST 5: Metrics Summary")
    print("=" * 70)
    
    evaluator = RAGEvaluator()
    
    # Log some queries
    for i in range(5):
        evaluator.log_query(
            query=f"Test query {i}",
            retrieved_docs=["doc1.pdf", "doc2.pdf"],
            final_answer=f"Test answer {i}",
            latency_ms=100.0 + (i * 50),
            cost=0.0006
        )
    
    # Get summary
    summary = evaluator.get_metrics_summary()
    
    print(f"\n Metrics Summary:")
    print(f"  Total Queries: {summary.get('total_queries')}")
    print(f"  Successful: {summary.get('successful_queries')}")
    print(f"  Failed: {summary.get('failed_queries')}")
    print(f"  Success Rate: {summary.get('success_rate', 0):.1%}")
    print(f"  Avg Latency: {summary.get('avg_latency_ms')} ms")
    print(f"  Total Cost: ${summary.get('total_cost'):.4f}")
    print(f"  Avg Cost: ${summary.get('avg_cost_usd'):.4f}")
    
    # Verify calculations
    assert summary['total_queries'] == 5
    assert summary['successful_queries'] == 5
    assert summary['success_rate'] == 1.0
    print(f"\n✓ Metrics summary calculations correct")


def test_query_metrics_tracker():
    """Test per-query metrics tracking"""
    print("\n" + "=" * 70)
    print("TEST 6: Query Metrics Tracker")
    print("=" * 70)
    
    evaluator = RAGEvaluator()
    query = "Test question about RAG systems"
    
    print(f"\nTracking metrics for: {query}")
    tracker = QueryMetricsTracker(query, evaluator)
    
    # Simulate phased execution
    tracker.start_phase("retrieval")
    time.sleep(0.05)  # 50ms
    tracker.end_phase()
    
    tracker.start_phase("answer_generation")
    time.sleep(0.1)   # 100ms
    tracker.end_phase()
    
    # Log result
    log_entry = tracker.log_result(
        retrieved_docs=["doc1.pdf", "doc2.pdf"],
        final_answer="Generated answer text",
        metadata={"reasoning_method": "hybrid"}
    )
    
    print(f"\n✓ Query execution tracked:")
    print(f"  Total Latency: {tracker.get_total_latency_ms():.1f} ms")
    print(f"  Retrieval: {tracker.phase_times.get('retrieval', 0):.1f} ms")
    print(f"  Answer Generation: {tracker.phase_times.get('answer_generation', 0):.1f} ms")
    print(f"  Estimated Cost: ${log_entry.get('cost_usd', 0):.4f}")
    
    assert tracker.get_total_latency_ms() > 150  # Should be > 150ms
    print(f"\n✓ Phase timing tracked correctly")


def test_logs_persistence():
    """Test saving and loading logs"""
    print("\n" + "=" * 70)
    print("TEST 7: Logs Persistence")
    print("=" * 70)
    
    evaluator = RAGEvaluator()
    
    # Log some queries
    evaluator.log_query(
        query="Test query",
        retrieved_docs=["doc1.pdf"],
        final_answer="Test answer",
        latency_ms=100,
        cost=0.0006
    )
    
    # Save logs
    print(f"\nSaving logs...")
    filepath = evaluator.save_logs(filename="test_logs.json")
    print(f"✓ Logs saved to {filepath}")
    
    # Load logs
    print(f"Loading logs...")
    loaded = evaluator.load_logs("test_logs.json")
    print(f"✓ Logs loaded successfully")
    
    # Verify content
    assert "summary" in loaded
    assert "query_logs" in loaded
    assert len(loaded["query_logs"]) == 1
    print(f"\n✓ Loaded data contains expected structure")
    print(f"  Queries logged: {len(loaded['query_logs'])}")
    print(f"  Total cost: ${loaded['summary']['total_cost']:.4f}")


def test_evaluation_framework_benefits():
    """Document benefits of evaluation framework"""
    print("\n" + "=" * 70)
    print("TEST 8: Evaluation Framework Benefits")
    print("=" * 70)
    
    print("""
EVALUATION FRAMEWORK BENEFITS:

1. RETRIEVAL QUALITY TRACKING
   ✓ Precision: Percentage of retrieved docs that are relevant
   ✓ Recall: Percentage of relevant docs that were retrieved
   ✓ F1 Score: Harmonic mean of precision and recall
   
   Metric Range:
   - 0.9-1.0: Excellent retrieval
   - 0.7-0.9: Good retrieval
   - 0.5-0.7: Acceptable retrieval
   - <0.5: Poor retrieval (needs improvement)

2. HALLUCINATION DETECTION & MEASUREMENT
   ✓ Tracks confidence scores from reflection mechanism
   ✓ Measures hallucination rate (estimated)
   ✓ Identifies when regeneration is needed
   ✓ Provides transparency on answer reliability
   
   Risk Levels:
   - LOW: Confidence > 0.8
   - MEDIUM: Confidence 0.6-0.8
   - HIGH: Confidence < 0.6

3. LATENCY MEASUREMENT
   ✓ Total query execution time
   ✓ Phase-level breakdown:
     * Retrieval time
     * Answer generation time
     * Database operations
     * Reflection/verification
   
   Targets:
   - < 500ms: Excellent
   - 500-1000ms: Good
   - 1-3s: Acceptable
   - > 3s: Needs optimization

4. COST TRACKING
   ✓ Per-query cost estimation
   ✓ Breakdown by operation:
     * Vector searches
     * LLM API calls
     * Regenerations
   
   Cost Reduction Strategies:
   - Use hybrid retrieval (balances cost/quality)
   - Limit regeneration attempts
   - Cache common queries
   - Monitor cost trends

5. COMPREHENSIVE LOGGING
   ✓ Every query logged with:
     * User question
     * Retrieved documents
     * Generated answer
     * Execution time
     * Final cost
     * Metadata (reasoning method, reflection)
   
   Use Cases:
   - Debugging and optimization
   - Quality assurance
   - Performance trending
   - Cost analysis
   - User behavior analysis

6. METRICS DASHBOARD
   ✓ Aggregate metrics:
     * Success rate
     * Average latency
     * Average cost
     * Hallucination rate
     * Precision/Recall trends
   
   Monitoring:
   - Track system health over time
   - Identify performance issues
   - Validate improvements
   - Cost management

7. PRODUCTION MONITORING
   ✓ Real-time metrics tracking
   ✓ Export and analysis capabilities
   ✓ Identify degradation patterns
   ✓ Support for SLA compliance
   
   Alerting Opportunities:
   - High latency (> 2s)
   - High hallucination rate (> 10%)
   - Low precision (< 0.6)
   - Cost spikes

8. CONTINUOUS IMPROVEMENT
   ✓ Data-driven optimization
   ✓ A/B testing support
   ✓ Version comparison tracking
   ✓ Feature impact measurement

REAL-WORLD EXAMPLE DASHBOARD:

Total Queries: 1,542
Success Rate: 98.5%

Performance:
  Avg Latency: 342 ms
  95th Percentile: 847 ms

Quality:
  Avg Precision: 0.82
  Avg Recall: 0.78
  Hallucination Rate: 2.1%

Cost:
  Avg per Query: $0.00087
  Daily Cost: ~$1.34
  Monthly Estimate: $40.26

Retrieval Methods Used:
  - Hybrid: 60% of queries
  - Multi-query: 30% of queries
  - Multi-hop: 10% of queries

Reflection Stats:
  Answers Verified: 98.5%
  Avg Regeneration Attempts: 0.15
  Verification Success: 97.1%

IMPLEMENTATION OPPORTUNITIES:
1. Set up dashboard for real-time monitoring
2. Configure alerting thresholds
3. Generate daily reports
4. Track improvements over time
5. Optimize based on metrics
    """)
    
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("EVALUATION FRAMEWORK TEST SUITE")
    print("=" * 70)
    
    try:
        test_retrieval_metrics()
        print("\n✅ Retrieval metrics tests passed!")
        
        test_hallucination_rate()
        print("\n✅ Hallucination rate tests passed!")
        
        test_cost_estimation()
        print("\n✅ Cost estimation tests passed!")
        
        test_query_logging()
        print("\n✅ Query logging tests passed!")
        
        test_metrics_summary()
        print("\n✅ Metrics summary tests passed!")
        
        test_query_metrics_tracker()
        print("\n✅ Query metrics tracker tests passed!")
        
        test_logs_persistence()
        print("\n✅ Logs persistence tests passed!")
        
        test_evaluation_framework_benefits()
        
        print("\n" + "=" * 70)
        print("EVALUATION FRAMEWORK IMPLEMENTATION COMPLETE ✓")
        print("=" * 70)
        print("""
IMPLEMENTATION SUMMARY:

✓ Retrieval Quality Metrics
  - Precision calculation
  - Recall calculation
  - F1 score (harmonic mean)
  - Per-query tracking

✓ Hallucination Rate Estimation
  - Based on reflection confidence
  - Risk level assessment
  - Trend tracking

✓ Latency Measurement
  - Total execution time
  - Phase-level timing
  - Performance tracking

✓ Cost Per Query Estimation
  - Vector search cost
  - LLM API call cost
  - Regeneration cost
  - Aggregate cost tracking

✓ Comprehensive Query Logging
  - User query
  - Retrieved documents
  - Final answer (first 500 chars)
  - Execution metrics
  - Metadata (reasoning method, reflection)
  - Timestamp and phase timings

✓ Metrics Aggregation
  - Success/failure rates
  - Average latency
  - Average cost
  - Hallucination rates
  - Precision/Recall averages

✓ Log Export & Analysis
  - Save logs to JSON
  - Load previous logs
  - Print summary reports
  - Support for analytics

✓ API Endpoints
  - GET /metrics - Full metrics
  - GET /metrics/summary - Concise summary
  - GET /logs/export - Export logs

✓ Integration with RAG Pipeline
  - Automatic query tracking
  - Phase timing capture
  - Metadata collection
  - Cost estimation

API EXAMPLES:

GET /metrics/summary
Response:
{
  "success": true,
  "summary": {
    "total_queries": 248,
    "success_rate": "98.5%",
    "avg_latency_ms": "342",
    "avg_cost_usd": "$0.00087",
    "hallucination_rate": "2.1%",
    "avg_precision": "0.82",
    "avg_recall": "0.78"
  }
}

GET /logs/export
Response:
{
  "success": true,
  "message": "Logs exported to ...",
  "filepath": "/path/to/logs/rag_logs_20260220_143052.json"
}

USE CASES:
1. Monitor system performance in production
2. Track cost and optimize queries
3. Measure improvement from new features
4. Debug quality issues
5. Ensure SLA compliance
6. Support business decision making

Next: Push to GitHub for production deployment
        """)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
