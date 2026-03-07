"""Evaluation module for Wongnai QA System.

This module provides evaluation metrics and testing functions to assess
the quality of retrieval and answer generation. It includes metrics for
measuring accuracy, relevance, and overall system performance.

Key functions:
    - Calculate retrieval accuracy and relevance scores
    - Evaluate answer quality against ground truth
    - Benchmark system performance on test queries
    - Generate evaluation reports
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from collections import defaultdict

from src.retrieval import WongnaiRetriever
from src.qa_generator import WongnaiQAGenerator


def get_demo_queries() -> List[Dict]:
    """Return a list of demo queries covering 5 required categories.
    
    Categories:
        - cuisine: Queries about specific cuisines
        - food_type: Queries about food types/dishes
        - atmosphere_price: Queries about ambiance and pricing
        - location: Queries about specific locations
        - combined: Complex queries combining multiple aspects
    
    Returns:
        List of dictionaries with keys: category, query, description
    """
    queries = [
        # Cuisine category (5 queries)
        {
            "category": "cuisine",
            "query": "อาหารญี่ปุ่นอร่อย",
            "description": "ค้นหาร้านอาหารญี่ปุ่นที่ได้รับความนิยม"
        },
        {
            "category": "cuisine",
            "query": "ร้านอาหารจีนแนะนำ",
            "description": "ค้นหาร้านอาหารจีนที่แนะนำ"
        },
        {
            "category": "cuisine",
            "query": "อาหารอิตาลีดี",
            "description": "ค้นหาร้านอาหารอิตาลีคุณภาพดี"
        },
        {
            "category": "cuisine",
            "query": "ร้านอาหารไทยรสเด็ด",
            "description": "ค้นหาร้านอาหารไทยรสชาติจัดจ้าน"
        },
        {
            "category": "cuisine",
            "query": "อาหารอินเดียเผ็ด",
            "description": "ค้นหาร้านอาหารอินเดียรสเผ็ด"
        },
        
        # Food type category (5 queries)
        {
            "category": "food_type",
            "query": "อาหารทะเลสด",
            "description": "ค้นหาร้านอาหารทะเลสดใหม่"
        },
        {
            "category": "food_type",
            "query": "พิซซ่าอร่อย",
            "description": "ค้นหาร้านพิซซ่าอร่อย"
        },
        {
            "category": "food_type",
            "query": "ร้านเบเกอรี่ดี",
            "description": "ค้นหาร้านเบเกอรี่คุณภาพดี"
        },
        {
            "category": "food_type",
            "query": "ร้านกาแฟบรรยากาศดี",
            "description": "ค้นหาร้านกาแฟบรรยากาศดี"
        },
        {
            "category": "food_type",
            "query": "ก๋วยเตี๋ยวเส้นเล็ก",
            "description": "ค้นหาร้านก๋วยเตี๋ยวเส้นเล็กอร่อย"
        },
        
        # Atmosphere and price category (5 queries)
        {
            "category": "atmosphere_price",
            "query": "ร้านอาหารหรูหรา",
            "description": "ค้นหาร้านอาหารระดับหรู"
        },
        {
            "category": "atmosphere_price",
            "query": "ร้านข้างทางราคาถูก",
            "description": "ค้นหาร้านข้างทางราคาประหยัด"
        },
        {
            "category": "atmosphere_price",
            "query": "ร้านบรรยากาศดีติดแอร์",
            "description": "ค้นหาร้านอาหารบรรยากาศดีมีแอร์"
        },
        {
            "category": "atmosphere_price",
            "query": "คาเฟ่น่านั่ง",
            "description": "ค้นหาคาเฟ่บรรยากาศน่านั่ง"
        },
        {
            "category": "atmosphere_price",
            "query": "ร้านอาหารราคานักศึกษา",
            "description": "ค้นหาร้านอาหารราคาไม่แพง"
        },
        
        # Location category (5 queries)
        {
            "category": "location",
            "query": "ร้านอาหารเชียงใหม่",
            "description": "ค้นหาร้านอาหารในเชียงใหม่"
        },
        {
            "category": "location",
            "query": "ร้านติดทะเลพัทยา",
            "description": "ค้นหาร้านอาหารติดทะเลที่พัทยา"
        },
        {
            "category": "location",
            "query": "ร้านอาหารบนเขาใหญ่",
            "description": "ค้นหาร้านอาหารบรรยากาศเขาใหญ่"
        },
        {
            "category": "location",
            "query": "ร้านอาหารกรุงเทพ",
            "description": "ค้นหาร้านอาหารในกรุงเทพ"
        },
        {
            "category": "location",
            "query": "ร้านอาหารหัวหิน",
            "description": "ค้นหาร้านอาหารในหัวหิน"
        },
        
        # Combined category (6 queries)
        {
            "category": "combined",
            "query": "อาหารทะเลแบบไทยติดชายหาดพัทยา",
            "description": "ค้นหาอาหารทะเลสไตล์ไทยริมหาดพัทยา"
        },
        {
            "category": "combined",
            "query": "ร้านอาหารญี่ปุ่นราคาไม่แพงเชียงใหม่",
            "description": "ค้นหาร้านอาหารญี่ปุ่นราคาประหยัดในเชียงใหม่"
        },
        {
            "category": "combined",
            "query": "คาเฟ่บรรยากาศดีถ่ายรูปสวยกรุงเทพ",
            "description": "ค้นหาคาเฟ่บรรยากาศดีถ่ายรูปสวยในกรุงเทพ"
        },
        {
            "category": "combined",
            "query": "ร้านก๋วยเตี๋ยวอร่อยราคาถูก",
            "description": "ค้นหาร้านก๋วยเตี๋ยวอร่อยราคาไม่แพง"
        },
        {
            "category": "combined",
            "query": "ร้านอาหารไทยบรรยากาศดีวิวสวย",
            "description": "ค้นหาร้านอาหารไทยบรรยากาศดีวิวสวย"
        },
        {
            "category": "combined",
            "query": "อาหารอิตาลีราคาหรูกรุงเทพ",
            "description": "ค้นหาร้านอาหารอิตาลีระดับหรูในกรุงเทพ"
        },
    ]
    
    return queries


def evaluate_retrieval(
    retriever: WongnaiRetriever,
    test_queries: List[str],
    ground_truth_labels: Optional[List[List[str]]] = None,
) -> Dict:
    """Evaluate retrieval performance on test queries.
    
    Runs each test query through the retriever and computes various metrics
    including average retrieval scores, star ratings, and metadata coverage.
    
    Args:
        retriever: The retriever instance to evaluate.
        test_queries: List of query strings to test.
        ground_truth_labels: Optional list of ground truth labels for each query.
            Each item is a list of relevant document identifiers.
    
    Returns:
        Dictionary containing evaluation metrics:
            - avg_retrieval_score: Average cosine similarity score
            - avg_star_rating: Average star rating of retrieved results
            - metadata_coverage: Percentage of results with detected metadata
            - query_results: Per-query results for detailed analysis
    """
    if retriever.index is None or retriever.df is None:
        raise RuntimeError("Retriever index not loaded. Call load_index() first.")
    
    all_scores = []
    all_ratings = []
    all_metadata_coverage = []
    query_results = []
    
    for i, query in enumerate(test_queries):
        # Run search
        results = retriever.search(query, top_k=5)
        
        # Extract metrics
        scores = [r['score'] for r in results]
        ratings = [r['star_rating'] for r in results]
        
        # Calculate metadata coverage for each result
        coverage_per_result = []
        for r in results:
            has_metadata = bool(
                r.get('cuisine_type', []) or 
                r.get('food_type', []) or 
                r.get('location', [])
            )
            coverage_per_result.append(1.0 if has_metadata else 0.0)
        
        # Aggregate metrics for this query
        avg_score = np.mean(scores) if scores else 0.0
        avg_rating = np.mean(ratings) if ratings else 0.0
        avg_coverage = np.mean(coverage_per_result) if coverage_per_result else 0.0
        
        all_scores.extend(scores)
        all_ratings.extend(ratings)
        all_metadata_coverage.extend(coverage_per_result)
        
        query_results.append({
            'query': query,
            'num_results': len(results),
            'avg_score': avg_score,
            'avg_rating': avg_rating,
            'metadata_coverage': avg_coverage,
            'results': results,
        })
    
    # Calculate overall metrics
    metrics = {
        'avg_retrieval_score': float(np.mean(all_scores)) if all_scores else 0.0,
        'avg_star_rating': float(np.mean(all_ratings)) if all_ratings else 0.0,
        'metadata_coverage': float(np.mean(all_metadata_coverage) * 100) if all_metadata_coverage else 0.0,
        'num_queries': len(test_queries),
        'total_results': len(all_scores),
        'query_results': query_results,
    }
    
    return metrics


def compare_retrievers(
    baseline_retriever: WongnaiRetriever,
    finetuned_retriever: WongnaiRetriever,
    test_queries: List[str],
) -> pd.DataFrame:
    """Compare baseline vs finetuned retriever performance.
    
    Runs the same queries through both retrievers and creates a comparison
    DataFrame showing performance differences.
    
    Args:
        baseline_retriever: The baseline retriever instance.
        finetuned_retriever: The finetuned retriever instance.
        test_queries: List of query strings to test.
    
    Returns:
        DataFrame with comparison columns:
            - query: The test query
            - baseline_avg_score: Average retrieval score for baseline
            - finetuned_avg_score: Average retrieval score for finetuned
            - baseline_avg_rating: Average star rating for baseline
            - finetuned_avg_rating: Average star rating for finetuned
            - score_improvement: Percentage improvement in score
    """
    # Evaluate both retrievers
    baseline_metrics = evaluate_retrieval(baseline_retriever, test_queries)
    finetuned_metrics = evaluate_retrieval(finetuned_retriever, test_queries)
    
    # Build comparison data
    comparison_data = []
    for i, query in enumerate(test_queries):
        baseline_result = baseline_metrics['query_results'][i]
        finetuned_result = finetuned_metrics['query_results'][i]
        
        baseline_score = baseline_result['avg_score']
        finetuned_score = finetuned_result['avg_score']
        
        # Calculate improvement percentage
        if baseline_score > 0:
            score_improvement = ((finetuned_score - baseline_score) / baseline_score) * 100
        else:
            score_improvement = 0.0 if finetuned_score == 0 else 100.0
        
        comparison_data.append({
            'query': query,
            'baseline_avg_score': baseline_score,
            'finetuned_avg_score': finetuned_score,
            'baseline_avg_rating': baseline_result['avg_rating'],
            'finetuned_avg_rating': finetuned_result['avg_rating'],
            'score_improvement': score_improvement,
        })
    
    df = pd.DataFrame(comparison_data)
    return df


def run_demo_evaluation(
    baseline_retriever: WongnaiRetriever,
    finetuned_retriever: WongnaiRetriever,
    qa_generator: WongnaiQAGenerator,
) -> Dict:
    """Run a complete demo evaluation comparing baseline and finetuned models.
    
    For each demo query, runs both retrievers, generates answers, and collects
    all metrics for comparison.
    
    Args:
        baseline_retriever: The baseline retriever instance.
        finetuned_retriever: The finetuned retriever instance.
        qa_generator: The QA generator for generating answers.
    
    Returns:
        Dictionary containing:
            - queries: All demo queries with their categories
            - comparison_df: DataFrame comparing retrievers
            - per_query_results: Detailed results for each query
            - summary_metrics: Aggregated metrics by category
    """
    demo_queries = get_demo_queries()
    test_queries = [q['query'] for q in demo_queries]
    
    print("Running evaluation on demo queries...")
    print(f"Total queries: {len(test_queries)}\n")
    
    # Get comparison DataFrame
    comparison_df = compare_retrievers(baseline_retriever, finetuned_retriever, test_queries)
    
    # Detailed per-query results
    per_query_results = []
    
    print("Processing queries and generating answers...")
    print("-" * 80)
    
    for i, query_info in enumerate(demo_queries):
        query = query_info['query']
        category = query_info['category']
        
        # Search with both retrievers
        baseline_results = baseline_retriever.search(query, top_k=5)
        finetuned_results = finetuned_retriever.search(query, top_k=5)
        
        # Generate answers
        baseline_answer = qa_generator.generate_answer(query, baseline_results)
        finetuned_answer = qa_generator.generate_answer(query, finetuned_results)
        
        # Get comparison metrics
        row = comparison_df.iloc[i]
        
        per_query_results.append({
            'category': category,
            'query': query,
            'description': query_info['description'],
            'baseline_results': baseline_results,
            'finetuned_results': finetuned_results,
            'baseline_answer': baseline_answer,
            'finetuned_answer': finetuned_answer,
            'baseline_avg_score': row['baseline_avg_score'],
            'finetuned_avg_score': row['finetuned_avg_score'],
            'baseline_avg_rating': row['baseline_avg_rating'],
            'finetuned_avg_rating': row['finetuned_avg_rating'],
            'score_improvement': row['score_improvement'],
        })
        
        print(f"Query {i+1}/{len(demo_queries)}: {query}")
    
    print("-" * 80)
    
    # Aggregate metrics by category
    category_metrics = defaultdict(lambda: {
        'count': 0,
        'baseline_avg_score': [],
        'finetuned_avg_score': [],
        'baseline_avg_rating': [],
        'finetuned_avg_rating': [],
        'score_improvements': [],
    })
    
    for result in per_query_results:
        cat = result['category']
        category_metrics[cat]['count'] += 1
        category_metrics[cat]['baseline_avg_score'].append(result['baseline_avg_score'])
        category_metrics[cat]['finetuned_avg_score'].append(result['finetuned_avg_score'])
        category_metrics[cat]['baseline_avg_rating'].append(result['baseline_avg_rating'])
        category_metrics[cat]['finetuned_avg_rating'].append(result['finetuned_avg_rating'])
        category_metrics[cat]['score_improvements'].append(result['score_improvement'])
    
    # Calculate averages per category
    summary_metrics = {}
    for cat, metrics in category_metrics.items():
        summary_metrics[cat] = {
            'count': metrics['count'],
            'baseline_avg_score': float(np.mean(metrics['baseline_avg_score'])),
            'finetuned_avg_score': float(np.mean(metrics['finetuned_avg_score'])),
            'baseline_avg_rating': float(np.mean(metrics['baseline_avg_rating'])),
            'finetuned_avg_rating': float(np.mean(metrics['finetuned_avg_rating'])),
            'avg_improvement': float(np.mean(metrics['score_improvements'])),
        }
    
    # Overall metrics
    overall = {
        'baseline_avg_score': float(comparison_df['baseline_avg_score'].mean()),
        'finetuned_avg_score': float(comparison_df['finetuned_avg_score'].mean()),
        'baseline_avg_rating': float(comparison_df['baseline_avg_rating'].mean()),
        'finetuned_avg_rating': float(comparison_df['finetuned_avg_rating'].mean()),
        'avg_improvement': float(comparison_df['score_improvement'].mean()),
    }
    
    results = {
        'queries': demo_queries,
        'comparison_df': comparison_df,
        'per_query_results': per_query_results,
        'category_summary': summary_metrics,
        'overall_metrics': overall,
    }
    
    # Print formatted comparison table
    print_demo_comparison_table(results)
    
    return results


def print_demo_comparison_table(results: Dict) -> None:
    """Print a formatted comparison table for demo evaluation results.
    
    Args:
        results: The results dictionary from run_demo_evaluation().
    """
    print("\n" + "=" * 100)
    print("DEMO EVALUATION RESULTS - PER QUERY COMPARISON")
    print("=" * 100)
    
    df = results['comparison_df']
    
    # Header
    header = f"{'Query':<40} {'Baseline':>12} {'Finetuned':>12} {'Improvement':>12}"
    print(header)
    print("-" * 100)
    
    # Print each query
    for _, row in df.iterrows():
        query_display = row['query'][:38] + ".." if len(row['query']) > 40 else row['query']
        improvement_str = f"{row['score_improvement']:+.1f}%"
        line = f"{query_display:<40} {row['baseline_avg_score']:>12.4f} {row['finetuned_avg_score']:>12.4f} {improvement_str:>12}"
        print(line)
    
    print("-" * 100)
    
    # Overall averages
    overall = results['overall_metrics']
    avg_improvement = overall['avg_improvement']
    improvement_indicator = "++" if avg_improvement > 0 else "--" if avg_improvement < 0 else "=="
    
    print(f"{'OVERALL AVERAGE':<40} {overall['baseline_avg_score']:>12.4f} {overall['finetuned_avg_score']:>12.4f} {avg_improvement:+.1f}%")
    print("=" * 100)


def print_evaluation_report(results: Dict) -> None:
    """Print a nicely formatted evaluation report in Thai.
    
    Shows per-category metrics, overall comparison, and highlights improvements
    between baseline and finetuned models.
    
    Args:
        results: The results dictionary from run_demo_evaluation().
    """
    print("\n" + "=" * 80)
    print("รายงานผลการประเมินระบบ Wongnai QA".center(80))
    print("=" * 80)
    
    # Category translations
    cat_names = {
        'cuisine': 'ประเภทอาหาร (Cuisine)',
        'food_type': 'ประเภทอาหาร/เมนู (Food Type)',
        'atmosphere_price': 'บรรยากาศและราคา (Atmosphere/Price)',
        'location': 'สถานที่ (Location)',
        'combined': 'แบบผสม (Combined)',
    }
    
    # Per-category metrics
    print("\n[1] ผลการประเมินแยกตามประเภทคำถาม")
    print("-" * 80)
    
    category_summary = results['category_summary']
    
    for cat_key, cat_name in cat_names.items():
        if cat_key not in category_summary:
            continue
            
        metrics = category_summary[cat_key]
        print(f"\n{cat_name} ({metrics['count']} คำถาม)")
        print(f"  - คะแนนความเกี่ยวข้องเฉลี่ย (Baseline): {metrics['baseline_avg_score']:.4f}")
        print(f"  - คะแนนความเกี่ยวข้องเฉลี่ย (Finetuned): {metrics['finetuned_avg_score']:.4f}")
        print(f"  - คะแนนดาวเฉลี่ย (Baseline): {metrics['baseline_avg_rating']:.2f}")
        print(f"  - คะแนนดาวเฉลี่ย (Finetuned): {metrics['finetuned_avg_rating']:.2f}")
        
        improvement = metrics['avg_improvement']
        if improvement > 0:
            print(f"  - การปรับปรุง: +{improvement:.1f}% (ดีขึ้น)")
        elif improvement < 0:
            print(f"  - การปรับปรุง: {improvement:.1f}% (ลดลง)")
        else:
            print(f"  - การปรับปรุง: ไม่มีการเปลี่ยนแปลง")
    
    # Overall metrics
    print("\n" + "=" * 80)
    print("[2] ผลการประเมินโดยรวม")
    print("=" * 80)
    
    overall = results['overall_metrics']
    total_queries = len(results['queries'])
    
    print(f"\nจำนวนคำถามทั้งหมด: {total_queries} คำถาม")
    print(f"\nคะแนนความเกี่ยวข้องเฉลี่ย:")
    print(f"  - Baseline:  {overall['baseline_avg_score']:.4f}")
    print(f"  - Finetuned: {overall['finetuned_avg_score']:.4f}")
    
    print(f"\nคะแนนดาวเฉลี่ยของผลลัพธ์:")
    print(f"  - Baseline:  {overall['baseline_avg_rating']:.2f}")
    print(f"  - Finetuned: {overall['finetuned_avg_rating']:.2f}")
    
    # Improvement summary
    print("\n" + "=" * 80)
    print("[3] สรุปผลการปรับปรุง")
    print("=" * 80)
    
    avg_improvement = overall['avg_improvement']
    
    if avg_improvement > 5:
        status = "มีการปรับปรุงอย่างมีนัยสำคัญ"
        recommendation = "แนะนำให้ใช้โมเดลที่ Finetuned"
    elif avg_improvement > 0:
        status = "มีการปรับปรุงเล็กน้อย"
        recommendation = "สามารถใช้โมเดลที่ Finetuned ได้"
    elif avg_improvement > -5:
        status = "ไม่มีการเปลี่ยนแปลงที่สำคัญ"
        recommendation = "อาจใช้โมเดล Baseline ก็ได้"
    else:
        status = "ประสิทธิภาพลดลง"
        recommendation = "ควรใช้โมเดล Baseline"
    
    print(f"\nการปรับปรุงเฉลี่ย: {avg_improvement:+.1f}%")
    print(f"สถานะ: {status}")
    print(f"คำแนะนำ: {recommendation}")
    
    # Category performance breakdown
    print("\n" + "=" * 80)
    print("[4] ประสิทธิภาพแยกตามประเภท")
    print("=" * 80)
    
    # Sort categories by improvement
    sorted_cats = sorted(
        category_summary.items(),
        key=lambda x: x[1]['avg_improvement'],
        reverse=True
    )
    
    print("\nประเภทที่ปรับปรุงได้ดีที่สุด:")
    for i, (cat_key, metrics) in enumerate(sorted_cats[:3], 1):
        cat_name = cat_names.get(cat_key, cat_key)
        print(f"  {i}. {cat_name}: +{metrics['avg_improvement']:.1f}%")
    
    print("\n" + "=" * 80)
    print("จบรายงาน".center(80))
    print("=" * 80 + "\n")
