"""Main pipeline script for Wongnai QA System.

This script orchestrates the entire pipeline with command-line arguments:
- preprocess: Run data preprocessing
- build-index: Build baseline FAISS index
- finetune: Fine-tune embedding model and build finetuned index
- evaluate: Compare baseline vs finetuned retrievers
- demo: Run quick demo with sample queries
- app: Launch Gradio web UI
- all: Run full pipeline

Usage:
    python run_pipeline.py preprocess
    python run_pipeline.py build-index
    python run_pipeline.py finetune
    python run_pipeline.py evaluate
    python run_pipeline.py demo
    python run_pipeline.py app [--share] [--llm]
    python run_pipeline.py all [--skip-finetune]
"""

import argparse
import os
import sys
import time
from typing import Optional

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    PROCESSED_DATA_PATH,
    FAISS_INDEX_PATH,
    FINETUNED_FAISS_INDEX_PATH,
    MODEL_CONFIG,
)


def print_header(title: str, width: int = 70) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f" {title}".center(width))
    print("=" * width + "\n")


def print_subheader(title: str, width: int = 70) -> None:
    """Print a formatted subsection header."""
    print("\n" + "-" * width)
    print(f" {title}")
    print("-" * width)


def print_elapsed_time(start_time: float, task_name: str = "Task") -> None:
    """Print elapsed time for a task."""
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\n[{task_name}] Elapsed time: {minutes}m {seconds}s")


def check_processed_data() -> Optional[str]:
    """Check if processed data exists, return path if yes, None otherwise."""
    processed_file = os.path.join(PROCESSED_DATA_PATH, "processed_reviews.pkl")
    if os.path.exists(processed_file):
        return processed_file
    return None


def check_baseline_index() -> bool:
    """Check if baseline FAISS index exists."""
    return os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_INDEX_PATH + "_df.pkl")


def check_finetuned_model() -> Optional[str]:
    """Check if finetuned model exists, return path if yes, None otherwise."""
    finetuned_path = os.path.join("models", "finetuned_embedding")
    if os.path.exists(finetuned_path) and os.path.isdir(finetuned_path):
        # Check for required model files
        config_file = os.path.join(finetuned_path, "config.json")
        if os.path.exists(config_file):
            return finetuned_path
    return None


def check_finetuned_index() -> bool:
    """Check if finetuned FAISS index exists."""
    return os.path.exists(FINETUNED_FAISS_INDEX_PATH) and os.path.exists(FINETUNED_FAISS_INDEX_PATH + "_df.pkl")


def cmd_preprocess(args: argparse.Namespace) -> int:
    """Run data preprocessing command."""
    start_time = time.time()
    print_header("PREPROCESSING DATA")
    
    try:
        from src.data_preprocessing import main as preprocessing_main
        preprocessing_main()
        print_elapsed_time(start_time, "Preprocessing")
        print("\n✓ Preprocessing completed successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Error during preprocessing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_build_index(args: argparse.Namespace) -> int:
    """Build baseline FAISS index command."""
    start_time = time.time()
    print_header("BUILDING BASELINE FAISS INDEX")
    
    # Check for processed data
    processed_file = check_processed_data()
    if not processed_file:
        print("✗ Error: Processed data not found.")
        print(f"  Expected at: {os.path.join(PROCESSED_DATA_PATH, 'processed_reviews.pkl')}")
        print("\n  Please run preprocessing first:")
        print("    python run_pipeline.py preprocess")
        return 1
    
    print(f"Loading processed data from: {processed_file}")
    
    try:
        import pandas as pd
        from src.retrieval import build_baseline_index
        
        processed_df = pd.read_pickle(processed_file)
        print(f"Loaded {len(processed_df)} processed reviews\n")
        
        # Build index
        retriever = build_baseline_index(processed_df)
        
        # Print stats
        print_subheader("Index Statistics")
        print(f"Index path: {FAISS_INDEX_PATH}")
        print(f"Number of vectors: {retriever.index.ntotal}")
        print(f"Vector dimension: {retriever.embeddings.shape[1]}")
        print(f"Model used: {retriever.model_name}")
        
        print_elapsed_time(start_time, "Index building")
        print("\n✓ Baseline index built successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error building index: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_finetune(args: argparse.Namespace) -> int:
    """Fine-tune embedding model command."""
    start_time = time.time()
    print_header("FINE-TUNING EMBEDDING MODEL")
    
    # Check for processed data
    processed_file = check_processed_data()
    if not processed_file:
        print("✗ Error: Processed data not found.")
        print(f"  Expected at: {os.path.join(PROCESSED_DATA_PATH, 'processed_reviews.pkl')}")
        print("\n  Please run preprocessing first:")
        print("    python run_pipeline.py preprocess")
        return 1
    
    try:
        import pandas as pd
        from src.data_preprocessing import load_food_dictionary, load_queries, FOOD_DICT_FILE, QUERY_JUDGES_FILE, QUERY_ALGO_FILE
        from src.finetune_embedding import generate_training_pairs, finetune_model
        from src.retrieval import build_finetuned_index
        
        # Load processed data
        print("[1/5] Loading processed data...")
        processed_df = pd.read_pickle(processed_file)
        print(f"  Loaded {len(processed_df)} processed reviews")
        
        # Load food dictionary and queries
        print("\n[2/5] Loading food dictionary and queries...")
        food_dict = load_food_dictionary(FOOD_DICT_FILE)
        queries_judges = load_queries(QUERY_JUDGES_FILE)
        queries_algo = load_queries(QUERY_ALGO_FILE)
        print(f"  Loaded {len(food_dict)} food items")
        print(f"  Loaded {len(queries_judges)} judge-labeled queries")
        print(f"  Loaded {len(queries_algo)} algorithm-labeled queries")
        
        # Generate training pairs
        print("\n[3/5] Generating training pairs...")
        num_pairs = args.num_pairs if hasattr(args, 'num_pairs') else 10000
        training_pairs = generate_training_pairs(
            processed_df=processed_df,
            food_dict=food_dict,
            queries_judges=queries_judges,
            queries_algo=queries_algo,
            num_pairs=num_pairs
        )
        
        # Fine-tune model
        print("\n[4/5] Fine-tuning model...")
        epochs = args.epochs if hasattr(args, 'epochs') else 3
        batch_size = args.batch_size if hasattr(args, 'batch_size') else 16
        
        finetuned_path = finetune_model(
            training_pairs=training_pairs,
            base_model_name=None,  # Use default from config
            output_path='models/finetuned_embedding',
            epochs=epochs,
            batch_size=batch_size,
            warmup_steps=100
        )
        
        # Build finetuned index
        print("\n[5/5] Building finetuned FAISS index...")
        retriever = build_finetuned_index(processed_df, finetuned_path)
        
        # Print summary
        print_subheader("Training Summary")
        print(f"Base model: {MODEL_CONFIG['embedding_model']}")
        print(f"Training pairs: {len(training_pairs)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Model saved to: {finetuned_path}")
        print(f"Index saved to: {FINETUNED_FAISS_INDEX_PATH}")
        print(f"Number of vectors in index: {retriever.index.ntotal}")
        
        print_elapsed_time(start_time, "Fine-tuning")
        print("\n✓ Fine-tuning completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during fine-tuning: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate retrievers command."""
    start_time = time.time()
    print_header("EVALUATING RETRIEVERS")
    
    # Check for baseline index
    if not check_baseline_index():
        print("✗ Error: Baseline index not found.")
        print(f"  Expected at: {FAISS_INDEX_PATH}")
        print("\n  Please build the baseline index first:")
        print("    python run_pipeline.py build-index")
        return 1
    
    try:
        from src.retrieval import WongnaiRetriever
        from src.qa_generator import WongnaiQAGenerator
        from src.evaluation import run_demo_evaluation, print_evaluation_report
        
        # Load baseline retriever
        print("[1/3] Loading baseline retriever...")
        baseline_retriever = WongnaiRetriever()
        baseline_retriever.load_index()
        
        # Load finetuned retriever if available
        finetuned_retriever = None
        if check_finetuned_index():
            print("\n[2/3] Loading finetuned retriever...")
            finetuned_model_path = check_finetuned_model()
            model_name = finetuned_model_path if finetuned_model_path else MODEL_CONFIG['embedding_model']
            finetuned_retriever = WongnaiRetriever(
                model_name=model_name,
                index_path=FINETUNED_FAISS_INDEX_PATH,
                is_finetuned=True,
            )
            finetuned_retriever.load_index()
        else:
            print("\n[2/3] Finetuned index not found, using baseline for comparison...")
            finetuned_retriever = baseline_retriever
        
        # Initialize QA generator
        print("\n[3/3] Initializing QA generator...")
        use_llm = args.llm if hasattr(args, 'llm') else False
        qa_generator = WongnaiQAGenerator(use_llm=use_llm)
        
        # Run evaluation
        print("\n" + "=" * 70)
        print("Running evaluation on demo queries...")
        print("=" * 70)
        
        results = run_demo_evaluation(
            baseline_retriever=baseline_retriever,
            finetuned_retriever=finetuned_retriever,
            qa_generator=qa_generator
        )
        
        # Print detailed report
        print_evaluation_report(results)
        
        print_elapsed_time(start_time, "Evaluation")
        print("\n✓ Evaluation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_demo(args: argparse.Namespace) -> int:
    """Run quick demo command."""
    start_time = time.time()
    print_header("DEMO MODE - SAMPLE QUERIES")
    
    # Check for baseline index
    if not check_baseline_index():
        print("✗ Error: Baseline index not found.")
        print(f"  Expected at: {FAISS_INDEX_PATH}")
        print("\n  Please build the index first:")
        print("    python run_pipeline.py build-index")
        return 1
    
    # Sample demo queries
    demo_queries = [
        "อาหารญี่ปุ่นอร่อย",
        "ร้านอาหารทะเลสดๆ",
        "คาเฟ่บรรยากาศดีกรุงเทพ",
        "ร้านอาหารหรูหรา",
    ]
    
    try:
        from src.retrieval import WongnaiRetriever
        from src.qa_generator import WongnaiQAGenerator
        
        # Load retrievers
        print("Loading retrievers...")
        baseline_retriever = WongnaiRetriever()
        baseline_retriever.load_index()
        
        has_finetuned = check_finetuned_index()
        finetuned_retriever = None
        if has_finetuned:
            finetuned_model_path = check_finetuned_model()
            model_name = finetuned_model_path if finetuned_model_path else MODEL_CONFIG['embedding_model']
            finetuned_retriever = WongnaiRetriever(
                model_name=model_name,
                index_path=FINETUNED_FAISS_INDEX_PATH,
                is_finetuned=True,
            )
            finetuned_retriever.load_index()
        
        qa_generator = WongnaiQAGenerator(use_llm=False)
        
        # Run demo queries
        print("\n" + "=" * 70)
        print("Running demo queries...")
        print("=" * 70)
        
        for i, query in enumerate(demo_queries, 1):
            print_subheader(f"Query {i}/{len(demo_queries)}: {query}")
            
            # Baseline results
            baseline_results = baseline_retriever.search(query, top_k=3)
            print("\nBaseline Results:")
            for j, result in enumerate(baseline_results, 1):
                score = result.get('score', 0)
                rating = result.get('star_rating', 0)
                review = result.get('review_text', '')[:100]
                print(f"  {j}. Score: {score:.4f} | Rating: {rating}/5")
                print(f"     {review}...")
            
            # Finetuned results if available
            if finetuned_retriever:
                finetuned_results = finetuned_retriever.search(query, top_k=3)
                print("\nFinetuned Results:")
                for j, result in enumerate(finetuned_results, 1):
                    score = result.get('score', 0)
                    rating = result.get('star_rating', 0)
                    review = result.get('review_text', '')[:100]
                    print(f"  {j}. Score: {score:.4f} | Rating: {rating}/5")
                    print(f"     {review}...")
            
            # Generate answer
            answer = qa_generator.generate_answer(query, baseline_results)
            print(f"\nGenerated Answer (Baseline):")
            print(f"  {answer[:300]}...")
            print()
        
        print_elapsed_time(start_time, "Demo")
        print("\n✓ Demo completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_app(args: argparse.Namespace) -> int:
    """Launch Gradio web UI command."""
    print_header("LAUNCHING WEB UI")
    
    # Check for baseline index
    if not check_baseline_index():
        print("✗ Error: Baseline index not found.")
        print(f"  Expected at: {FAISS_INDEX_PATH}")
        print("\n  Please build the index first:")
        print("    python run_pipeline.py build-index")
        return 1
    
    try:
        from src.app import launch_app
        
        share = args.share if hasattr(args, 'share') else False
        use_llm = args.llm if hasattr(args, 'llm') else False
        
        print(f"Launching Gradio app...")
        print(f"  - Share: {'Yes' if share else 'No'}")
        print(f"  - LLM mode: {'Yes' if use_llm else 'No'}")
        print(f"\nAccess the app at: http://localhost:7860")
        if share:
            print("A public link will be generated...")
        print()
        
        launch_app(share=share)
        return 0
        
    except Exception as e:
        print(f"\n✗ Error launching app: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_all(args: argparse.Namespace) -> int:
    """Run full pipeline command."""
    start_time = time.time()
    print_header("FULL PIPELINE EXECUTION")
    print("Steps: preprocess -> build-index -> finetune -> evaluate\n")
    
    skip_finetune = args.skip_finetune if hasattr(args, 'skip_finetune') else False
    
    # Step 1: Preprocess
    print("\n" + "=" * 70)
    print("STEP 1/4: PREPROCESSING")
    print("=" * 70)
    
    # Check if already preprocessed
    if check_processed_data():
        print("Processed data already exists. Skipping preprocessing...")
        print(f"  Found at: {check_processed_data()}")
    else:
        ret = cmd_preprocess(args)
        if ret != 0:
            print("\n✗ Pipeline failed at preprocessing step.")
            return ret
    
    # Step 2: Build baseline index
    print("\n" + "=" * 70)
    print("STEP 2/4: BUILDING BASELINE INDEX")
    print("=" * 70)
    
    if check_baseline_index():
        print("Baseline index already exists. Skipping...")
        print(f"  Found at: {FAISS_INDEX_PATH}")
    else:
        ret = cmd_build_index(args)
        if ret != 0:
            print("\n✗ Pipeline failed at index building step.")
            return ret
    
    # Step 3: Fine-tune
    if not skip_finetune:
        print("\n" + "=" * 70)
        print("STEP 3/4: FINE-TUNING MODEL")
        print("=" * 70)
        
        if check_finetuned_index():
            print("Finetuned index already exists. Skipping...")
            print(f"  Found at: {FINETUNED_FAISS_INDEX_PATH}")
        else:
            ret = cmd_finetune(args)
            if ret != 0:
                print("\n✗ Pipeline failed at fine-tuning step.")
                return ret
    else:
        print("\n" + "=" * 70)
        print("STEP 3/4: FINE-TUNING (SKIPPED)")
        print("=" * 70)
        print("  --skip-finetune flag is set. Skipping fine-tuning step.")
    
    # Step 4: Evaluate
    print("\n" + "=" * 70)
    print("STEP 4/4: EVALUATION")
    print("=" * 70)
    ret = cmd_evaluate(args)
    if ret != 0:
        print("\n✗ Pipeline failed at evaluation step.")
        return ret
    
    print_elapsed_time(start_time, "Full pipeline")
    print("\n" + "=" * 70)
    print("✓ FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Run demo: python run_pipeline.py demo")
    print("  - Launch web UI: python run_pipeline.py app")
    
    return 0


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Wongnai QA System - Main Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py preprocess
  python run_pipeline.py build-index
  python run_pipeline.py finetune
  python run_pipeline.py evaluate
  python run_pipeline.py demo
  python run_pipeline.py app --share
  python run_pipeline.py all --skip-finetune
        """
    )
    
    # Global flags
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output with detailed error messages"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # preprocess command
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Run data preprocessing"
    )
    
    # build-index command
    build_index_parser = subparsers.add_parser(
        "build-index",
        help="Build baseline FAISS index"
    )
    
    # finetune command
    finetune_parser = subparsers.add_parser(
        "finetune",
        help="Fine-tune embedding model and build finetuned index"
    )
    finetune_parser.add_argument(
        "--num-pairs",
        type=int,
        default=10000,
        help="Number of training pairs to generate (default: 10000)"
    )
    finetune_parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    finetune_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)"
    )
    
    # evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Compare baseline vs finetuned retrievers"
    )
    evaluate_parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM-based answer generation (requires GPU)"
    )
    
    # demo command
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run quick demo with sample queries"
    )
    
    # app command
    app_parser = subparsers.add_parser(
        "app",
        help="Launch Gradio web UI"
    )
    app_parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    app_parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM-based answer generation (requires GPU)"
    )
    
    # all command
    all_parser = subparsers.add_parser(
        "all",
        help="Run full pipeline: preprocess -> build-index -> finetune -> evaluate"
    )
    all_parser.add_argument(
        "--skip-finetune",
        action="store_true",
        help="Skip the fine-tuning step"
    )
    all_parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM-based answer generation in evaluation (requires GPU)"
    )
    all_parser.add_argument(
        "--num-pairs",
        type=int,
        default=10000,
        help="Number of training pairs for fine-tuning (default: 10000)"
    )
    all_parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    all_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if command is provided
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate command handler
    commands = {
        "preprocess": cmd_preprocess,
        "build-index": cmd_build_index,
        "finetune": cmd_finetune,
        "evaluate": cmd_evaluate,
        "demo": cmd_demo,
        "app": cmd_app,
        "all": cmd_all,
    }
    
    if args.command in commands:
        return commands[args.command](args)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
