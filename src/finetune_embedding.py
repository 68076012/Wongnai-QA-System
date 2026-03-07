"""Fine-tuning module for the embedding model in Wongnai QA System.

This module provides functionality to fine-tune sentence-transformers models
using contrastive learning with MultipleNegativesRankingLoss. It generates
training pairs from queries and reviews for domain adaptation.
"""

import os
import random
from typing import List, Set, Dict

import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import losses, evaluation
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    MODEL_CONFIG,
    PROCESSED_DATA_PATH,
    FINETUNED_FAISS_INDEX_PATH,
    DEVICE,
)
from src.data_preprocessing import load_food_dictionary, load_queries, FOOD_DICT_FILE, QUERY_JUDGES_FILE, QUERY_ALGO_FILE


def generate_training_pairs(
    processed_df: pd.DataFrame,
    food_dict: List[str],
    queries_judges: List[str],
    queries_algo: List[str],
    num_pairs: int = 10000
) -> List[InputExample]:
    """Generate training pairs for contrastive learning.
    
    Uses three strategies to create positive pairs:
    - Strategy A: Query-Review pairs using labeled queries
    - Strategy B: Similar Review pairs grouped by metadata
    - Strategy C: Rating-based pairs with cross-food-type negatives
    
    Args:
        processed_df: DataFrame with processed reviews and metadata.
        food_dict: List of food names for entity matching.
        queries_judges: List of judge-labeled queries.
        queries_algo: List of algorithm-labeled queries.
        num_pairs: Total number of training pairs to generate.
        
    Returns:
        List of InputExample objects for training.
    """
    training_pairs = []
    food_dict_set = set(food_dict)
    
    # Combine all queries
    all_queries = list(set(queries_judges + queries_algo))
    print(f"Total unique queries: {len(all_queries)}")
    
    # Convert lists in dataframe to sets for faster lookup
    print("Preprocessing metadata for pair generation...")
    
    # Create lookup structures for efficient sampling
    cuisine_to_indices: Dict[str, List[int]] = {}
    foodtype_to_indices: Dict[str, List[int]] = {}
    location_to_indices: Dict[str, List[int]] = {}
    high_rating_indices: List[int] = []
    
    for idx, row in tqdm(processed_df.iterrows(), total=len(processed_df), desc="Building lookup"):
        # High rating reviews (4-5 stars)
        if row.get('star_rating', 0) >= 4:
            high_rating_indices.append(idx)
        
        # Group by cuisine
        for cuisine in row.get('cuisine_type', []):
            if cuisine not in cuisine_to_indices:
                cuisine_to_indices[cuisine] = []
            cuisine_to_indices[cuisine].append(idx)
        
        # Group by food type
        for food_type in row.get('food_type', []):
            if food_type not in foodtype_to_indices:
                foodtype_to_indices[food_type] = []
            foodtype_to_indices[food_type].append(idx)
        
        # Group by location
        for loc in row.get('location', []):
            if loc not in location_to_indices:
                location_to_indices[loc] = []
            location_to_indices[loc].append(idx)
    
    print(f"High rating reviews: {len(high_rating_indices)}")
    print(f"Cuisine groups: {len(cuisine_to_indices)}")
    print(f"Food type groups: {len(foodtype_to_indices)}")
    print(f"Location groups: {len(location_to_indices)}")
    
    # Strategy A: Query-Review pairs (30% of total)
    num_query_pairs = int(num_pairs * 0.3)
    print(f"\nGenerating {num_query_pairs} query-review pairs...")
    
    query_pairs_count = 0
    max_attempts = num_query_pairs * 10
    attempts = 0
    
    # Pre-index reviews by text for faster searching
    review_texts = processed_df['review_text'].tolist()
    search_texts = processed_df['search_text'].tolist()
    
    while query_pairs_count < num_query_pairs and attempts < max_attempts:
        attempts += 1
        query = random.choice(all_queries)
        query_terms = query.lower().split()
        
        # Find reviews containing query terms
        matching_indices = []
        for idx, review_text in enumerate(review_texts):
            review_lower = review_text.lower()
            # Check if any significant query term appears in review
            if any(len(term) >= 3 and term in review_lower for term in query_terms):
                matching_indices.append(idx)
        
        if len(matching_indices) >= 1:
            # Create positive pair (query, matching review)
            pos_idx = random.choice(matching_indices)
            training_pairs.append(InputExample(
                texts=[query, search_texts[pos_idx]]
            ))
            query_pairs_count += 1
    
    print(f"Generated {query_pairs_count} query-review pairs")
    
    # Strategy B: Similar Review pairs by metadata (40% of total)
    num_similar_pairs = int(num_pairs * 0.4)
    print(f"\nGenerating {num_similar_pairs} similar review pairs...")
    
    similar_pairs_count = 0
    max_attempts = num_similar_pairs * 5
    attempts = 0
    
    # Combine all grouping structures
    all_groups = []
    for indices in cuisine_to_indices.values():
        if len(indices) >= 2:
            all_groups.append(('cuisine', indices))
    for indices in foodtype_to_indices.values():
        if len(indices) >= 2:
            all_groups.append(('food_type', indices))
    for indices in location_to_indices.values():
        if len(indices) >= 2:
            all_groups.append(('location', indices))
    
    print(f"Available metadata groups: {len(all_groups)}")
    
    while similar_pairs_count < num_similar_pairs and attempts < max_attempts:
        attempts += 1
        
        # Pick a random group that has at least 2 reviews
        valid_groups = [g for g in all_groups if len(g[1]) >= 2]
        if not valid_groups:
            break
            
        group_type, indices = random.choice(valid_groups)
        
        # Sample two different reviews from the same group
        idx1, idx2 = random.sample(indices, 2)
        
        training_pairs.append(InputExample(
            texts=[search_texts[idx1], search_texts[idx2]]
        ))
        similar_pairs_count += 1
    
    print(f"Generated {similar_pairs_count} similar review pairs")
    
    # Strategy C: Rating-based pairs (30% of total)
    num_rating_pairs = int(num_pairs * 0.3)
    print(f"\nGenerating {num_rating_pairs} rating-based pairs...")
    
    rating_pairs_count = 0
    max_attempts = num_rating_pairs * 5
    attempts = 0
    
    # Get food type for each review for cross-food-type negative sampling
    review_food_types = processed_df['food_type'].tolist()
    
    while rating_pairs_count < num_rating_pairs and attempts < max_attempts:
        attempts += 1
        
        if len(high_rating_indices) < 2:
            break
        
        # Sample two high rating reviews
        idx1 = random.choice(high_rating_indices)
        idx2 = random.choice(high_rating_indices)
        
        if idx1 == idx2:
            continue
        
        # Check if they share the same food type (positive pair)
        foods1 = set(review_food_types[idx1]) if isinstance(review_food_types[idx1], list) else set()
        foods2 = set(review_food_types[idx2]) if isinstance(review_food_types[idx2], list) else set()
        
        if foods1 and foods2 and foods1.intersection(foods2):
            # Positive pair - same food type, high rating
            training_pairs.append(InputExample(
                texts=[search_texts[idx1], search_texts[idx2]]
            ))
            rating_pairs_count += 1
    
    print(f"Generated {rating_pairs_count} rating-based pairs")
    
    # If we didn't generate enough pairs, fill with random similar pairs
    total_generated = len(training_pairs)
    if total_generated < num_pairs:
        print(f"\nFilling remaining {num_pairs - total_generated} pairs with random sampling...")
        while len(training_pairs) < num_pairs:
            idx1, idx2 = random.sample(range(len(processed_df)), 2)
            training_pairs.append(InputExample(
                texts=[search_texts[idx1], search_texts[idx2]]
            ))
    
    # Shuffle the pairs
    random.shuffle(training_pairs)
    
    # Trim to exact number requested
    training_pairs = training_pairs[:num_pairs]
    
    print(f"\nTotal training pairs: {len(training_pairs)}")
    return training_pairs


def finetune_model(
    training_pairs: List[InputExample],
    base_model_name: str = None,
    output_path: str = 'models/finetuned_embedding',
    epochs: int = 3,
    batch_size: int = 16,
    warmup_steps: int = 100
) -> str:
    """Fine-tune a sentence-transformers model with contrastive learning.
    
    Uses MultipleNegativesRankingLoss which is optimal for retrieval tasks.
    The model learns to distinguish relevant from irrelevant pairs within
    each training batch.
    
    Args:
        training_pairs: List of InputExample objects for training.
        base_model_name: Name of the base model to fine-tune.
                        Defaults to MODEL_CONFIG['embedding_model'].
        output_path: Path to save the fine-tuned model.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        warmup_steps: Number of warmup steps for learning rate scheduling.
        
    Returns:
        Path to the saved fine-tuned model.
    """
    # Determine base model
    if base_model_name is None:
        base_model_name = MODEL_CONFIG['embedding_model']
    
    print(f"\nLoading base model: {base_model_name}")
    model = SentenceTransformer(base_model_name, device=DEVICE)
    
    # Create DataLoader
    train_dataloader = DataLoader(
        training_pairs,
        shuffle=True,
        batch_size=batch_size
    )
    
    # Use MultipleNegativesRankingLoss - best for retrieval fine-tuning
    # This loss treats all other examples in the batch as negatives
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Calculate total training steps
    total_steps = len(train_dataloader) * epochs
    print(f"Training steps per epoch: {len(train_dataloader)}")
    print(f"Total training steps: {total_steps}")
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    print(f"\nStarting fine-tuning for {epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Output path: {output_path}")
    
    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
    )
    
    print(f"\nFine-tuning completed!")
    print(f"Model saved to: {output_path}")
    
    return output_path


def main():
    """Main function to run the fine-tuning pipeline."""
    print("=" * 60)
    print("Wongnai QA System - Embedding Model Fine-tuning")
    print("=" * 60)
    
    # Load processed data
    print("\n[1/4] Loading processed data...")
    processed_data_path = os.path.join(PROCESSED_DATA_PATH, 'processed_reviews.pkl')
    
    if not os.path.exists(processed_data_path):
        print(f"Error: Processed data not found at {processed_data_path}")
        print("Please run data_preprocessing.py first.")
        return
    
    processed_df = pd.read_pickle(processed_data_path)
    print(f"Loaded {len(processed_df)} processed reviews")
    
    # Load food dictionary and queries
    print("\n[2/4] Loading food dictionary and queries...")
    food_dict = load_food_dictionary(FOOD_DICT_FILE)
    queries_judges = load_queries(QUERY_JUDGES_FILE)
    queries_algo = load_queries(QUERY_ALGO_FILE)
    print(f"Loaded {len(food_dict)} food items")
    print(f"Loaded {len(queries_judges)} judge-labeled queries")
    print(f"Loaded {len(queries_algo)} algorithm-labeled queries")
    
    # Generate training pairs
    print("\n[3/4] Generating training pairs...")
    training_pairs = generate_training_pairs(
        processed_df=processed_df,
        food_dict=food_dict,
        queries_judges=queries_judges,
        queries_algo=queries_algo,
        num_pairs=10000
    )
    
    # Fine-tune model
    print("\n[4/4] Fine-tuning model...")
    output_path = finetune_model(
        training_pairs=training_pairs,
        base_model_name=None,  # Use default from config
        output_path='models/finetuned_embedding',
        epochs=3,
        batch_size=16,
        warmup_steps=100
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Base model: {MODEL_CONFIG['embedding_model']}")
    print(f"Training pairs: {len(training_pairs)}")
    print(f"Epochs: 3")
    print(f"Batch size: 16")
    print(f"Model saved to: {output_path}")
    print("\nNext steps:")
    print("1. Build a new FAISS index with the fine-tuned model")
    print("2. Evaluate retrieval performance")
    print("=" * 60)


if __name__ == "__main__":
    main()
