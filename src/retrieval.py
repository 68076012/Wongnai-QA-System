"""Retrieval module for Wongnai QA System.

This module provides the WongnaiRetriever class for semantic search using
sentence-transformers and FAISS. It supports building indices, loading indices,
and searching with optional filters.
"""

import os
import pickle
from typing import List, Dict, Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import (
    MODEL_CONFIG,
    FAISS_INDEX_PATH,
    FINETUNED_FAISS_INDEX_PATH,
    DEVICE,
)


class WongnaiRetriever:
    """Retriever class for semantic search over Wongnai restaurant reviews.
    
    Uses sentence-transformers for encoding text and FAISS for efficient
    similarity search. Supports both baseline and finetuned models.
    
    Attributes:
        model: The sentence-transformers model for encoding.
        index: The FAISS index for similarity search.
        df: The DataFrame containing review data.
        embeddings: The normalized embeddings of the documents.
        index_path: Path to save/load the FAISS index.
        df_path: Path to save/load the DataFrame pickle.
        is_e5_model: Whether the model is an E5 model (requires prefixes).
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        index_path: Optional[str] = None,
        is_finetuned: bool = False,
    ):
        """Initialize the retriever.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Defaults to MODEL_CONFIG['embedding_model'].
            index_path: Path to save/load the FAISS index.
                       Defaults to FINETUNED_FAISS_INDEX_PATH if is_finetuned,
                       otherwise FAISS_INDEX_PATH.
            is_finetuned: Whether to use the finetuned model path.
        """
        # Load model
        self.model_name = model_name or MODEL_CONFIG['embedding_model']
        self.model = SentenceTransformer(self.model_name, device=DEVICE)
        
        # Determine index path
        if index_path is None:
            if is_finetuned:
                self.index_path = FINETUNED_FAISS_INDEX_PATH
            else:
                self.index_path = FAISS_INDEX_PATH
        else:
            self.index_path = index_path
        
        # Set DataFrame pickle path
        self.df_path = self.index_path + "_df.pkl"
        
        # Check if using E5 model (requires prefixes)
        self.is_e5_model = 'e5' in self.model_name.lower()
        
        # Initialize attributes
        self.index = None
        self.df = None
        self.embeddings = None
    
    def build_index(self, processed_df: pd.DataFrame, batch_size: int = 64) -> None:
        """Build the FAISS index from processed reviews.
        
        Encodes all search_text entries, normalizes embeddings, and builds
        an inner product FAISS index.
        
        Args:
            processed_df: DataFrame with 'search_text' column and metadata.
            batch_size: Batch size for encoding.
        """
        self.df = processed_df.reset_index(drop=True)
        
        # Get texts to encode
        texts = self.df['search_text'].tolist()
        
        # Add passage prefix for E5 models
        if self.is_e5_model:
            texts = [f"passage: {text}" for text in texts]
        
        # Encode in batches with progress bar
        print(f"Encoding {len(texts)} documents with {self.model_name}...")
        self.embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        
        # Normalize embeddings (L2) for cosine similarity via inner product
        faiss.normalize_L2(self.embeddings)
        
        # Build FAISS index (inner product = cosine sim after L2 norm)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, self.index_path)
        
        # Save DataFrame
        self.df.to_pickle(self.df_path)
        
        print(f"Saved FAISS index to {self.index_path}")
        print(f"Saved DataFrame to {self.df_path}")
    
    def load_index(self) -> None:
        """Load the FAISS index and DataFrame from disk."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}")
        if not os.path.exists(self.df_path):
            raise FileNotFoundError(f"DataFrame pickle not found at {self.df_path}")
        
        self.index = faiss.read_index(self.index_path)
        self.df = pd.read_pickle(self.df_path)
        
        print(f"Loaded FAISS index from {self.index_path}")
        print(f"Loaded DataFrame from {self.df_path}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar reviews.
        
        Args:
            query: The search query string.
            top_k: Number of results to return.
            
        Returns:
            List of dictionaries containing review information and similarity scores.
        """
        if self.index is None or self.df is None:
            raise RuntimeError("Index not loaded. Call load_index() or build_index() first.")
        
        # Prepare query with prefix for E5 models
        query_text = query
        if self.is_e5_model:
            query_text = f"query: {query}"
        
        # Encode query
        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for not enough results
                continue
                
            row = self.df.iloc[idx]
            result = {
                'review_text': row.get('review_text', ''),
                'star_rating': int(row.get('star_rating', 0)),
                'score': float(score),
                'cuisine_type': row.get('cuisine_type', []),
                'food_type': row.get('food_type', []),
                'atmosphere': row.get('atmosphere', []),
                'price_level': row.get('price_level', []),
                'location': row.get('location', []),
                'mentioned_foods': row.get('mentioned_foods', []),
            }
            results.append(result)
        
        return results
    
    def search_with_filters(
        self,
        query: str,
        top_k: int = 5,
        min_rating: Optional[int] = None,
        cuisine_filter: Optional[str] = None,
        food_type_filter: Optional[str] = None,
        location_filter: Optional[str] = None,
    ) -> List[Dict]:
        """Search with post-filtering on metadata.
        
        First retrieves top_k * 3 results, then filters based on criteria,
        and returns top_k filtered results.
        
        Args:
            query: The search query string.
            top_k: Number of results to return after filtering.
            min_rating: Minimum star rating (1-5).
            cuisine_filter: Filter by cuisine type.
            food_type_filter: Filter by food type.
            location_filter: Filter by location.
            
        Returns:
            Filtered list of dictionaries containing review information.
        """
        # Get more results to allow for filtering
        initial_k = top_k * 3
        results = self.search(query, top_k=initial_k)
        
        # Apply filters
        filtered_results = []
        for result in results:
            # Filter by minimum rating
            if min_rating is not None and result['star_rating'] < min_rating:
                continue
            
            # Filter by cuisine type
            if cuisine_filter is not None:
                cuisines = [c.lower() for c in result['cuisine_type']]
                if cuisine_filter.lower() not in cuisines:
                    continue
            
            # Filter by food type
            if food_type_filter is not None:
                food_types = [f.lower() for f in result['food_type']]
                if food_type_filter.lower() not in food_types:
                    continue
            
            # Filter by location
            if location_filter is not None:
                locations = [l.lower() for l in result['location']]
                if location_filter.lower() not in locations:
                    continue
            
            filtered_results.append(result)
        
        # Return top_k filtered results
        return filtered_results[:top_k]


def build_baseline_index(processed_df: pd.DataFrame) -> WongnaiRetriever:
    """Build a baseline index with the default embedding model.
    
    Args:
        processed_df: DataFrame with 'search_text' column.
        
    Returns:
        WongnaiRetriever with built index.
    """
    retriever = WongnaiRetriever()
    retriever.build_index(processed_df)
    return retriever


def build_finetuned_index(
    processed_df: pd.DataFrame,
    finetuned_model_path: str,
) -> WongnaiRetriever:
    """Build an index with a finetuned model.
    
    Args:
        processed_df: DataFrame with 'search_text' column.
        finetuned_model_path: Path to the finetuned sentence-transformers model.
        
    Returns:
        WongnaiRetriever with built index at FINETUNED_FAISS_INDEX_PATH.
    """
    retriever = WongnaiRetriever(
        model_name=finetuned_model_path,
        index_path=FINETUNED_FAISS_INDEX_PATH,
        is_finetuned=True,
    )
    retriever.build_index(processed_df)
    return retriever
