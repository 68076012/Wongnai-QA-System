"""Configuration module for Wongnai QA System.

This module contains all configuration settings and file paths
used throughout the Wongnai QA System.
"""

import os
import torch

# Data directories and file paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Dataset")
REVIEW_TRAIN_FILE = os.path.join(DATA_DIR, "review_dataset", "w_review_train.csv")
REVIEW_TEST_FILE = os.path.join(DATA_DIR, "review_dataset", "test_file.csv")
FOOD_DICT_FILE = os.path.join(DATA_DIR, "food_dictionary.txt")
QUERY_JUDGES_FILE = os.path.join(DATA_DIR, "labeled_queries_by_judges.txt")
QUERY_ALGO_FILE = os.path.join(DATA_DIR, "labeled_queries_by_algo.txt")

# Model configurations
MODEL_CONFIG = {
    "embedding_model": "intfloat/multilingual-e5-base",
    "qa_model": "scb10x/llama3.1-typhoon2-8b-instruct",
}

# FAISS and processed data paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FAISS_INDEX_PATH = os.path.join(MODELS_DIR, "faiss_index")
FINETUNED_FAISS_INDEX_PATH = os.path.join(MODELS_DIR, "finetuned_faiss_index")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")

# Retrieval settings
TOP_K = 5

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
