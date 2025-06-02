"""
Customer Churn Prediction Package

This package contains modules for predicting customer churn using machine learning.

Modules:
- data_preprocessing: Data loading, cleaning, and feature engineering
- model_training: Model creation and training
- model_evaluation: Performance evaluation and metrics
- prediction: Inference and model deployment
"""

__version__ = "1.0.0"
__author__ = "Your Data Science Team"

# Import main functions for easy access
from .data_preprocessing import load_and_preprocess_data
from .model_training import train_model
from .model_evaluation import evaluate_model
from .prediction import predict_churn, save_model, load_model

__all__ = [
    'load_and_preprocess_data',
    'train_model',
    'evaluate_model',
    'predict_churn',
    'save_model',
    'load_model'
]