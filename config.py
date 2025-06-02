"""
Configuration settings for the Customer Churn Prediction project.
Modify these settings to customize the project for your needs.
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'sample_data.csv')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'customer_churn_model.pkl')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')

# Data processing settings
TEST_SIZE = 0.15        # Portion of data reserved for final testing
VALIDATION_SIZE = 0.15  # Portion of data used for model validation
RANDOM_STATE = 42       # For reproducible results

# Model settings
MODEL_TYPE = 'random_forest'  # Options: 'random_forest', 'logistic_regression'
N_ESTIMATORS = 100           # Number of trees (for random forest)
MAX_DEPTH = 10               # Maximum tree depth

# Feature engineering
NUMERICAL_FEATURES = [
    'age',
    'tenure_months',
    'monthly_charges',
    'total_charges',
    'support_calls'
]

CATEGORICAL_FEATURES = [
    'contract_type',
    'payment_method'
]

TARGET_COLUMN = 'churn'

# Evaluation settings
CLASSIFICATION_THRESHOLD = 0.5  # Probability threshold for classification

# Display settings
RANDOM_SEED = 42
FIGSIZE = (10, 6)  # Default figure size for plots