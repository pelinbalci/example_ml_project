"""
Data preprocessing module for customer churn prediction.

This module handles:
- Loading data from CSV files
- Cleaning and preparing data for machine learning
- Splitting data into training and testing sets
- Feature engineering and scaling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import config


def load_data(file_path):
    """
    Load data from CSV file.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pandas.DataFrame: Loaded data
    """
    try:
        data = pd.read_csv(file_path)
        print(f"   ✓ Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def clean_data(data):
    """
    Clean the dataset by handling missing values and outliers.

    Args:
        data (pandas.DataFrame): Raw data

    Returns:
        pandas.DataFrame: Cleaned data
    """
    print("   🧹 Cleaning data...")

    # Make a copy to avoid modifying original data
    cleaned_data = data.copy()

    # Handle missing values
    missing_before = cleaned_data.isnull().sum().sum()

    # Fill missing numerical values with median
    numerical_cols = cleaned_data.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if cleaned_data[col].isnull().any():
            median_value = cleaned_data[col].median()
            cleaned_data[col].fillna(median_value, inplace=True)

    # Fill missing categorical values with mode (most frequent value)
    categorical_cols = cleaned_data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != config.TARGET_COLUMN and cleaned_data[col].isnull().any():
            mode_value = cleaned_data[col].mode()[0]
            cleaned_data[col].fillna(mode_value, inplace=True)

    missing_after = cleaned_data.isnull().sum().sum()
    print(f"   ✓ Missing values handled: {missing_before} → {missing_after}")

    # Remove extreme outliers (values beyond 3 standard deviations)
    original_size = len(cleaned_data)

    for col in config.NUMERICAL_FEATURES:
        if col in cleaned_data.columns:
            mean_val = cleaned_data[col].mean()
            std_val = cleaned_data[col].std()
            outlier_threshold = 3

            # Keep values within 3 standard deviations
            cleaned_data = cleaned_data[
                (cleaned_data[col] >= mean_val - outlier_threshold * std_val) &
                (cleaned_data[col] <= mean_val + outlier_threshold * std_val)
                ]

    outliers_removed = original_size - len(cleaned_data)
    if outliers_removed > 0:
        print(f"   ✓ Outliers removed: {outliers_removed} rows")

    return cleaned_data


def create_features(data):
    """
    Create new features from existing ones (feature engineering).

    Args:
        data (pandas.DataFrame): Cleaned data

    Returns:
        pandas.DataFrame: Data with additional features
    """
    print("   🔧 Engineering features...")

    # Make a copy to avoid modifying original data
    featured_data = data.copy()

    # Create new features
    try:
        # Average monthly charge (total charges divided by tenure)
        featured_data['avg_monthly_charge'] = (
                featured_data['total_charges'] /
                np.maximum(featured_data['tenure_months'], 1)  # Avoid division by zero
        )

        # High support calls flag
        featured_data['high_support'] = (featured_data['support_calls'] >= 3).astype(int)

        # Customer tenure category
        featured_data['tenure_category'] = pd.cut(
            featured_data['tenure_months'],
            bins=[0, 6, 24, 72],
            labels=['New', 'Medium', 'Long'],
            include_lowest=True
        )

        # High value customer (above median total charges)
        median_charges = featured_data['total_charges'].median()
        featured_data['high_value'] = (featured_data['total_charges'] > median_charges).astype(int)

        print("   ✓ New features created: avg_monthly_charge, high_support, tenure_category, high_value")

    except Exception as e:
        print(f"   ⚠️  Feature engineering warning: {str(e)}")

    return featured_data


def prepare_features_and_target(data):
    """
    Separate features and target variable, and prepare for modeling.

    Args:
        data (pandas.DataFrame): Processed data

    Returns:
        tuple: (X, y) - features and target
    """
    print("   🎯 Preparing features and target...")

    # Separate target variable
    if config.TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{config.TARGET_COLUMN}' not found in data")

    y = data[config.TARGET_COLUMN]

    # Define features to use
    feature_columns = (
            config.NUMERICAL_FEATURES +
            config.CATEGORICAL_FEATURES +
            ['avg_monthly_charge', 'high_support', 'tenure_category', 'high_value']
    )

    # Select only available features
    available_features = [col for col in feature_columns if col in data.columns]
    X = data[available_features]

    print(f"   ✓ Features selected: {len(available_features)} columns")
    print(f"   ✓ Target distribution: {y.value_counts().to_dict()}")

    return X, y


def create_preprocessor(X):
    """
    Create preprocessing pipeline for features.

    Args:
        X (pandas.DataFrame): Features

    Returns:
        sklearn.ColumnTransformer: Preprocessing pipeline
    """
    print("   ⚙️  Creating preprocessing pipeline...")

    # Identify numerical and categorical columns
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Create preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'  # Keep any remaining columns as-is
    )

    print(f"   ✓ Preprocessing pipeline created")
    print(f"     - Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"     - Categorical features ({len(categorical_features)}): {categorical_features}")

    return preprocessor


def load_and_preprocess_data():
    """
    Complete data preprocessing pipeline.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor)
    """
    # Step 1: Load data
    data = load_data(config.DATA_PATH)

    # Step 2: Clean data
    cleaned_data = clean_data(data)

    # Step 3: Feature engineering
    featured_data = create_features(cleaned_data)

    # Step 4: Prepare features and target
    X, y = prepare_features_and_target(featured_data)

    # Step 5: Split data
    print("   📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y  # Maintain class distribution in splits
    )

    print(f"   ✓ Data split completed:")
    print(f"     - Training set: {len(X_train)} samples")
    print(f"     - Test set: {len(X_test)} samples")

    # Step 6: Create and fit preprocessor
    preprocessor = create_preprocessor(X_train)

    # Fit preprocessor on training data only
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"   ✓ Features preprocessed: {X_train_processed.shape[1]} final features")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Test the preprocessing pipeline
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()
    print("\n🎉 Preprocessing completed successfully!")
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")