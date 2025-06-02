"""
Model training module for customer churn prediction.

This module handles:
- Model selection and configuration
- Training machine learning models
- Model hyperparameter tuning
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import config


def create_model():
    """
    Create a machine learning model based on configuration.

    Returns:
        sklearn model: Configured model ready for training
    """
    if config.MODEL_TYPE == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS,
            max_depth=config.MAX_DEPTH,
            random_state=config.RANDOM_STATE,
            class_weight='balanced'  # Handle class imbalance
        )
        print(f"   ✓ Random Forest model created with {config.N_ESTIMATORS} trees")

    elif config.MODEL_TYPE == 'logistic_regression':
        model = LogisticRegression(
            random_state=config.RANDOM_STATE,
            class_weight='balanced',  # Handle class imbalance
            max_iter=1000  # Ensure convergence
        )
        print("   ✓ Logistic Regression model created")

    else:
        raise ValueError(f"Unknown model type: {config.MODEL_TYPE}")

    return model


def train_model(X_train, y_train):
    """
    Train the machine learning model.

    Args:
        X_train: Training features
        y_train: Training target values

    Returns:
        trained model: Fitted model ready for predictions
    """
    print("   🏋️  Training model...")

    # Create model
    model = create_model()

    # Perform cross-validation to get training performance estimate
    print("   📊 Performing cross-validation...")
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5,  # 5-fold cross-validation
        scoring='accuracy'
    )

    print(f"   ✓ Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Train the final model on all training data
    model.fit(X_train, y_train)

    # Calculate training accuracy
    train_accuracy = model.score(X_train, y_train)
    print(f"   ✓ Training accuracy: {train_accuracy:.3f}")

    # Show feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        print("   📈 Top 5 most important features:")
        feature_importance = model.feature_importances_

        # Create feature names (since we don't have them after preprocessing)
        n_features = len(feature_importance)
        feature_names = [f"feature_{i}" for i in range(n_features)]

        # Get top 5 features
        top_indices = np.argsort(feature_importance)[-5:][::-1]
        for i, idx in enumerate(top_indices):
            importance = feature_importance[idx]
            print(f"     {i + 1}. {feature_names[idx]}: {importance:.3f}")

    print("   ✅ Model training completed")
    return model


def get_model_info(model):
    """
    Get information about the trained model.

    Args:
        model: Trained model

    Returns:
        dict: Model information
    """
    info = {
        'model_type': type(model).__name__,
        'parameters': model.get_params()
    }

    # Add model-specific information
    if hasattr(model, 'n_estimators'):
        info['n_estimators'] = model.n_estimators

    if hasattr(model, 'max_depth'):
        info['max_depth'] = model.max_depth

    if hasattr(model, 'feature_importances_'):
        info['has_feature_importance'] = True
        info['n_features'] = len(model.feature_importances_)

    return info


def quick_model_comparison(X_train, y_train):
    """
    Quickly compare different model types to help with selection.

    Args:
        X_train: Training features
        y_train: Training target values

    Returns:
        dict: Comparison results
    """
    print("   🏆 Comparing model types...")

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=50,  # Smaller for speed
            random_state=config.RANDOM_STATE,
            class_weight='balanced'
        ),
        'Logistic Regression': LogisticRegression(
            random_state=config.RANDOM_STATE,
            class_weight='balanced',
            max_iter=1000
        )
    }

    results = {}

    for name, model in models.items():
        # Perform 3-fold cross-validation for speed
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        results[name] = {
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std()
        }
        print(f"     {name}: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]['mean_accuracy'])
    print(f"   ✓ Best performing model: {best_model}")

    return results


if __name__ == "__main__":
    # Test the training module (requires preprocessed data)
    print("🧪 Testing model training module...")

    # This is just for testing - normally called from main.py
    from data_preprocessing import load_and_preprocess_data

    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()

    # Compare models
    comparison = quick_model_comparison(X_train, y_train)

    # Train the configured model
    model = train_model(X_train, y_train)

    # Get model info
    info = get_model_info(model)
    print(f"\n📋 Model info: {info}")

    print("\n🎉 Model training test completed!")