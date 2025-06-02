"""
Model evaluation module for customer churn prediction.

This module handles:
- Calculating various performance metrics
- Generating evaluation reports
- Visualizing model performance
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import config


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)

    Returns:
        dict: Dictionary of calculated metrics
    """
    metrics = {}

    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='binary')
    metrics['recall'] = recall_score(y_true, y_pred, average='binary')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')

    # AUC score (if probabilities are provided)
    if y_pred_proba is not None:
        metrics['auc_score'] = roc_auc_score(y_true, y_pred_proba)

    # Confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp

    # Additional metrics
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0

    return metrics


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Comprehensive model evaluation on training and test sets.

    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        dict: Evaluation results
    """
    print("   📊 Evaluating model performance...")

    results = {}

    # Training set evaluation
    print("   📈 Training set evaluation...")
    y_train_pred = model.predict(X_train)
    train_proba = None

    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        train_proba = model.predict_proba(X_train)[:, 1]

    train_metrics = calculate_metrics(y_train, y_train_pred, train_proba)

    # Test set evaluation
    print("   🎯 Test set evaluation...")
    y_test_pred = model.predict(X_test)
    test_proba = None

    if hasattr(model, 'predict_proba'):
        test_proba = model.predict_proba(X_test)[:, 1]

    test_metrics = calculate_metrics(y_test, y_test_pred, test_proba)

    # Organize results
    results = {
        'Training Accuracy': train_metrics['accuracy'],
        'Training Precision': train_metrics['precision'],
        'Training Recall': train_metrics['recall'],
        'Training F1': train_metrics['f1_score'],
        'Test Accuracy': test_metrics['accuracy'],
        'Test Precision': test_metrics['precision'],
        'Test Recall': test_metrics['recall'],
        'Test F1': test_metrics['f1_score']
    }

    # Add AUC if available
    if train_proba is not None:
        results['Training AUC'] = train_metrics['auc_score']
    if test_proba is not None:
        results['Test AUC'] = test_metrics['auc_score']

    # Print detailed confusion matrix for test set
    print("\n   📋 Test Set Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"      Predicted:    No Churn  |  Churn")
    print(f"   Actual:")
    print(f"   No Churn:        {cm[0, 0]:4d}    |   {cm[0, 1]:4d}")
    print(f"   Churn:           {cm[1, 0]:4d}    |   {cm[1, 1]:4d}")

    # Business interpretation
    print("\n   💡 Business Interpretation:")
    total_customers = len(y_test)
    actual_churners = sum(y_test)
    predicted_churners = sum(y_test_pred)
    correctly_identified = test_metrics['true_positives']

    print(f"   • Out of {total_customers} customers:")
    print(f"     - {actual_churners} actually churned")
    print(f"     - {predicted_churners} predicted to churn")
    print(f"     - {correctly_identified} correctly identified as churners")
    print(f"   • We catch {test_metrics['recall']:.1%} of actual churners")
    print(f"   • {test_metrics['precision']:.1%} of our churn predictions are correct")

    # Check for overfitting
    train_test_diff = train_metrics['accuracy'] - test_metrics['accuracy']
    if train_test_diff > 0.05:  # 5% difference threshold
        print(f"\n   ⚠️  Warning: Possible overfitting detected")
        print(
            f"     Training accuracy ({train_metrics['accuracy']:.3f}) is much higher than test accuracy ({test_metrics['accuracy']:.3f})")
    else:
        print(f"\n   ✅ Model shows good generalization")
        print(f"     Training vs Test accuracy difference: {train_test_diff:.3f}")

    return results


def generate_classification_report(y_true, y_pred):
    """
    Generate detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        str: Formatted classification report
    """
    return classification_report(
        y_true, y_pred,
        target_names=['No Churn', 'Churn'],
        digits=3
    )


def interpret_results(results):
    """
    Provide business-friendly interpretation of results.

    Args:
        results (dict): Model evaluation results

    Returns:
        str: Business interpretation
    """
    test_acc = results['Test Accuracy']
    test_precision = results['Test Precision']
    test_recall = results['Test Recall']

    interpretation = []

    # Overall performance
    if test_acc >= 0.85:
        interpretation.append("🟢 Excellent model performance")
    elif test_acc >= 0.75:
        interpretation.append("🟡 Good model performance")
    else:
        interpretation.append("🔴 Model needs improvement")

    # Precision interpretation
    if test_precision >= 0.80:
        interpretation.append(f"🎯 High precision: {test_precision:.1%} of churn predictions are correct")
    else:
        interpretation.append(f"⚠️ Lower precision: Only {test_precision:.1%} of churn predictions are correct")

    # Recall interpretation
    if test_recall >= 0.80:
        interpretation.append(f"🔍 High recall: We catch {test_recall:.1%} of customers who will churn")
    else:
        interpretation.append(f"⚠️ Lower recall: We only catch {test_recall:.1%} of customers who will churn")

    # Recommendations
    interpretation.append("\n💼 Business Recommendations:")
    if test_precision < 0.70:
        interpretation.append(
            "• Focus on reducing false positives to avoid wasting resources on customers who won't churn")
    if test_recall < 0.70:
        interpretation.append("• Consider lowering the classification threshold to catch more churners")
    if test_acc >= 0.80:
        interpretation.append("• Model is ready for deployment - consider A/B testing")

    return "\n".join(interpretation)


if __name__ == "__main__":
    # Test the evaluation module (requires trained model)
    print("🧪 Testing model evaluation module...")

    # This would normally be called from main.py with real data
    from data_preprocessing import load_and_preprocess_data
    from model_training import train_model

    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()
    model = train_model(X_train, y_train)

    results = evaluate_model(model, X_train, X_test, y_train, y_test)

    print(f"\n📊 Evaluation Results:")
    for metric, value in results.items():
        print(f"   {metric}: {value:.3f}")

    interpretation = interpret_results(results)
    print(f"\n{interpretation}")

    print("\n🎉 Model evaluation test completed!")