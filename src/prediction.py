"""
Prediction module for customer churn prediction.

This module handles:
- Loading trained models
- Making predictions on new data
- Saving and loading model artifacts
"""

import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import config


def save_model(model, preprocessor, file_path):
    """
    Save trained model and preprocessor to disk.

    Args:
        model: Trained model
        preprocessor: Fitted preprocessing pipeline
        file_path (str): Path to save the model
    """
    # Ensure models directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save both model and preprocessor together
    model_data = {
        'model': model,
        'preprocessor': preprocessor,
        'saved_at': datetime.now().isoformat(),
        'model_type': config.MODEL_TYPE,
        'features': config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES
    }

    joblib.dump(model_data, file_path)
    print(f"   ✓ Model and preprocessor saved to {file_path}")


def load_model(file_path):
    """
    Load trained model and preprocessor from disk.

    Args:
        file_path (str): Path to the saved model

    Returns:
        tuple: (model, preprocessor, metadata)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")

    try:
        model_data = joblib.load(file_path)
        model = model_data['model']
        preprocessor = model_data['preprocessor']
        metadata = {
            'saved_at': model_data.get('saved_at', 'Unknown'),
            'model_type': model_data.get('model_type', 'Unknown'),
            'features': model_data.get('features', [])
        }

        print(f"   ✓ Model loaded from {file_path}")
        print(f"     Model type: {metadata['model_type']}")
        print(f"     Saved at: {metadata['saved_at']}")

        return model, preprocessor, metadata

    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")


def preprocess_new_data(data, preprocessor):
    """
    Preprocess new data using the fitted preprocessor.

    Args:
        data (pandas.DataFrame): New data to preprocess
        preprocessor: Fitted preprocessing pipeline

    Returns:
        numpy.ndarray: Preprocessed features
    """
    print("   🔧 Preprocessing new data...")

    # Make a copy to avoid modifying original data
    processed_data = data.copy()

    # Apply the same feature engineering as in training
    try:
        # Create the same engineered features
        processed_data['avg_monthly_charge'] = (
                processed_data['total_charges'] /
                np.maximum(processed_data['tenure_months'], 1)
        )

        processed_data['high_support'] = (processed_data['support_calls'] >= 3).astype(int)

        processed_data['tenure_category'] = pd.cut(
            processed_data['tenure_months'],
            bins=[0, 6, 24, 72],
            labels=['New', 'Medium', 'Long'],
            include_lowest=True
        )

        median_charges = processed_data['total_charges'].median()
        processed_data['high_value'] = (processed_data['total_charges'] > median_charges).astype(int)

    except Exception as e:
        print(f"   ⚠️  Warning in feature engineering: {str(e)}")

    # Select the same features used in training
    feature_columns = (
            config.NUMERICAL_FEATURES +
            config.CATEGORICAL_FEATURES +
            ['avg_monthly_charge', 'high_support', 'tenure_category', 'high_value']
    )

    # Select only available features
    available_features = [col for col in feature_columns if col in processed_data.columns]
    X = processed_data[available_features]

    # Apply preprocessing
    X_processed = preprocessor.transform(X)

    print(f"   ✓ Data preprocessed: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")

    return X_processed


def predict_churn(data_path, model_path=None):
    """
    Predict customer churn for new data.

    Args:
        data_path (str): Path to CSV file with customer data
        model_path (str, optional): Path to saved model. Uses default if None.

    Returns:
        pandas.DataFrame: Predictions with probabilities
    """
    print(f"🔮 Making churn predictions for {data_path}")

    # Use default model path if not provided
    if model_path is None:
        model_path = config.MODEL_PATH

    # Load the trained model
    model, preprocessor, metadata = load_model(model_path)

    # Load new data
    try:
        new_data = pd.read_csv(data_path)
        print(f"   ✓ Data loaded: {len(new_data)} customers")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

    # Store customer IDs if available
    customer_ids = new_data['customer_id'] if 'customer_id' in new_data.columns else range(len(new_data))

    # Preprocess the data
    X_processed = preprocess_new_data(new_data, preprocessor)

    # Make predictions
    print("   🎯 Generating predictions...")
    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)[:, 1]  # Probability of churn

    # Create results dataframe
    results = pd.DataFrame({
        'customer_id': customer_ids,
        'churn_prediction': predictions,
        'churn_probability': probabilities,
        'risk_level': pd.cut(
            probabilities,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High']
        )
    })

    # Add confidence level
    results['confidence'] = np.where(
        probabilities > 0.7, 'High',
        np.where(probabilities < 0.3, 'High', 'Medium')
    )

    # Summary statistics
    n_churners = sum(predictions)
    avg_probability = probabilities.mean()
    high_risk = sum(probabilities > 0.7)

    print(f"   📊 Prediction Summary:")
    print(f"     • Total customers: {len(new_data)}")
    print(f"     • Predicted churners: {n_churners} ({n_churners / len(new_data):.1%})")
    print(f"     • Average churn probability: {avg_probability:.1%}")
    print(f"     • High-risk customers (>70% probability): {high_risk}")

    return results


def predict_single_customer(customer_data, model_path=None):
    """
    Predict churn for a single customer.

    Args:
        customer_data (dict): Dictionary with customer features
        model_path (str, optional): Path to saved model

    Returns:
        dict: Prediction result
    """
    # Convert to DataFrame
    df = pd.DataFrame([customer_data])

    # Save to temporary file and use regular prediction function
    temp_file = 'temp_customer.csv'
    df.to_csv(temp_file, index=False)

    try:
        results = predict_churn(temp_file, model_path)
        result = results.iloc[0].to_dict()

        # Clean up temporary file
        os.remove(temp_file)

        return result

    except Exception as e:
        # Clean up temporary file in case of error
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise e


def explain_prediction(customer_data, model_path=None):
    """
    Provide explanation for a customer's churn prediction.

    Args:
        customer_data (dict): Customer features
        model_path (str, optional): Path to saved model

    Returns:
        str: Human-readable explanation
    """
    result = predict_single_customer(customer_data, model_path)

    probability = result['churn_probability']
    prediction = result['churn_prediction']
    risk_level = result['risk_level']

    explanation = []
    explanation.append(f"🎯 Churn Prediction: {'Will Churn' if prediction else 'Will Stay'}")
    explanation.append(f"📊 Probability: {probability:.1%}")
    explanation.append(f"⚠️ Risk Level: {risk_level}")

    # Risk factors based on customer data
    explanation.append("\n🔍 Key Risk Factors:")

    if customer_data.get('support_calls', 0) >= 3:
        explanation.append("• High number of support calls (retention risk)")

    if customer_data.get('monthly_charges', 0) > 80:
        explanation.append("• High monthly charges (price sensitivity)")

    if customer_data.get('tenure_months', 0) < 6:
        explanation.append("• New customer (higher churn rate)")

    if customer_data.get('contract_type') == 'Monthly':
        explanation.append("• Month-to-month contract (easier to cancel)")

    # Recommendations
    explanation.append("\n💡 Recommended Actions:")
    if probability > 0.7:
        explanation.append("• Immediate intervention required")
        explanation.append("• Consider retention offers or discounts")
        explanation.append("• Schedule call with customer success team")
    elif probability > 0.3:
        explanation.append("• Monitor customer closely")
        explanation.append("• Send engagement campaigns")
        explanation.append("• Check satisfaction levels")
    else:
        explanation.append("• Customer appears stable")
        explanation.append("• Continue regular engagement")

    return "\n".join(explanation)


if __name__ == "__main__":
    # Test the prediction module
    print("🧪 Testing prediction module...")

    # Example customer data
    test_customer = {
        'customer_id': 'TEST_001',
        'age': 45,
        'tenure_months': 3,
        'monthly_charges': 85.50,
        'total_charges': 256.50,
        'contract_type': 'Monthly',
        'payment_method': 'Credit Card',
        'support_calls': 5
    }

    print("📝 Test customer data:")
    for key, value in test_customer.items():
        print(f"   {key}: {value}")

    # Note: This requires a trained model to exist
    try:
        result = predict_single_customer(test_customer)
        print(f"\n🎯 Prediction result: {result}")

        explanation = explain_prediction(test_customer)
        print(f"\n{explanation}")

    except FileNotFoundError:
        print("\n⚠️  No trained model found. Run main.py first to train a model.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

    print("\n🎉 Prediction module test completed!")