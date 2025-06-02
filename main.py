"""
Main script to run the complete Customer Churn Prediction pipeline.

This script will:
1. Load and preprocess the data
2. Train a machine learning model
3. Evaluate the model performance
4. Save the trained model for future use

Usage:
    python main.py
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add src directory to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from src.prediction import save_model
import config


def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['data/raw', 'data/processed', 'models', 'results']

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Directory '{directory}' ready")


def generate_sample_data():
    """Generate sample data if no data file exists."""
    if not os.path.exists(config.DATA_PATH):
        print("📊 No data file found. Generating sample data...")

        import numpy as np
        np.random.seed(config.RANDOM_STATE)

        # Generate synthetic customer data
        n_customers = 1000

        data = {
            'customer_id': [f'CUST_{i:04d}' for i in range(1, n_customers + 1)],
            'age': np.random.normal(45, 15, n_customers).astype(int),
            'tenure_months': np.random.exponential(20, n_customers).astype(int),
            'monthly_charges': np.random.normal(65, 20, n_customers).round(2),
            'total_charges': np.random.exponential(1500, n_customers).round(2),
            'contract_type': np.random.choice(['Monthly', 'Yearly', 'Two-Year'], n_customers, p=[0.5, 0.3, 0.2]),
            'payment_method': np.random.choice(['Credit Card', 'Bank Transfer', 'Electronic Check'], n_customers,
                                               p=[0.4, 0.3, 0.3]),
            'support_calls': np.random.poisson(2, n_customers)
        }

        # Create realistic churn based on features
        churn_probability = (
                0.1 +  # Base churn rate
                (data['support_calls'] > 3) * 0.3 +  # High support calls increase churn
                (data['monthly_charges'] > 80) * 0.2 +  # High charges increase churn
                (data['tenure_months'] < 6) * 0.4 +  # New customers more likely to churn
                (np.array(data['contract_type']) == 'Monthly') * 0.2  # Monthly contracts more churn
        )

        data['churn'] = np.random.binomial(1, np.clip(churn_probability, 0, 1), n_customers)

        # Clean up unrealistic values
        data['age'] = np.clip(data['age'], 18, 90)
        data['tenure_months'] = np.clip(data['tenure_months'], 1, 72)
        data['monthly_charges'] = np.clip(data['monthly_charges'], 20, 150)
        data['total_charges'] = np.maximum(data['total_charges'],
                                           data['monthly_charges'] * data['tenure_months'])

        df = pd.DataFrame(data)

        # Ensure data directory exists
        os.makedirs(os.path.dirname(config.DATA_PATH), exist_ok=True)
        df.to_csv(config.DATA_PATH, index=False)
        print(f"✓ Sample data saved to {config.DATA_PATH}")

        return df

    return None


def main():
    """Run the complete machine learning pipeline."""
    print("🚀 Starting Customer Churn Prediction Pipeline")
    print("=" * 50)

    start_time = datetime.now()

    try:
        # Step 1: Setup
        print("\n📁 Setting up project structure...")
        create_directories()

        # Step 2: Generate or load data
        sample_data = generate_sample_data()

        # Step 3: Load and preprocess data
        print(f"\n📊 Loading data from {config.DATA_PATH}...")
        X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()

        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Features: {X_train.shape[1]}")

        # Step 4: Train model
        print(f"\n🤖 Training {config.MODEL_TYPE} model...")
        model = train_model(X_train, y_train)
        print("   ✓ Model training completed")

        # Step 5: Evaluate model
        print("\n📈 Evaluating model performance...")
        results = evaluate_model(model, X_train, X_test, y_train, y_test)

        # Display results
        print("\n🎯 Model Performance Results:")
        print("-" * 30)
        for metric, value in results.items():
            print(f"   {metric}: {value:.3f}")

        # Step 6: Save model and preprocessor
        print(f"\n💾 Saving model to {config.MODEL_PATH}...")
        save_model(model, preprocessor, config.MODEL_PATH)
        print("   ✓ Model saved successfully")

        # Step 7: Save results
        results_file = os.path.join(config.RESULTS_PATH, 'model_performance.txt')
        with open(results_file, 'w') as f:
            f.write(f"Customer Churn Prediction - Model Performance\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model type: {config.MODEL_TYPE}\n\n")

            for metric, value in results.items():
                f.write(f"{metric}: {value:.3f}\n")

        print(f"   ✓ Results saved to {results_file}")

        # Summary
        elapsed_time = datetime.now() - start_time
        print("\n🎉 Pipeline completed successfully!")
        print(f"   Total time: {elapsed_time.total_seconds():.1f} seconds")
        print(f"   Model accuracy: {results['Test Accuracy']:.1%}")

        print("\n📋 Next steps:")
        print("   • Review model performance in results/model_performance.txt")
        print("   • Use the trained model for predictions with src/prediction.py")
        print("   • Explore data further with notebooks/data_exploration.ipynb")

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("   Check that all dependencies are installed and data is available")
        sys.exit(1)


if __name__ == "__main__":
    main()