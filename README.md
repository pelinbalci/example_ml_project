# Customer Churn Prediction - ML Project Template

## 📋 Project Overview

This project predicts whether customers will stop using our service (churn) based on their usage patterns and characteristics. It serves as a template for machine learning projects in our company.

**Problem Type:** Binary Classification  
**Business Value:** Identify at-risk customers early to improve retention  
**Model Accuracy:** ~85% (see results section)

## 🏗️ Project Structure

```
customer-churn-prediction/
│
├── README.md                 # This file - project documentation
├── requirements.txt          # Python dependencies
├── .gitignore               # Files to ignore in git
├── main.py                  # Main script to run everything
├── config.py                # Configuration settings
│
├── src/                     # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py # Data cleaning and preparation
│   ├── model_training.py    # Model training logic
│   ├── model_evaluation.py  # Model evaluation metrics
│   └── prediction.py        # Inference/prediction logic
│
├── data/                    # Data files
│   ├── raw/                 # Original, unprocessed data
│   ├── processed/           # Cleaned, ready-to-use data
│   └── sample_data.csv      # Example dataset
│
├── models/                  # Saved trained models
│   └── customer_churn_model.pkl
│
├── notebooks/               # Jupyter notebooks for exploration
│   └── data_exploration.ipynb
│
└── results/                 # Model results and reports
    └── model_performance.txt
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or download this project
# Navigate to project directory
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
# This will train the model and make predictions
python main.py
```

### 3. Make Predictions on New Data

```bash
# After training, predict on new customers
python -c "
from src.prediction import predict_churn
result = predict_churn('data/new_customer.csv')
print(result)
"
```

## 📊 Dataset Information

### Sample Data Format
The dataset should contain customer information with these columns:

| Column | Description | Example |
|--------|-------------|---------|
| customer_id | Unique identifier | CUST_001 |
| age | Customer age | 35 |
| tenure_months | Months as customer | 24 |
| monthly_charges | Monthly fee | 65.50 |
| total_charges | Total paid | 1570.00 |
| contract_type | Contract length | Monthly/Yearly |
| payment_method | How they pay | Credit Card |
| support_calls | Support tickets | 3 |
| churn | Target variable | 0 (stay) or 1 (leave) |

### Data Split
- **Training Set:** 70% of data (used to teach the model)
- **Validation Set:** 15% of data (used to tune the model)
- **Test Set:** 15% of data (used to evaluate final performance)

## 🎯 Model Performance

### Current Results
- **Training Accuracy:** 87.2%
- **Validation Accuracy:** 85.8% 
- **Test Accuracy:** 85.1%
- **Precision:** 83.4% (of predicted churners, 83.4% actually churn)
- **Recall:** 79.6% (we catch 79.6% of actual churners)

### What These Numbers Mean
- **Accuracy:** How often our predictions are correct overall
- **Precision:** When we predict someone will churn, how often are we right?
- **Recall:** Of all customers who actually churn, how many do we catch?

## 🔧 Configuration

Edit `config.py` to customize the project:

```python
# Model settings
MODEL_TYPE = 'random_forest'  # or 'logistic_regression'
TEST_SIZE = 0.15             # Portion of data for testing
RANDOM_STATE = 42            # For reproducible results

# File paths
DATA_PATH = 'data/sample_data.csv'
MODEL_PATH = 'models/customer_churn_model.pkl'
```

## 📝 How to Adapt This Project

### For Your Own Classification Problem

1. **Replace the dataset:**
   - Put your CSV file in `data/raw/`
   - Update `DATA_PATH` in `config.py`

2. **Update column names:**
   - Modify `data_preprocessing.py` to match your columns
   - Change the target variable name from 'churn'

3. **Adjust the README:**
   - Change project title and description
   - Update the dataset table with your columns
   - Modify business value statement

4. **Customize features:**
   - Add/remove features in `data_preprocessing.py`
   - Update the feature engineering section

### For Regression Problems
- Change `MODEL_TYPE` to regression models
- Update evaluation metrics in `model_evaluation.py`
- Modify prediction output format

## 📚 Code Explanation

### Main Components

**main.py** - Orchestrates the entire process:
1. Loads and cleans data
2. Trains the model
3. Evaluates performance
4. Saves the trained model

**data_preprocessing.py** - Prepares data for modeling:
- Handles missing values
- Converts text to numbers
- Scales numerical features
- Splits data into train/test sets

**model_training.py** - Creates and trains the model:
- Defines model architecture
- Fits model to training data
- Saves trained model for later use

**prediction.py** - Makes predictions on new data:
- Loads saved model
- Processes new data same way as training data
- Returns predictions with confidence scores

## 🛠️ Dependencies

See `requirements.txt` for full list. Main libraries:
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning algorithms
- **numpy**: Numerical computations
- **matplotlib**: Data visualization

## 🤔 Troubleshooting

### Common Issues

**"Module not found" error:**
```bash
pip install -r requirements.txt
```

**"File not found" error:**
- Check file paths in `config.py`
- Ensure data files are in correct directories

**Poor model performance:**
- Check data quality
- Try different model types in `config.py`
- Increase training data size

## 📈 Next Steps

1. **Collect more data** - More examples usually improve accuracy
2. **Feature engineering** - Create new features from existing ones
3. **Try different models** - Test other algorithms
4. **Hyperparameter tuning** - Optimize model settings
5. **Deploy model** - Set up automatic predictions

## 👥 Team Guidelines

### Before Starting Your Project
1. Copy this template to a new folder
2. Update the README with your project details
3. Replace sample data with your dataset
4. Test that everything runs without errors

### Code Standards
- Keep functions simple and well-commented
- Use descriptive variable names
- Test your code before sharing
- Update documentation when you make changes

---

**Need Help?** Contact the Data Science team or create an issue in this repository.

**Last Updated:** June 2025