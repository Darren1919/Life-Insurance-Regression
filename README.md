# 🚀 Life Insurance Regression (FYP)

This project applies **Robust Ridge Regression techniques** to predict life insurance claims.  
We compare **OLS, Ridge, WRM, and WRMM models** and evaluate their performance using RMSE, MAE, and R².

## 📊 Dataset Information
- **Source:** Synthetic dataset for insurance claims analysis
- **Rows:** 1,000
- **Columns:** 25
- **Target Variable:** `total_claim_amount`

## 🚀 How to Run (with Dataset)
```bash
# Install dependencies
pip install -r requirements.txt

# Run preprocessing
python src/data_preprocessing.py

# Train model using dataset
python src/train_model.py


## 📊 Visualizations
Here are some key results from the analysis:

### Actual vs Predicted Claims
![Actual vs Predicted](results/actual_vs_predicted.png)

### Feature Importance
![Feature Importance](results/feature_importance.png)
