# Life-Insurance-Regression.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("insurance_claims.csv")  # Ensure the file is in the same directory

# Display first few rows
print("Dataset Preview:\n", df.head())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Drop irrelevant columns (Modify if needed)
df = df.drop(columns=['policy_number', 'policy_bind_date', 'incident_location'], errors='ignore')

# Convert categorical variables to numerical
df = pd.get_dummies(df, drop_first=True)

# Define target variable (e.g., 'total_claim_amount') and features
target = "total_claim_amount"
X = df.drop(columns=[target], axis=1)
y = df[target]

# Split data into train and test sets (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:\nMAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR² Score: {r2:.2f}")

# Visualization: Actual vs Predicted
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.xlabel("Actual Claim Amount")
plt.ylabel("Predicted Claim Amount")
plt.title("Actual vs Predicted Claim Amount")
plt.show()

# Save results
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_df.to_csv("Regression_Results.csv", index=False)
