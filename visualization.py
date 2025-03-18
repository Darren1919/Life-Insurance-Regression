# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_actual_vs_predicted(y_true, y_pred):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.xlabel("Actual Claims")
    plt.ylabel("Predicted Claims")
    plt.title("Actual vs Predicted Claims")
    plt.show()

def plot_feature_importance(model, feature_names):
    importance = np.abs(model.coef_)
    sorted_idx = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10,6))
    plt.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title("Ridge Regression Feature Importance")
    plt.show()
