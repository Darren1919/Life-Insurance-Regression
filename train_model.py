# train_model.py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from data_preprocessing import load_data, preprocess_data

def train_ridge_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameter tuning
    param_grid = {'alpha': [0.1, 1, 10, 100]}
    grid = GridSearchCV(Ridge(), param_grid, cv=5)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    return best_model, X_test, y_test
