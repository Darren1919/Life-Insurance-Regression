# data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, categorical_features, numerical_features):
    # Handling categorical encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    X = preprocessor.fit_transform(df.drop(columns=['target']))
    y = df['target']
    return X, y, preprocessor
