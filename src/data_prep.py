import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
    print(df.info())
    return df

def preprocess_pipeline():
    # Define column types (adapt to your exact column names)
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['cp', 'restecg', 'slope', 'thal', 'ca']
    binary_features = ['sex', 'fbs', 'exang']  # Keep as-is
    
    # Numeric pipeline: impute + scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: impute + one-hot
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Full preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ],
        remainder='passthrough'  # Keep binary features unchanged
    )
    return preprocessor, numeric_features, categorical_features

def prepare_data(df, test_size=0.2, random_state=42):
    # Target: binarize if needed (0=no disease, 1=disease)
    if df['target'].nunique() > 2:
        df['target'] = (df['target'] > 0).astype(int)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
