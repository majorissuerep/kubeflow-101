import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class DataProcessor:
    """Utility class for data preprocessing."""
    
    def __init__(self, categorical_cols=None, numerical_cols=None):
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.preprocessor = None
    
    def fit_transform(self, X, y=None):
        """Fit the preprocessor and transform the data."""
        # Automatically detect column types if not specified
        if self.categorical_cols is None or self.numerical_cols is None:
            self._detect_column_types(X)
        
        # Create preprocessing pipelines
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        
        # Create column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ]
        )
        
        # Fit and transform the data
        X_transformed = self.preprocessor.fit_transform(X)
        
        return X_transformed, y
    
    def transform(self, X):
        """Transform new data using the fitted preprocessor."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted yet.")
        return self.preprocessor.transform(X)
    
    def _detect_column_types(self, X):
        """Automatically detect categorical and numerical columns."""
        if isinstance(X, pd.DataFrame):
            self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        else:
            raise ValueError("Input must be a pandas DataFrame")
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def load_data(self, file_path):
        """Load data from a file."""
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Parquet files.")
    
    def save_data(self, data, file_path):
        """Save data to a file."""
        if isinstance(data, pd.DataFrame):
            if file_path.endswith('.csv'):
                data.to_csv(file_path, index=False)
            elif file_path.endswith('.parquet'):
                data.to_parquet(file_path, index=False)
            else:
                raise ValueError("Unsupported file format. Use CSV or Parquet files.")
        else:
            raise ValueError("Data must be a pandas DataFrame") 