import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.xgboost

class XGBoostModel:
    """XGBoost classification model with MLflow tracking."""
    
    def __init__(
        self,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='auc',
        num_boost_round=100,
        early_stopping_rounds=10,
        random_state=42
    ):
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'objective': objective,
            'eval_metric': eval_metric,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'seed': random_state
        }
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the XGBoost model."""
        # Create DMatrix for training
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Create validation set if provided
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'validation'))
        
        # Train model
        params = {
            'objective': self.params['objective'],
            'eval_metric': self.params['eval_metric'],
            'max_depth': self.params['max_depth'],
            'eta': self.params['learning_rate'],
            'subsample': self.params['subsample'],
            'colsample_bytree': self.params['colsample_bytree'],
            'min_child_weight': self.params['min_child_weight'],
            'seed': self.params['seed']
        }
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds
        )
        
        return self
    
    def predict(self, X):
        """Make predictions with the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def evaluate(self, X, y):
        """Evaluate the model performance."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        y_pred_proba = self.predict(X)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1_score': f1_score(y, y_pred, average='weighted'),
            'auc': roc_auc_score(y, y_pred_proba)
        }
        
        return metrics
    
    def save(self, path):
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        self.model.save_model(path)
    
    def load(self, path):
        """Load a model from disk."""
        self.model = xgb.Booster()
        self.model.load_model(path)
        return self
    
    def log_to_mlflow(self, experiment_name, run_name, X_val=None, y_val=None):
        """Log the model and metrics to MLflow."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            for param, value in self.params.items():
                mlflow.log_param(param, value)
            mlflow.log_param('num_boost_round', self.num_boost_round)
            mlflow.log_param('early_stopping_rounds', self.early_stopping_rounds)
            
            # Log metrics if validation data is provided
            if X_val is not None and y_val is not None:
                metrics = self.evaluate(X_val, y_val)
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
            
            # Log the model
            mlflow.xgboost.log_model(self.model, "xgboost_model") 