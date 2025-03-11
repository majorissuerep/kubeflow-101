import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import os

def setup_mlflow(tracking_uri: str, experiment_name: str):
    """Set up MLflow tracking."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """Log metrics to MLflow."""
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value, step=step)

def log_params(params: Dict[str, Any]):
    """Log parameters to MLflow."""
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)

def log_artifacts(artifacts: Dict[str, Any], artifact_dir: str):
    """Log artifacts to MLflow."""
    os.makedirs(artifact_dir, exist_ok=True)
    
    for artifact_name, artifact_data in artifacts.items():
        if isinstance(artifact_data, (pd.DataFrame, np.ndarray)):
            file_path = os.path.join(artifact_dir, f"{artifact_name}.csv")
            if isinstance(artifact_data, pd.DataFrame):
                artifact_data.to_csv(file_path, index=False)
            else:
                pd.DataFrame(artifact_data).to_csv(file_path, index=False)
            mlflow.log_artifact(file_path)
        else:
            file_path = os.path.join(artifact_dir, f"{artifact_name}.txt")
            with open(file_path, 'w') as f:
                f.write(str(artifact_data))
            mlflow.log_artifact(file_path)

def log_model(model: Any, model_name: str, registered_model_name: Optional[str] = None):
    """Log a model to MLflow."""
    if hasattr(model, 'save_model'):  # XGBoost model
        mlflow.xgboost.log_model(model, model_name, registered_model_name=registered_model_name)
    else:
        mlflow.sklearn.log_model(model, model_name, registered_model_name=registered_model_name)

def get_best_model(experiment_name: str, metric: str = 'auc', ascending: bool = False):
    """Get the best model from an experiment based on a metric."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")
    
    best_run = runs.sort_values(by=f'metrics.{metric}', ascending=ascending).iloc[0]
    model_uri = f"runs:/{best_run.run_id}/model"
    
    return mlflow.xgboost.load_model(model_uri)

def compare_runs(experiment_name: str, metric: str = 'auc'):
    """Compare runs in an experiment based on a metric."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")
    
    comparison = runs[[f'metrics.{metric}', 'params.learning_rate', 'params.max_depth', 'params.n_estimators']]
    return comparison.sort_values(by=f'metrics.{metric}', ascending=False)

def get_model_versions(model_name: str):
    """Get all versions of a registered model."""
    try:
        versions = mlflow.search_model_versions(f"name='{model_name}'")
        return versions
    except Exception as e:
        print(f"Error getting model versions: {e}")
        return None 