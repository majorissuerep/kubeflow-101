import kfp
from kfp.components import InputPath, OutputPath, create_component_from_func

def train(
    x_train_path: InputPath(str),
    y_train_path: InputPath(str),
    x_test_path: InputPath(str),
    y_test_path: InputPath(str),
    model_path: OutputPath(str),
    run_id: OutputPath(str),
    mlflow_tracking_uri: str,
    experiment_name: str,
    model_name: str,
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    objective: str = 'binary:logistic',
    eval_metric: str = 'auc',
    num_boost_round: int = 100,
    early_stopping_rounds: int = 10,
    random_state: int = 42
):
    """Train an XGBoost model and track with MLflow."""
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    import mlflow
    import mlflow.xgboost
    import os
    import json
    import logging
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    
    # Configure MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    # Load the data
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0]
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).iloc[:, 0]
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Define parameters
    params = {
        'objective': objective,
        'eval_metric': eval_metric,
        'max_depth': max_depth,
        'eta': learning_rate,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'seed': random_state
    }
    
    # Define watchlist for evaluation during training
    watchlist = [(dtrain, 'train'), (dtest, 'eval')]
    
    # Start MLflow run
    with mlflow.start_run() as run:
        run_id_value = run.info.run_id
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param('num_boost_round', num_boost_round)
        mlflow.log_param('early_stopping_rounds', early_stopping_rounds)
        
        # Create TensorBoard callback
        class TensorBoardCallback(xgb.callback.TrainingCallback):
            def __init__(self, log_dir):
                self.log_dir = log_dir
                os.makedirs(log_dir, exist_ok=True)
                
            def after_iteration(self, model, epoch, evals_log):
                logs = {}
                for data_name, metric_values in evals_log.items():
                    for metric_name, metric_values_per_iter in metric_values.items():
                        logs[f"{data_name}-{metric_name}"] = metric_values_per_iter[-1]
                
                # Write logs for TensorBoard
                with open(f"{self.log_dir}/xgboost_metrics.json", 'a') as f:
                    f.write(json.dumps({"epoch": epoch, **logs}) + '\n')
                
                return False
        
        # Create callback
        tensorboard_callback = TensorBoardCallback(log_dir="/logs")
        
        # Train the model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=watchlist,
            early_stopping_rounds=early_stopping_rounds,
            callbacks=[tensorboard_callback]
        )
        
        # Evaluate the model
        y_pred = model.predict(dtest)
        
        # Generate and log ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig("/logs/roc_curve.png")
        mlflow.log_artifact("/logs/roc_curve.png")
        
        # Generate and log Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        pr_auc = average_precision_score(y_test, y_pred)
        plt.figure()
        plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig("/logs/pr_curve.png")
        mlflow.log_artifact("/logs/pr_curve.png")
        
        # Log metrics
        mlflow.log_metric("auc", roc_auc)
        mlflow.log_metric("average_precision", pr_auc)
        
        # Log feature importance
        feature_importance = model.get_score(importance_type='gain')
        feat_imp = pd.DataFrame({
            'feature': list(feature_importance.keys()),
            'importance': list(feature_importance.values())
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feat_imp['feature'], feat_imp['importance'])
        plt.xlabel('Feature Importance (gain)')
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        plt.savefig("/logs/feature_importance.png")
        mlflow.log_artifact("/logs/feature_importance.png")
        
        # Save the model
        mlflow.xgboost.log_model(model, "xgboost_model", registered_model_name=model_name)
        model.save_model(model_path)
        
        # Write the run ID to output
        with open(run_id, 'w') as f:
            f.write(run_id_value)

# Create KFP component
train_op = create_component_from_func(
    func=train,
    base_image="python:3.9",
    packages_to_install=["pandas", "xgboost", "scikit-learn", "mlflow", "matplotlib", "numpy"]
) 