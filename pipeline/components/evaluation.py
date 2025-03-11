import kfp
from kfp.components import InputPath, create_component_from_func

def evaluate(
    model_path: InputPath(str),
    x_test_path: InputPath(str),
    y_test_path: InputPath(str),
    mlflow_tracking_uri: str,
    experiment_name: str,
    run_id: str
):
    """Evaluate the trained model on the test set."""
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    import mlflow
    import mlflow.xgboost
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    # Configure MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    # Load the model
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Load test data
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).iloc[:, 0].values
    
    # Make predictions
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Log metrics to MLflow
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log confusion matrix as a text file
        np.savetxt("confusion_matrix.csv", conf_matrix, delimiter=",")
        mlflow.log_artifact("confusion_matrix.csv")
        
        print(f"Evaluation results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")

# Create KFP component
evaluate_op = create_component_from_func(
    func=evaluate,
    base_image="python:3.9",
    packages_to_install=["pandas", "xgboost", "scikit-learn", "mlflow", "numpy"]
) 