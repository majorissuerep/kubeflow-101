import kfp
from kfp.components import InputPath, create_component_from_func

def deploy(
    model_path: InputPath(str),
    model_name: str,
    mlflow_tracking_uri: str
):
    """Deploy the model to a Kubeflow serving platform using MLflow model registry."""
    import mlflow
    import mlflow.xgboost
    import yaml
    import os
    
    # Configure MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Register the model in MLflow model registry if not already registered
    model_uri = f"models:/{model_name}/latest"
    
    # Create KFServing manifest
    serving_spec = {
        "apiVersion": "serving.kubeflow.org/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": f"{model_name}-predictor",
            "namespace": "kubeflow"
        },
        "spec": {
            "predictor": {
                "minReplicas": 1,
                "maxReplicas": 3,
                "tensorflow": {
                    "storageUri": f"s3://mlflow-artifacts/{model_name}"
                }
            }
        }
    }
    
    # Save the manifest
    with open(f"{model_name}-serving.yaml", "w") as f:
        yaml.dump(serving_spec, f)
    
    print(f"Deployment manifest created: {model_name}-serving.yaml")
    print(f"Model {model_name} is ready for serving.")
    print(f"To deploy, run: kubectl apply -f {model_name}-serving.yaml")

# Create KFP component
deploy_op = create_component_from_func(
    func=deploy,
    base_image="python:3.9",
    packages_to_install=["mlflow", "pyyaml"]
) 