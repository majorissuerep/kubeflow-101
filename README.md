# XGBoost Classification Pipeline on Kubeflow

This repository contains a Kubeflow Pipeline for training, evaluating, and deploying an XGBoost classifier with MLflow integration for experiment tracking.

## Prerequisites

1. A running Kubeflow cluster
2. MLflow server configured and accessible
3. S3-compatible storage for MLflow artifacts
4. Kubernetes CLI (kubectl) configured
5. Python 3.9 or later

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure MLflow:
   - Ensure MLflow server is running and accessible
   - Default URI: `http://mlflow-service.kubeflow.svc.cluster.local:5000`
   - Update the URI in `pipeline.py` if different

3. Prepare your data:
   - Place your dataset in a location accessible to the pipeline
   - Default path: `/mnt/data/your_dataset.csv`
   - Update the path in `pipeline.py` if different

## Running the Pipeline

1. Compile and run the pipeline:
```bash
python pipeline/pipeline.py
```

This will:
- Compile the pipeline to YAML
- Create an experiment if it doesn't exist
- Submit a pipeline run
- Create a recurring run (weekly)

## Monitoring and Debugging

### Kubeflow Dashboard

1. Access the Kubeflow Dashboard:
```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```
Then open `http://localhost:8080` in your browser.

2. Navigate to:
   - "Pipelines" to see your pipeline runs
   - "Experiments" to see experiment results
   - "Runs" to see individual run details

### MLflow Tracking

1. Access MLflow UI:
```bash
kubectl port-forward -n kubeflow svc/mlflow 5000:5000
```
Then open `http://localhost:5000` in your browser.

2. In MLflow UI you can:
   - View experiments
   - Compare runs
   - View metrics and artifacts
   - Download models

### Debugging Tips

1. Check pod logs:
```bash
kubectl logs -n kubeflow <pod-name>
```

2. Check pod status:
```bash
kubectl get pods -n kubeflow
```

3. Common issues:
   - Volume mounting problems: Check PVC creation
   - MLflow connection issues: Verify MLflow service
   - Resource constraints: Check pod resource limits

## Pipeline Components

1. Data Preparation (`data_prep.py`):
   - Splits data into train/test sets
   - Saves splits to shared volume

2. Training (`training.py`):
   - Trains XGBoost model
   - Logs metrics and artifacts to MLflow
   - Generates visualizations

3. Evaluation (`evaluation.py`):
   - Evaluates model performance
   - Logs metrics to MLflow
   - Generates confusion matrix

4. Deployment (`deployment.py`):
   - Creates KFServing manifest
   - Prepares model for serving

## Customization

1. Modify pipeline parameters in `pipeline.py`:
   - Model hyperparameters
   - Data paths
   - MLflow settings

2. Adjust component configurations:
   - Resource limits
   - Base images
   - Package dependencies

## Troubleshooting

1. Pipeline fails to start:
   - Check Kubeflow cluster status
   - Verify MLflow connection
   - Check resource quotas

2. Components fail:
   - Check pod logs
   - Verify volume mounts
   - Check resource limits

3. MLflow issues:
   - Verify MLflow service
   - Check storage permissions
   - Verify network connectivity 