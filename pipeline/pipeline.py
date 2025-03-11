import kfp
from kfp import dsl
from kfp.components import InputPath, OutputPath, create_component_from_func
import mlflow.xgboost
from kubernetes import client as k8s_client

# Import components
from components.data_prep import data_prep_op
from components.training import train_op
from components.evaluation import evaluate_op
from components.deployment import deploy_op

# Define pipeline
@dsl.pipeline(
    name="XGBoost Classification Pipeline",
    description="A pipeline that trains an XGBoost classifier"
)
def xgboost_pipeline(
    data_path: str,
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
    test_size: float = 0.2,
    random_state: int = 42
):
    # Create PVC to share data between components
    vop = dsl.VolumeOp(
        name="create-pvc",
        resource_name="xgboost-data-pvc",
        size="1Gi",
        modes=dsl.VOLUME_MODE_RWO
    )
    
    # Data preparation step
    data_prep_task = data_prep_op(
        data_path=data_path,
        test_size=test_size,
        random_state=random_state
    ).add_pvolumes({"/data": vop.volume})
    
    # Add TensorBoard visualization
    tensorboard_vis = dsl.ContainerOp(
        name="tensorboard-vis",
        image="tensorflow/tensorflow:latest",
        command=["tensorboard"],
        arguments=["--logdir=/logs", "--port=6006"],
        pvolumes={"/logs": vop.volume}
    )
    
    # Training step
    train_task = train_op(
        x_train_path=data_prep_task.outputs['x_train_path'],
        y_train_path=data_prep_task.outputs['y_train_path'],
        x_test_path=data_prep_task.outputs['x_test_path'],
        y_test_path=data_prep_task.outputs['y_test_path'],
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        model_name=model_name,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective=objective,
        eval_metric=eval_metric,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state
    ).add_pvolumes({"/data": vop.volume, "/logs": vop.volume})
    train_task.after(data_prep_task)
    
    # Evaluation step
    eval_task = evaluate_op(
        model_path=train_task.outputs['model_path'],
        x_test_path=data_prep_task.outputs['x_test_path'],
        y_test_path=data_prep_task.outputs['y_test_path'],
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        run_id=train_task.outputs['run_id']
    ).add_pvolumes({"/data": vop.volume})
    eval_task.after(train_task)
    
    # Deployment step
    deploy_task = deploy_op(
        model_path=train_task.outputs['model_path'],
        model_name=model_name,
        mlflow_tracking_uri=mlflow_tracking_uri
    )
    deploy_task.after(eval_task)
    
    # Add metadata for KFP artifacts tracking
    dsl.get_pipeline_conf().add_op_transformer(
        lambda op: op.set_display_name(op.name)
        .add_pod_annotation("kubeflow-kale-sdk-version", "0.6.0")
    )

def run_pipeline():
    """Compile and run the Kubeflow pipeline."""
    import kfp
    import os
    
    # Compile the pipeline
    pipeline_func = xgboost_pipeline
    pipeline_filename = pipeline_func.__name__ + '.yaml'
    kfp.compiler.Compiler().compile(pipeline_func, pipeline_filename)
    
    # Get KFP client
    client = kfp.Client()
    
    # Create an experiment
    experiment_name = 'xgboost-classification'
    try:
        experiment = client.get_experiment(experiment_name=experiment_name)
    except:
        experiment = client.create_experiment(name=experiment_name)
    
    # Submit a pipeline run
    run_name = 'xgboost-classification-run'
    run_result = client.run_pipeline(
        experiment_id=experiment.id,
        job_name=run_name,
        pipeline_package_path=pipeline_filename,
        params={
            'data_path': '/mnt/data/your_dataset.csv',
            'mlflow_tracking_uri': 'http://mlflow-service.kubeflow.svc.cluster.local:5000',
            'experiment_name': 'xgboost-classification',
            'model_name': 'xgboost-classifier',
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1
        }
    )
    
    # Create a recurring run
    client.create_recurring_run(
        experiment_id=experiment.id,
        job_name=run_name + '-recurring',
        description='Weekly retraining of XGBoost classifier',
        cron_expression='0 0 * * 0',  # Run weekly on Sunday at midnight
        pipeline_package_path=pipeline_filename,
        params={
            'data_path': '/mnt/data/your_dataset.csv',
            'mlflow_tracking_uri': 'http://mlflow-service.kubeflow.svc.cluster.local:5000',
            'experiment_name': 'xgboost-classification',
            'model_name': 'xgboost-classifier',
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1
        }
    )

if __name__ == "__main__":
    run_pipeline() 