import mlflow.pytorch
# from azureml.core import Workspace
# from azureml.core.authentication import InteractiveLoginAuthentication
from mlflow.tracking import MlflowClient

# # authenticate with Azure
# auth = InteractiveLoginAuthentication()
# ws = Workspace.from_config(auth=auth)

# set the tracking URI to the Azure ML workspace
# mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# define the run ID for the run that contains the model
run_id = "8f2ce14047d14951858e2d51ad738ea5"  # fill in the run ID

# Log the trained YOLOv5 model to an MLflow run
with mlflow.start_run(run_id=run_id) as run:
    mlflow.pytorch.log_model("model", "runs/train/exp23/weights/", registered_model_name="yolov5")

    # Register the YOLOv5 model in the MLflow model registry
    mlflow.register_model(f"runs:/{run.info.run_id}/model", "yolov5")
