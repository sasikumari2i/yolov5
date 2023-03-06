import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
# set the experiment name and metric name
experiment_name = "chrp_exp5"
metric_name = "Classifier_Accuracy"

# create an MLflow client and get the experiment ID
client = MlflowClient()
experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

# get all runs for the experiment
runs = client.search_runs(experiment_id)



# create a dictionary to store the metric values for each run
metric_values = {}

# loop through each run and get the metric value
for run in runs:
    run_id = run.info.run_id
    # Get the MLflow run object for the run ID
    run_obj = client.get_run(run_id)
    # Get the value of the metric
    if metric_name in run.data.metrics and run.data.metrics[metric_name] is not None:
        # Get the value of the metric
        metric_value = run_obj.data.metrics[metric_name]
        metric_values[run_id] = metric_value

# find the run with the best metric value
best_run_id = max(metric_values, key=metric_values.get)
print(best_run_id)

# Get the run name from the run object
best_run=client.get_run(best_run_id)
run_name = best_run.info.run_name

print("Run name:", run_name)


