from mlflow.tracking import MlflowClient

client = MlflowClient()

# Replace with your experiment name if you’ve set one explicitly
experiment_name = "Default"
experiment = client.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

runs = client.search_runs(experiment_ids=[experiment_id], order_by=["metrics.accuracy DESC"])

if not runs:
    print("No runs found!")
else:
    best_run = runs[0]
    model_name = "BestIrisModel"
    run_id = best_run.info.run_id
    model_path = f"runs:/{run_id}/{best_run.data.tags['mlflow.runName']}"

    print(f"Best model: {best_run.data.tags['mlflow.runName']} with accuracy {best_run.data.metrics['accuracy']:.2f}")

    # Create model registry entry (skip if already created)
    try:
        client.create_registered_model(model_name)
    except Exception as e:
        print(f"Model already exists or error occurred: {e}")

    # Register a new version of the model
    client.create_model_version(name=model_name, source=model_path, run_id=run_id)
    print(f"Registered {model_name} successfully")