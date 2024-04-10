import mlflow

# Tracking URI
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

experiment_name:str = "Loan Eligibility"
# Experiment ID
experiment_id:str = mlflow.create_experiment(name=experiment_name)

with mlflow.start_run(run_id="Logistic Regression") as run:
    pass

mlflow.end_run()
