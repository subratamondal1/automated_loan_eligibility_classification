import os
import time
import argparse

import mlflow

def eval(p1, p2) -> float:
    return p1**2 + p2**2

def main(input1, input2) -> None:
    with mlflow.start_run():
        mlflow.log_param(key="parameter1", value=input1)
        mlflow.log_param(key="parameter2", value=input2)
        metric = eval(p1=input1, p2=input2)
        mlflow.log_metric(key="Eval_Metric", value=metric)

        os.makedirs(name="dummy", exist_ok=True)
        with open(file="dummy/example.txt", mode="wt") as f:
            f.write(f"Artifact created at: {time.asctime()}")
        mlflow.log_artifacts(local_dir="dummy")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--parameter1", "-p1", type=int, default=5)
    args.add_argument("--parameter2", "-p2", type=int, default=10)
    parsed_args = args.parse_args()

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    # accessing the user given input
    main(input1=parsed_args.parameter1, input2=parsed_args.parameter2)

