import os
import pdb
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import config
from .processing.data_handling import load_pipeline, load_dataset

classification_pipeline:Pipeline = load_pipeline(pipeline_to_load=config.MODEL_NAME)

def generate_prediction(data_input) -> dict:
    data = pd.DataFrame(data_input)
    y_pred = classification_pipeline.predict(X=data[config.FEATURES])
    output = np.where(y_pred==1, "Y", "N")
    return {
        "prediction": output
    }

# def generate_prediction() -> None:
#     test_data:pd.DataFrame = load_dataset(filename=config.TEST_FILE)
#     y_pred = classification_pipeline.predict(X=test_data[config.FEATURES])
#     output = np.where(y_pred==1, "Y", "N")
#     print(output)

if __name__ == "__main__":
    data_csv_path:str=os.path.join(config.DATA_PATH,config.TEST_FILE)
    generate_prediction(data_input=data_csv_path)
