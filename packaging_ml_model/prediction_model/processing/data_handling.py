# Functions required to Load the dataset
# Functions required to Save the Trained ML Model

import os
import pandas as pd
import joblib

from prediction_model.config import config

def load_dataset(filename) -> pd.DataFrame:
    """Load the dataset"""
    "Load either Train or Test data based on 'filename'"
    filepath:str = os.path.join(config.DATAPATH, filename)
    _data:pd.DataFrame = pd.read_csv(filepath_or_buffer=filepath)
    return _data

def save_pipeline(pipeline_to_save) -> None:
    """Serialization"""
    savepath:str = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(value=pipeline_to_save, filename=savepath)
    print(f"Model has been saved: {config.MODEL_NAME}")

def load_pipeline(pipeline_to_load) -> None:
    """De-Serialization"""
    savepath:str = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    loaded_model = joblib.dump(value=pipeline_to_load, filename=savepath)
    print(f"Model has been saved: {config.MODEL_NAME}")
    return loaded_model
