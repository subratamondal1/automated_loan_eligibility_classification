# Functions required to Load the dataset
# Functions required to Save the Trained ML Model
import os
import joblib

import pandas as pd
from sklearn.pipeline import Pipeline

from prediction_model.config import config

def load_dataset(filename: str) -> pd.DataFrame:
    """
    Load either Train or Test dataset based on 'filename'.

    Parameters
    ----------
    filename : str
        The name of the dataset file to be loaded.

    Returns
    -------
    pd.DataFrame
        The loaded dataset as a pandas DataFrame.
    """
    filepath: str = os.path.join(config.DATA_PATH, filename)
    _data: pd.DataFrame = pd.read_csv(filepath_or_buffer=filepath)
    return _data

def save_pipeline(pipeline_to_save) -> None:
    """
    Serialization: Save the Model.

    Parameters
    ----------
    pipeline_to_save : Pipeline
        The scikit-learn Pipeline object to be saved.

    Returns
    -------
    None
    """
    savepath: str = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(value=pipeline_to_save, filename=savepath)
    print(f"Model has been saved: {config.MODEL_NAME}")

def load_pipeline() -> Pipeline:
    """
    De-Serialization: Load the Model.

    Returns
    -------
    Pipeline
        The loaded scikit-learn Pipeline object.
    """
    savepath: str = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    loaded_model = joblib.load(filename=savepath)
    print(f"Model has been loaded: {config.MODEL_NAME}")
    return loaded_model

