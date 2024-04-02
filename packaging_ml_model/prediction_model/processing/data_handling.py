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

def load_pipeline(pipeline_to_load: str) -> Pipeline:
    """
    Load a saved scikit-learn Pipeline object from disk.

    This function loads a saved Pipeline object from the specified file path using joblib.

    Parameters
    ----------
    pipeline_to_load : str
        The name of the saved Pipeline object to load.

    Returns
    -------
    Pipeline
        The loaded scikit-learn Pipeline object.
    """
    savepath: str = os.path.join(config.SAVE_MODEL_PATH, pipeline_to_load)
    loaded_model = joblib.load(filename=savepath)
    print(f"Model has been loaded: {pipeline_to_load}")
    return loaded_model


