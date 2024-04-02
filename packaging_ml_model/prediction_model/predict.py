import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from prediction_model.config import config
from prediction_model.processing.data_handling import load_pipeline

classification_pipeline:Pipeline = load_pipeline(pipeline_to_load=config.MODEL_NAME)

def generate_prediction(data:pd.DataFrame) -> dict:
    _df:pd.DataFrame = data
    _y_pred = classification_pipeline.predict(X=_df[config.FEATURES])
    _result = np.where(_y_pred == 1, "Y", "N")
    return {
        "predictions":_result
    }