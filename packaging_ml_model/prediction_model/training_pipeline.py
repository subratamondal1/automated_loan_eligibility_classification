import pandas as pd

from .config import config
from prediction_model.processing.data_handling import load_dataset, save_pipeline
from prediction_model import pipeline

def perform_training() -> None:
    train_data:pd.DataFrame = load_dataset(filename=config.TRAIN_FILE)
    train_X:pd.DataFrame = train_data[config.FEATURES]
    train_y:pd.Series = train_data[config.TARGET_FEATURE].map(arg={
        "N":0,
        "Y":1
    })
    pipeline.classification_pipeline.fit(X=train_X, y=train_y)
    save_pipeline(pipeline_to_save=pipeline.classification_pipeline)

if __name__ == "__main__":
    perform_training()