from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

from prediction_model.config import config
from prediction_model.processing import data_preprocessing as data_pp

classification_pipeline = Pipeline(
    steps= [
        ("MeanImputation", data_pp.MeanImputer(numerical_features=config.NUM_FEATURES)),
        ("ModeImputation", data_pp.ModeImputer(categorical_features=config.CAT_FEATURES)),
        ("CombineColumns", data_pp.CombineColumns(columnA=config.FEATURES_TO_MODIFY, columnB=config.FEATURES_TO_ADD)),
        ("DropColumns", data_pp.DropColumns(columns_to_drop=config.DROP_FEATURES)),
        ("LabelEncoding", data_pp.CustomLabelEncoder(categorical_features=config.FEATURES_TO_ENCODE)),
        ("LogTransformation", data_pp.LogTransform(numerical_features=config.FEATURES_TO_LOG_TRANSFORM)),
        ("MinMaxScaling", MinMaxScaler()),
        ("LogisticRegression", LogisticRegression(random_state=0))
    ]
)