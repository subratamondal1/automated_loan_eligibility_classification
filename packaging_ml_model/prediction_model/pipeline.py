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

"""
A machine learning pipeline for classification tasks.

The pipeline performs the following steps:
1. Imputes missing values in numerical features using mean imputation.
2. Imputes missing values in categorical features using mode imputation.
3. Combines two columns into one by adding their values.
4. Drops unnecessary columns from the input data.
5. Encodes categorical features as numerical values based on their frequency in the input data.
6. Applies logarithmic transformations to numerical features with a positively skewed distribution.
7. Scales numerical features to a fixed range using Min-Max scaling.
8. Trains a logistic regression model on the preprocessed data.
"""