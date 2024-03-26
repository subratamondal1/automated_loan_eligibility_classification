import os
import pathlib

import prediction_model # the directory itself where we are working

# Access the Path of the Root Directory
PACKAGE_ROOT:pathlib.Path = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH:str = os.path.join(PACKAGE_ROOT, "data")

TRAIN_FILE:str = "loan-train.csv"
TEST_FILE:str = "loan-test.csv"

MODEL_NAME = "Classification.pkl"
SAVE_MODEL_PATH:str = os.path.join(PACKAGE_ROOT, "trained_models")

# Final features used in the Model
FEATURES:list[str] = ['ApplicantIncome', 'Credit_History', 'Dependents', 'Education', 'Gender', 'LoanAmount', 'Loan_Amount_Term', 'Married', 'Property_Area', 'Self_Employed']

# Numerical Features
NUM_FEATURES:list[str] = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

# Categorical Features
CAT_FEATURES:list[str] = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']

TARGET_FEATURE:str = "Loan_Status"

# Features that we need to Encode
FEATURES_TO_ENCODE:list[str] = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']

FEATURES_TO_MODIFY:list[str] = ["ApplicantIncome"]
FEATURES_TO_ADD:list[str] = ["CoapplicantIncome"]

DROP_FEATURES:list[str] = ["CoapplicantIncome"]

# Feature Engineering: Log Transformations
FEATURES_TO_LOG_TRANSFORM:list[str] = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
