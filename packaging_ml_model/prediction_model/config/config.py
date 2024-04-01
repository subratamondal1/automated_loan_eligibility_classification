import os
import pathlib

import prediction_model  # Import the directory itself where we are working

# Access the Path of the Root Directory
PACKAGE_ROOT_PATH = pathlib.Path(prediction_model.__file__).resolve().parent  # Get the root path of the package

DATA_PATH:str = os.path.join(PACKAGE_ROOT_PATH, "data")  # Path to the data directory

TRAIN_FILE:str = "loan-train.csv"  # Name of the training dataset file
TEST_FILE:str = "loan-test.csv"  # Name of the testing dataset file

MODEL_NAME:str = "Classification.pkl"  # Name of the saved model file
SAVE_MODEL_PATH:str= os.path.join(PACKAGE_ROOT_PATH, "trained_models")  # Path to the directory where the trained model will be saved

# Final features used in the Model
FEATURES:list[str] = ['ApplicantIncome', 'Credit_History', 'Dependents', 'Education', 'Gender', 'LoanAmount', 'Loan_Amount_Term', 'Married', 'Property_Area', 'Self_Employed']

# Numerical Features
NUM_FEATURES:list[str] = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

# Categorical Features
CAT_FEATURES:list[str] = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']

TARGET_FEATURE:str = "Loan_Status"  # Target variable to predict

# Features that we need to Encode
FEATURES_TO_ENCODE:list[str] = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']

FEATURES_TO_MODIFY:str = "ApplicantIncome"  # Feature to modify
FEATURES_TO_ADD:str = "CoapplicantIncome"  # Feature to add

DROP_FEATURES:list[str] = ["CoapplicantIncome"]  # Features to drop

# Feature Engineering: Log Transformations
FEATURES_TO_LOG_TRANSFORM:list[str] = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']  # Features to apply log transformation
