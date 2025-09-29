import os
from datetime import date

DATABASE_NAME = "AIML_1013_Project1"
COLLECTION_NAME = "telco_data2025"
MONGODBURL = "MongoDB_URL"

PIPELINE_NAME: str = 'telco_churn_pipeline'
ARTIFACT_DIR: str = 'artifacts'

MODEL_FILE_NAME = "model.pkl"

TARGET_COLUMN = "TotalCharges"
PREPROCESSING_OBJECT_FILE_NAME = "preprocess.pkl"

FILE_NAME = str = "telco_churn.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join('config', "schmea.yaml")


AWS_ACCESS_KEY_ID_ENV = 'AWS_ACCESS_KEY_ID_ENV'
AWS_SECRET_ACCESS_KEY = 'AWS_SECRET_ACCESS_KEY'
REGION_NAME = "us-east-1"

"""
Data Ingestion related constants with DATA_INGESTION VAR NAME
"""
DATA_INGENSTION_COLLECTION_NAME: str = "telco_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 2.0

"""
Data Validation related constants with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str ="data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str ="drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str ="report.yaml"

"""
Data Transformation related constants with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

"""
MODEL TRAINER related constants with MODEL_TRAINER VAR NAME
"""
MODEL_TRAINER_DIR_NAME: str ="model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 6.0
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")

MODEL_EVALUTAION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "model-evaluation-bucket"
MODEL_PUSHER_S3_KEY = "model-registry"

APP_HOST = "0.0.0.0"
APP_PORT = 8080