import sys 

import numpy as np 
import pandas as pd 
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from telco_churn.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from telco_churn.entity.config_entity import DataTransformationConfig
from telco_churn.entity.articfact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact

from telco_churn.exceptions import custom_exception
from telco_churn.logger import logging
from telco_churn.utils import save_object, save_numpy_array_data, read_yaml_file,drop_columns
from telco_churn.entity.estimator import TargetValueMapping



class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        Description: this is used to initialize the DataTransformation class.
        Parameters: data_ingestion_artifact: this contains the path of the data 
                    data_transformation_config: this contains the path of the schema file 
                    data_validation_artifact: contains path to the schema file 
        """

        try:
            self.data_ingestion_artifact =  data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self.schema_config = read_yaml_file(file_path = SCHEMA_FILE_PATH)
        
        except Exception as e:
            raise custom_exception(e, sys)
        
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Description: this function is used to read data from the given file path
        Parameters: file_path: str. this is the file_path from which we have read the data 
        Returns: pd.DataFrame  

        """
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise custom_exception(e, sys)
        
    def get_data_transformer_object(self) -> Pipeline:
        """
        Description: The method creates and returns a data transformer object for the data 

        Returns: data transformer object is created and returned 
        On Failure: Write an exception log and then raise an expecption
        """
        logging.info("Entered the data_transformer_object method of DataTransformation class")

        try:
            logging.info("Got numeric cols from schema config")
            numerical_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()

            logging.info("Initialized StandardScaler, OneHotEncoder, OrdinalEncoder")

            oh_columns = self.schema_config['oh_columns']
            or_columns = self.schema_config['or_columns']
            num_features = self.schema_config['num_features']

            logging.info(f'Initialized Preprocessing')

            preprocessor = ColumnTransformer(
                transformers = [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("OrdinalEncoder", ordinal_encoder, or_columns),
                    ("StandardScaler", numerical_transformer, num_features)

                ]
            )

            logging.info("Preprocessing object created")
            return preprocessor
        except Exception as e:
            raise custom_exception(e, sys)
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Description: This method is used to initiate the data transformation process. 
        Output: DataTransformationArtifact 
        On Failure: Raise an exception and write it in the log 
        """
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataIngestionArtifact.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                input_feature_train_df = train_df.drop(columns = [TARGET_COLUMN], axis = 1)
                input_feature_test_df = test_df.drop(columns = [TARGET_COLUMN], axis = 1)

                logging.info(f"Dropping columns {drop_columns} from the train and test dataframes")

                target_feature_train_df = target_feature_train_df.replace(
                    TargetValueMapping._asdict()
                )

                input_feature_test_df = test_df.

                logging.info("Got train features and test features of the dataset")

                logging.info("Applying preprocessing object on the data ")

                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

                logging.info("Used the preprocessor object to fit transform the train features")

                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                logging.info("Used the preprocessor object to transform the test features")

                logging.info("Applying SMOTEENN on Training dataset")

                smt = SMOTEENN(sampling_strategy = "minority")

                input_feature_train_final, target_feature_train_final = smt.fit_resample(
                    input_feature_train_arr, target_feature_train_df
                )

                 input_feature_test_final, target_feature_test_final = smt.fit_resample(
                    input_feature_test_arr, target_feature_test_df
                )

                
                logging.info("Applied SMOTEENN on training dataset")

                logging.info("Applying SMOTEENN on testing dataset")



