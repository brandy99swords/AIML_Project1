import os 
import sys 
import pandas as pd 
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from AIML_1013_Project1.entity.config_entity import DataIngestionConfig
from AIML_1013_Project1.entity.artifact_entity import DataIngestionArtifact

from AIML_1013_Project1.exceptions import custom_exception
from AIML_1013_Project1.logger import logging
from AIML_1013_Project1.database_access.mongo_extract import project1Data


class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        """
        Parameters: data_ingestion_config: configuration for data ingestion

        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise custom_exception(e, sys)
    
    def export_data_into_feature_store(self) -> DataFrame:
        """
        Method Name: export_data_into_feature_store
        Description: This method exports data from mongodb to csv file

        Output: data is returned as an artifact of the data ingestion component 
        On Failure: write an exception log and then raise an exception
        """
        try: 
            logging.info(f"Exporting data from mongodb")
            project1_data = TelcoData()
            dataframe = project1_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            logging.info(f"Shape of the dataframe: {dataframe.shape}")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving exported data into feature store file: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index = False, header = True)
            return dataframe 
        
        except Exception as e: 
            raise custom_exception(e, sys)
        
    def split_data_as_train_test(self, dataframe: DataFrame) -> None: 
        """
        Method Name: split_data_as_train_test
        Description: This method splits the dataframe into train/test based on a split ratio

        Output: Folder is created in s3 bucket 
        On Failure: Write an exception log and raise an exception
        """
        logging.info("Entered the split_data_as_train_test method of Data_Ingestion Class")
        
        try: 
            if dataframe.empty:
                raise ValueError("The dataframe is empty. Please check the data loading process.")
            
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train test split on the dataframe")
            logging.info("Exited the split_as_train_test_data method of Data_Ingestion Class")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting train and test file path.")
            train_set.to_csv(self.data_ingestion_config.testing_file_path, index = False, header = True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index = False, header = True)

            logging.info(f"Exported train and test file path.")
        
        except Exception as e:
            raise custom_exception(e, sys) from e
        

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name: initiate_data_ingestion
        Description: This method intiates the data ingestion components of training pipeline

        Output: train set and test set are returned as artifacts of data ingestion components 
        On Failure: Write an exception log and raise an exception
        """
        logging.info("Entered the initiate_data_ingestion method of the Data_Ingestion class")

        try: 
            dataframe = self.export_data_into_feature_store()
            logging.info("Got the data from mongo")

            if dataframe.empty:
                raise ValueError("The dataframe fetched from Mongo is empty. Please check the data loading process")
            
            self.split_data_as_train_test(dataframe)
            logging.info("Performed train test split on the dataset")

            logging.info("Exited initiate_data_ingestion method")

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise custom_exception(e, sys) from e
        
            