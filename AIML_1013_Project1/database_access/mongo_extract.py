from telco_churn.configuration.mongo_db_connect import MongoDBClient
from telco_churn.constants import DATABASE_NAME
from telco_churn.exceptions import custom_exception
from telco_churn.logger import logging

import pandas as pd
import sys 
from typing import Optional 
import numpy as np

class TelcoData:
    """
    Class Name: TelcoData
    Description: This class is responsible for extracting data from MongoDB and converting it into a DataFrame.
    
    Output: DataFrame containing the data from the MongoDB collection
    On Failure: Raises Exception with a custom error message
    """

    def __init__(self):
        """
        Initialize the MongoDB client and set the db names.
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
            logging.info("MongoDB client initialized successfully.")
        except Exception as e:
            raise custom_exception(e, sys)
    
    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str]) -> pd.DataFrame:
        """
        Extract data from MongoDB collection and convert it into a DataFrame.
        
        Args:
            collection_name (str): The name of the MongoDB collection to extract data from.
            database_name (Optional[str]): The name of the MongoDB database. If None, use the default database.
        
        Returns:
            pd.DataFrame: A DataFrame containing the extracted data.
        """
        try:
            logging.info(f'Exporting data from collection: {collection_name}')

            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else: 
                collection = self.mongo_client.client[database_name][collection_name]

            data = list(collection.find())
            logging.info(f"Number of records extracted: {len(data)}")

            df = pd.DataFrame(data)

            if df.empty:
                logging.warning("No data found in the collection. {collection_name} is empty.")
                return df
            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis=1)
                logging.info("Dropped the _id column from the DataFrame.")

            df.replace({"na": np.nan}, inplace=True)
            logging.info(f"Dataframe shape after processing {df.shape}")
            return df
        except Exception as e:
            raise custom_exception(e, sys)
