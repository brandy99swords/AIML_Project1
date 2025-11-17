from AIML_1013_Project1.configuration.mongo_db_connect import MongoDBClient
from AIML_1013_Project1.constants import DATABASE_NAME
from AIML_1013_Project1.exceptions import custom_exception
from AIML_1013_Project1.logger import logging

import pandas as pd
import sys 
from typing import Optional 
import numpy as np

class project1Data:

    def __init__(self):

        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
            logging.info("MongoDB client initialized successfully.")
        except Exception as e:
            raise custom_exception(e, sys)
    
    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str]) -> pd.DataFrame:

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
