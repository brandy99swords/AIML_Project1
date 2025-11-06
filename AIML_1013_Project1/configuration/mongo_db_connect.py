import sys 
import os 
import pymongo

from AIML_1013_Project1.exceptions import custom_exception
from AIML_1013_Project1.logger import logging

from AIML_1013_Project1.constants import DATABASE_NAME, MONGODBURL

class MongoDBClient:
    """
    Class Name: MondoDBClient
    Description: This class is responsible for creating a MongoDB client and connecting to the database

    Output: Connection to the MongoDB database
    On Fialure: Raises Exception with a custom error message
    """
    client = None

    def __init__(self, database_name: str = DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODBURL)
                if mongo_db_url is None:
                    raise Exception("MONGODBURL environment variable not set")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url)
                logging.info(f"MongoDB client created with URL: {mongo_db_url}")
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info(f"Connected to database: {self.database_name}")
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            raise custom_exception(e, sys) from e
                  