import boto3
from AIML_1013_Project1.configuration.aws_connection import S3Client
from io import StringIO
from typing import Union, List
import os, sys
from AIML_1013_Project1.logger import logging
from mypy_boto3_s3.service_resource import Bucket
from AIML_1013_Project1.exceptions import custom_exception
from botocore.exceptions import ClientError
from pandas import DataFrame, read_csv
import pickle

class SimpleStorageService:

    def __init__(self):

        s3_client = S3Client()
        self.s3_resource = s3_client.s3_resource
        self.s3_client = s3_client.s3_client

    def s3_key_path_available(self, bucket_name, s3_key) -> bool:

        try:
            bucket = self.get_bucket(bucket_name)
            # Collect all objects matching the provided prefix
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=s3_key)]
            # Presence of any matching object implies availability
            if len(file_objects) > 0:
                return True
            else:
                return False
        except Exception as e:
            raise custom_exception(e, sys)

    @staticmethod
    def read_object(object_name: str, decode: bool = True, make_readable: bool = False) -> Union[StringIO, str]:

        logging.info("Entered the read_object method of S3Operations class")

        try:
            # Choose raw or decoded read function based on `decode`
            func = (
                lambda: object_name.get()["Body"].read().decode()
                if decode is True
                else object_name.get()["Body"].read()
            )
            # Optionally convert to StringIO for downstream CSV/readers
            conv_func = lambda: StringIO(func()) if make_readable is True else func()
            logging.info("Exited the read_object method of S3Operations class")
            return conv_func()

        except Exception as e:
            raise custom_exception(e, sys) from e

    def get_bucket(self, bucket_name: str) -> Bucket:

        logging.info("Entered the get_bucket method of S3Operations class")

        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            logging.info("Exited the get_bucket method of S3Operations class")
            return bucket
        except Exception as e:
            raise custom_exception(e, sys) from e

    def get_file_object(self, filename: str, bucket_name: str) -> Union[List[object], object]:

        logging.info("Entered the get_file_object method of S3Operations class")

        try:
            bucket = self.get_bucket(bucket_name)
            # Collect all objects with matching prefix
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=filename)]
            # Return a single object if exactly one, else the full list
            func = lambda x: x[0] if len(x) == 1 else x
            file_objs = func(file_objects)
            logging.info("Exited the get_file_object method of S3Operations class")
            return file_objs

        except Exception as e:
            raise custom_exception(e, sys) from e

    def load_model(self, model_name: str, bucket_name: str, model_dir: str = None) -> object:

        logging.info("Entered the load_model method of S3Operations class")

        try:
            # Construct full S3 key from optional directory
            func = (
                lambda: model_name
                if model_dir is None
                else model_dir + "/" + model_name
            )
            model_file = func()
            # Retrieve and read the object, then unpickle
            file_object = self.get_file_object(model_file, bucket_name)
            model_obj = self.read_object(file_object, decode=False)
            model = pickle.loads(model_obj)
            logging.info("Exited the load_model method of S3Operations class")
            return model

        except Exception as e:
            raise custom_exception(e, sys) from e

    def create_folder(self, folder_name: str, bucket_name: str) -> None:

        logging.info("Entered the create_folder method of S3Operations class")

        try:
            # Check if the folder object exists; will raise if not found
            self.s3_resource.Object(bucket_name, folder_name).load()

        except ClientError as e:
            # If object does not exist, create a zero-byte object with trailing slash
            if e.response["Error"]["Code"] == "404":
                folder_obj = folder_name + "/"
                self.s3_client.put_object(Bucket=bucket_name, Key=folder_obj)
            else:
                # For other client errors, do nothing here (follows original logic)
                pass
            logging.info("Exited the create_folder method of S3Operations class")

    def upload_file(self, from_filename: str, to_filename: str, bucket_name: str, remove: bool = True):

        logging.info("Entered the upload_file method of S3Operations class")

        try:
            logging.info(
                f"Uploading {from_filename} file to {to_filename} file in {bucket_name} bucket"
            )

            # Perform the file upload using the low-level client via resource meta
            self.s3_resource.meta.client.upload_file(
                from_filename, bucket_name, to_filename
            )

            logging.info(
                f"Uploaded {from_filename} file to {to_filename} file in {bucket_name} bucket"
            )

            # Remove local file if requested
            if remove is True:
                os.remove(from_filename)
                logging.info(f"Remove is set to {remove}, deleted the file")
            else:
                logging.info(f"Remove is set to {remove}, not deleted the file")

            logging.info("Exited the upload_file method of S3Operations class")

        except Exception as e:
            raise custom_exception(e, sys) from e

    def upload_df_as_csv(self, data_frame: DataFrame, local_filename: str, bucket_filename: str, bucket_name: str,) -> None:

        logging.info("Entered the upload_df_as_csv method of S3Operations class")

        try:
            # Persist DataFrame locally as CSV
            data_frame.to_csv(local_filename, index=None, header=True)

            # Upload the saved CSV to S3 (may remove local copy depending on upload_file's `remove`)
            self.upload_file(local_filename, bucket_filename, bucket_name)

            logging.info("Exited the upload_df_as_csv method of S3Operations class")

        except Exception as e:
            raise custom_exception(e, sys) from e

    def get_df_from_object(self, object_: object) -> DataFrame:

        logging.info("Entered the get_df_from_object method of S3Operations class")

        try:
            content = self.read_object(object_, make_readable=True)
            df = read_csv(content, na_values="na")
            logging.info("Exited the get_df_from_object method of S3Operations class")
            return df
        except Exception as e:
            raise custom_exception(e, sys) from e

    def read_csv(self, filename: str, bucket_name: str) -> DataFrame:

        logging.info("Entered the read_csv method of S3Operations class")

        try:
            csv_obj = self.get_file_object(filename, bucket_name)
            df = self.get_df_from_object(csv_obj)
            logging.info("Exited the read_csv method of S3Operations class")
            return df
        except Exception as e:
            raise custom_exception(e, sys) from e