"""
Simple Storage Service (S3) Operations Module

This module defines the `SimpleStorageService` class, a convenience wrapper around
boto3 S3 interactions for the Telco Churn project. It relies on an already-configured
`S3Client` (which provides shared `s3_resource` and `s3_client`) and exposes common
operations used throughout the pipeline:

- Check if a given S3 key (prefix) exists in a bucket.
- Retrieve bucket and object handles.
- Read raw objects (bytes or decoded text) from S3.
- Deserialize a pickled model object stored in S3.
- Create a "folder" (zero-byte object with trailing slash) in S3.
- Upload local files to S3 (with optional local file removal).
- Save a pandas DataFrame locally as CSV and upload it to S3.
- Read a CSV in S3 directly into a pandas DataFrame.

Notes
-----
- This module preserves existing function names, signatures, and logic exactly as provided.
- Exceptions are wrapped and re-raised using the project-specific `custom_exception`.
"""

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
    """
    A thin wrapper over boto3 S3 resource/client for common S3 operations.

    Initialization relies on `S3Client` to supply shared, preconfigured
    `s3_resource` and `s3_client` instances (credentials and region handled
    by `S3Client`).

    Attributes
    ----------
    s3_resource : boto3.resources.factory.s3.ServiceResource
        The high-level S3 resource interface for object-based operations.
    s3_client : botocore.client.S3
        The low-level S3 client interface for explicit API calls.
    """

    def __init__(self):
        """
        Initialize the service by constructing an `S3Client` and capturing its
        `s3_resource` and `s3_client` handles for subsequent operations.
        """
        s3_client = S3Client()
        self.s3_resource = s3_client.s3_resource
        self.s3_client = s3_client.s3_client

    def s3_key_path_available(self, bucket_name, s3_key) -> bool:
        """
        Check whether any object(s) exist for a given key prefix in the bucket.

        Parameters
        ----------
        bucket_name : str
            Name of the S3 bucket to search.
        s3_key : str
            Key or key prefix to filter objects on.

        Returns
        -------
        bool
            True if one or more objects exist under the prefix; otherwise False.

        Raises
        ------
        custom_exception
            Wrapped exception if the bucket query fails.
        """
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
        """
        Read the content of an S3 object (as bytes or decoded text), optionally
        wrapping the result in a StringIO buffer for pandas compatibility.

        Method Name
        -----------
        read_object

        Description
        -----------
        Calls `.get()` on the provided object handle and returns either:
        - Decoded text (str) if `decode=True`;
        - Raw bytes if `decode=False`;
        Optionally wraps the content in `StringIO` if `make_readable=True`.

        Parameters
        ----------
        object_name : object
            An S3 object handle (e.g., result of `Bucket.objects.filter(...)[0]`).
        decode : bool, default True
            If True, decode bytes to text via `.decode()`.
        make_readable : bool, default False
            If True, wrap the returned content in a `StringIO` buffer.

        Returns
        -------
        Union[StringIO, str]
            The object content as StringIO (if requested) or as text/bytes.

        Raises
        ------
        custom_exception
            Wrapped exception if the read fails.

        Version
        -------
        1.2

        Revisions
        ---------
        moved setup to cloud
        """
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
        """
        Get a `Bucket` resource handle for the given bucket name.

        Method Name
        -----------
        get_bucket

        Description
        -----------
        Obtains a handle to the S3 bucket so that callers may filter/list objects
        or perform object-level operations via the resource API.

        Parameters
        ----------
        bucket_name : str
            Name of the S3 bucket.

        Returns
        -------
        Bucket
            Boto3 bucket resource object.

        Raises
        ------
        custom_exception
            Wrapped exception if bucket access fails.

        Version
        -------
        1.2

        Revisions
        ---------
        moved setup to cloud
        """
        logging.info("Entered the get_bucket method of S3Operations class")

        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            logging.info("Exited the get_bucket method of S3Operations class")
            return bucket
        except Exception as e:
            raise custom_exception(e, sys) from e

    def get_file_object(self, filename: str, bucket_name: str) -> Union[List[object], object]:
        """
        Retrieve one or more S3 object handles that match the provided key prefix.

        Method Name
        -----------
        get_file_object

        Description
        -----------
        Filters objects in the specified bucket by the given prefix (`filename`).
        If exactly one object matches, that object is returned; otherwise a list
        of matching objects is returned.

        Parameters
        ----------
        filename : str
            Key or key prefix to filter objects on.
        bucket_name : str
            Name of the S3 bucket.

        Returns
        -------
        Union[List[object], object]
            A single object if exactly one match; otherwise a list of objects.

        Raises
        ------
        custom_exception
            Wrapped exception if listing/filtering fails.

        Version
        -------
        1.2

        Revisions
        ---------
        moved setup to cloud
        """
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
        """
        Load a serialized (pickled) model from S3 and deserialize it into memory.

        Method Name
        -----------
        load_model

        Description
        -----------
        Builds the object key from `model_dir` (if provided) and `model_name`,
        fetches the corresponding S3 object, reads its raw bytes, and unpickles
        the model.

        Parameters
        ----------
        model_name : str
            File name (S3 key tail) of the serialized model.
        bucket_name : str
            Name of the bucket that contains the model file.
        model_dir : str, optional
            Optional directory/prefix in which the model file resides.

        Returns
        -------
        object
            The deserialized model instance.

        Raises
        ------
        custom_exception
            Wrapped exception if object retrieval or deserialization fails.

        Version
        -------
        1.2

        Revisions
        ---------
        moved setup to cloud
        """
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
        """
        Create a "folder" (a zero-byte object with a trailing slash) in the bucket.

        Method Name
        -----------
        create_folder

        Description
        -----------
        Attempts to `load()` the object representing the folder. If it does not
        exist (404), it creates the placeholder "folder" object (i.e., `folder/`).

        Parameters
        ----------
        folder_name : str
            The "folder" name (prefix) to create.
        bucket_name : str
            Name of the target S3 bucket.

        Returns
        -------
        None

        Raises
        ------
        ClientError
            Propagated if the error is not a 404 (object not found).

        Version
        -------
        1.2

        Revisions
        ---------
        moved setup to cloud
        """
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
        """
        Upload a local file to S3 and (optionally) remove the local file afterward.

        Method Name
        -----------
        upload_file

        Description
        -----------
        Uses the resource's meta client to upload the local file to the specified
        bucket/key. Logs the operation and optionally deletes the local file if
        `remove=True`.

        Parameters
        ----------
        from_filename : str
            Local path of the source file to upload.
        to_filename : str
            Target S3 key under which the file will be stored.
        bucket_name : str
            Name of the destination S3 bucket.
        remove : bool, default True
            If True, remove the local file after successful upload.

        Returns
        -------
        None

        Raises
        ------
        custom_exception
            Wrapped exception if the upload or removal fails.

        Version
        -------
        1.2

        Revisions
        ---------
        moved setup to cloud
        """
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
        """
        Save a DataFrame as a local CSV file and upload it to S3.

        Method Name
        -----------
        upload_df_as_csv

        Description
        -----------
        Writes `data_frame` to a local CSV named `local_filename` (with header, no index)
        and calls `upload_file` to send it to the given `bucket_filename` in `bucket_name`.

        Parameters
        ----------
        data_frame : pandas.DataFrame
            The DataFrame to persist and upload.
        local_filename : str
            Local path for the temporary CSV file.
        bucket_filename : str
            Destination S3 key for the uploaded CSV.
        bucket_name : str
            Target S3 bucket name.

        Returns
        -------
        None

        Raises
        ------
        custom_exception
            Wrapped exception if saving or uploading fails.

        Version
        -------
        1.2

        Revisions
        ---------
        moved setup to cloud
        """
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
        """
        Read an S3 object and parse it into a pandas DataFrame (CSV format assumed).

        Method Name
        -----------
        get_df_from_object

        Description
        -----------
        Reads the object's body as text (wrapped in `StringIO`) and parses it using
        `pandas.read_csv`, treating string `"na"` values as NaN.

        Parameters
        ----------
        object_ : object
            An S3 object handle obtained from filtering/listing on a bucket.

        Returns
        -------
        pandas.DataFrame
            Parsed DataFrame from the CSV content.

        Raises
        ------
        custom_exception
            Wrapped exception if reading or parsing fails.

        Version
        -------
        1.2

        Revisions
        ---------
        moved setup to cloud
        """
        logging.info("Entered the get_df_from_object method of S3Operations class")

        try:
            content = self.read_object(object_, make_readable=True)
            df = read_csv(content, na_values="na")
            logging.info("Exited the get_df_from_object method of S3Operations class")
            return df
        except Exception as e:
            raise custom_exception(e, sys) from e

    def read_csv(self, filename: str, bucket_name: str) -> DataFrame:
        """
        Load a CSV in S3 directly into a pandas DataFrame.

        Method Name
        -----------
        get_df_from_object

        Description
        -----------
        Retrieves the S3 object for `filename` within `bucket_name` and parses it
        into a DataFrame via `get_df_from_object`.

        Parameters
        ----------
        filename : str
            S3 key (or key prefix) of the CSV file to read.
        bucket_name : str
            Name of the S3 bucket.

        Returns
        -------
        pandas.DataFrame
            DataFrame parsed from the CSV content.

        Raises
        ------
        custom_exception
            Wrapped exception if retrieval or parsing fails.

        Version
        -------
        1.2

        Revisions
        ---------
        moved setup to cloud
        """
        logging.info("Entered the read_csv method of S3Operations class")

        try:
            csv_obj = self.get_file_object(filename, bucket_name)
            df = self.get_df_from_object(csv_obj)
            logging.info("Exited the read_csv method of S3Operations class")
            return df
        except Exception as e:
            raise custom_exception(e, sys) from e