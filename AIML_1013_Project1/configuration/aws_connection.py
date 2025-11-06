"""
S3 Client Module for Telco Churn Project

This module defines the `S3Client` class, which establishes and manages a connection to
Amazon S3 using credentials stored in environment variables. It leverages `boto3` to
create both a client and a resource interface for interacting with S3 buckets.

Responsibilities
----------------
- Retrieve AWS credentials (Access Key ID and Secret Access Key) from environment variables.
- Initialize a global `boto3` S3 client and resource if they are not already set.
- Provide these handles to subclasses or other components that require S3 access.

Notes
-----
- The credentials are fetched from environment variables defined in the project constants.
- This design ensures that S3 connection objects (`s3_client` and `s3_resource`)
  are shared across all instances of the class to minimize repeated connection overhead.
"""

import boto3
import os
from AIML_1013_Project1.constants import AWS_SECRET_ACCESS_KEY, AWS_ACCESS_KEY_ID_ENV, REGION_NAME


class S3Client:
    """
    A helper class for creating and managing AWS S3 connections using `boto3`.

    Attributes
    ----------
    s3_client : boto3.client
        Class-level reference to the S3 client, initialized once per runtime.
    s3_resource : boto3.resource
        Class-level reference to the S3 resource, initialized once per runtime.

    Methods
    -------
    __init__(region_name=REGION_NAME)
        Initializes the S3 client and resource using environment-based AWS credentials.
    """

    # Class-level (shared) attributes to ensure single S3 session across instances.
    s3_client = None
    s3_resource = None

    def __init__(self, region_name=REGION_NAME):
        """
        Initialize the S3 client and resource.

        Description
        -----------
        This constructor retrieves AWS credentials from environment variables defined in
        the project constants (`AWS_ACCESS_KEY_ID_ENV` and `AWS_SECRET_ACCESS_KEY`).
        If credentials are missing, an exception is raised.
        When valid credentials are found, it creates and stores the boto3 S3 client and
        resource objects at the class level so that subsequent instances reuse them.

        Parameters
        ----------
        region_name : str, optional
            The AWS region to connect to (default: value from `REGION_NAME` constant).

        Raises
        ------
        Exception
            If AWS access key ID or secret access key environment variables are not set.
        """

        # Only initialize S3 connections once for the entire class
        if S3Client.s3_resource == None or S3Client.s3_client == None:

            # Fetch AWS credentials from environment variables
            __access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV,)
            __secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY,)

            # Validate that required environment variables are set
            if __access_key_id is None:
                raise Exception(f"Environment variable: {AWS_ACCESS_KEY_ID_ENV} is not not set.")
            if __secret_access_key is None:
                raise Exception(f"Environment variable: {AWS_SECRET_ACCESS_KEY} is not set.")

            # Create a new boto3 S3 resource interface
            S3Client.s3_resource = boto3.resource(
                's3',
                aws_access_key_id=__access_key_id,
                aws_secret_access_key=__secret_access_key,
                region_name=region_name
            )

            # Create a new boto3 S3 client interface
            S3Client.s3_client = boto3.client(
                's3',
                aws_access_key_id=__access_key_id,
                aws_secret_access_key=__secret_access_key,
                region_name=region_name
            )

        # Assign class-level client/resource to the instance
        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client