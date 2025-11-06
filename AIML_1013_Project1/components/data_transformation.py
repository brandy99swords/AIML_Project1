"""
Data Transformation Module for Telco Churn Project

This module defines the `DataTransformation` class responsible for preparing
raw input data for downstream machine learning tasks. It centralizes routines
for:
- Reading datasets
- Building a scikit-learn `ColumnTransformer` that applies:
  * One-hot encoding for nominal categorical features
  * Ordinal encoding for ordered categorical features
  * Standardization for numeric features
- Performing class balancing using SMOTEENN
- Persisting transformed artifacts and NumPy arrays
 
"""

import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from AIML_1013_Project1.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from AIML_1013_Project1.config_entity import DataTransformationConfig
from AIML_1013_Project1.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
)

from AIML_1013_Project1.exceptions import custom_exception
from AIML_1013_Project1.logger import logging
from AIML_1013_Project1.utils import (
    save_object,
    save_numpy_array_data,
    read_yaml_file,
    drop_columns,
)
from AIML_1013_Project1.entity.estimator import TargetValueMapping


class DataTransformation:
    """
    Encapsulates data transformation logic for the Telco Churn pipeline.

    This class is responsible for:
    - Loading schema configuration
    - Building a preprocessing `ColumnTransformer`
    - Reading training/testing CSV data
    - Transforming features (encoding/scaling)
    - Addressing class imbalance via SMOTEENN
    - Saving the fitted preprocessor and transformed arrays to disk

    Parameters
    ----------
    data_ingestion_artifact : DataIngestionArtifact
        Artifact containing paths to the ingested train/test datasets.
    data_transformation_config : DataTransformationConfig
        Configuration containing output paths for transformed artifacts.
    data_validation_artifact : DataValidationArtifact
        Artifact conveying schema/validation information and status.

    Attributes
    ----------
    data_ingestion_artifact : DataIngestionArtifact
        Stored reference to ingestion artifact with file paths.
    data_transformation_config : DataTransformationConfig
        Stored reference to transformation configuration with output paths.
    data_validation_artifact : DataValidationArtifact
        Stored reference to validation artifact (e.g., status, messages).
    schema_config : dict
        Parsed YAML schema loaded from `SCHEMA_FILE_PATH`.

    Raises
    ------
    custom_exception
        If initialization fails (e.g., schema cannot be read).
    """

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        """
        Description
        -----------
        Initialize the DataTransformation class by storing artifacts and
        loading the schema configuration from YAML.

        Notes
        -----
        - Reads `SCHEMA_FILE_PATH` to populate `self.schema_config`.
        - Keeps naming/behavior identical to provided code.

        Raises
        ------
        custom_exception
            Wraps any underlying exception with project-specific context.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            # Load schema YAML containing column groups and other directives.
            self.schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise custom_exception(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Read a CSV file into a pandas DataFrame.

        Parameters
        ----------
        file_path : str
            Absolute or relative path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.

        Raises
        ------
        custom_exception
            If the file cannot be read (e.g., missing, malformed).
        """
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise custom_exception(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Build and return the preprocessing transformer.

        Description
        -----------
        Constructs a `ColumnTransformer` composed of:
        - OneHotEncoder for nominal categorical columns (from schema: `oh_columns`)
        - OrdinalEncoder for ordinal categorical columns (from schema: `or_columns`)
        - StandardScaler for numeric features (from schema: `num_features`)

        The transformer is intended to be fitted on the training data and reused
        for inference/transforming test data.

        Returns
        -------
        Pipeline or ColumnTransformer
            A preprocessing transformer that can be fit/transformed on data frames.

        Raises
        ------
        custom_exception
            If the transformer construction fails for any reason.
        """
        logging.info("Entered the get_data_transformer_object method of DataTransformation class")

        try:
            # Instantiate transformers for different feature types.
            logging.info("Got numerical cols from schema config")
            numerical_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()

            logging.info("Inintialized StandardScaler, OneHotEncoder and OrdinalEncoder")

            # Column groups expected in schema YAML.
            oh_columns = self.schema_config["oh_columns"]
            or_columns = self.schema_config["or_columns"]
            num_features = self.schema_config["num_features"]

            logging.info(f"Initialized Preprocessing")

            # Create composite preprocessor across column subsets.
            preprocessor = ColumnTransformer(
                transformers=[
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("OrdinalEncoder", ordinal_encoder, or_columns),
                    ("StandardScaler", numerical_transformer, num_features),
                ]
            )
            logging.info("Preprocessing object created")
            return preprocessor

        except Exception as e:
            raise custom_exception(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Execute the full transformation workflow and persist artifacts.

        Workflow
        --------
        1) Check validation status from `data_valiation_artifact` (as named).
        2) Build preprocessing transformer via `get_data_transformer_object()`.
        3) Read train and test CSVs using `DataTransformation.read_data`.
        4) Split features/target using `TARGET_COLUMN`.
        5) Optionally drop columns as specified by schema (`drop_columns`).
        6) Encode the target labels using `TargetValueMapping`.
        7) Fit/transform training features; transform test features.
        8) Apply SMOTEENN to address class imbalance on train and test.
        9) Concatenate features and target back into NumPy arrays.
        10) Save preprocessor and transformed arrays via project utilities.
        11) Return a `DataTransformationArtifact` with output file paths.

        Returns
        -------
        DataTransformationArtifact
            Paths to the persisted preprocessor, transformed train, and test arrays.

        Raises
        ------
        custom_exception
            Wraps any exceptions occurring during transformation.

        
        """
        try:
            # Validation gate: proceed only if validation status is True.
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")

                # Build preprocessing transformer.
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                # Read the train and test datasets.
                # NOTE: Attribute name `trained_file_path` is preserved as given.
                train_df = DataTransformation.read_data(
                    file_path=self.data_ingestion_artifact.trained_file_path
                )
                test_df = DataTransformation.read_data(
                    file_path=self.data_ingestion_artifact.test_file_path
                )

                # Split into input features and target for training set.
                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                logging.info("Got train features and test features of Training dataset")

           
                drop_cols = self.schema_config["drop_columns"] #realized had the issue of same name
                input_feature_train_df = drop_columns(df=input_feature_train_df, cols=drop_cols)#change here

                logging.info(f"Dropping columns {drop_columns} from the train and test dataframes")

                # Map target labels to numeric values using TargetValueMapping.
                target_feature_train_df = target_feature_train_df.replace(
                    TargetValueMapping()._asdict()
                )

                # Prepare test feature/target splits.
                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]

           
                input_feature_test_df = drop_columns(df=input_feature_test_df, cols=drop_cols)#change here

                logging.info("drop the columns in drop_cols of Test dataset")

                target_feature_test_df = target_feature_test_df.replace(
                    TargetValueMapping()._asdict()
                )

                logging.info("Got train features and test features of Testing dataset")

                logging.info(
                    "Applying preprocessing object on training dataframe and testing dataframe"
                )

                # Fit the preprocessor on training features and transform them.
                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                logging.info("Used the preprocessor object to fit transform the train features")

                # Transform test features using the already-fitted preprocessor.
                input_feature_test_arr = preprocessor.transform(input_feature_test_df)
                logging.info("Used the preprocessor object to transform the test features")

                # Address class imbalance with SMOTEENN on training set.
                logging.info("Applying SMOTEENN on Training dataset")
                smt = SMOTEENN(sampling_strategy="minority")

                input_feature_train_final, target_feature_train_final = smt.fit_resample(
                    input_feature_train_arr, target_feature_train_df
                )
                logging.info("Applied SMOTEENN on training dataset")

                # Apply SMOTEENN on the test set as well (per provided code).
                logging.info("Applying SMOTEENN on testing dataset")
                input_feature_test_final, target_feature_test_final = smt.fit_resample(
                    input_feature_test_arr, target_feature_test_df
                )
                logging.info("Applied SMOTEENN on testing dataset")

                logging.info("Created train array and test array")

                # Concatenate features with target column for both splits.
                train_arr = np.c_[
                    input_feature_train_final, np.array(target_feature_train_final)
                ]
                test_arr = np.c_[
                    input_feature_test_final, np.array(target_feature_test_final)
                ]

                # Persist the fitted preprocessor and transformed arrays.
                save_object(
                    self.data_transformation_config.transformed_object_file_path, preprocessor
                )
                save_numpy_array_data(
                    self.data_transformation_config.transformed_train_file_path, array=train_arr
                )
                save_numpy_array_data(
                    self.data_transformation_config.transformed_test_file_path, array=test_arr
                )

                logging.info("Saved the preprocessor object")
                logging.info(
                    "Exited initiate_data_transformation method of Data_Transformation class"
                )

                # Package artifact with references to saved file paths.
                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                )
                return data_transformation_artifact

            else:
                # If validation failed, surface its message.
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            # Propagate wrapped exception with context.
            raise custom_exception(e, sys) from e