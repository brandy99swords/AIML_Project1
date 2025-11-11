"""
Data Validation Module for Telco Churn Project

This module defines the `DataValidation` class, which performs schema-based checks
and data drift detection for the Telco Churn pipeline. Responsibilities include:

- Loading a YAML schema that specifies expected columns and their types/groups.
- Verifying that incoming datasets match the expected number of columns.
- Verifying that all expected numerical and categorical columns exist.
- Detecting dataset drift between a reference (training) dataframe and a current (testing) dataframe
  using Evidently's Profile + DataDriftProfileSection.
- Emitting a `DataValidationArtifact` that summarizes validation/drift outcomes and paths to reports.


"""

import json
import sys

import pandas as pd
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

from pandas import DataFrame

from AIML_1013_Project1.exceptions import custom_exception
from AIML_1013_Project1.logger import logging
from AIML_1013_Project1.utils import read_yaml_file, write_yaml_file
from AIML_1013_Project1.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from AIML_1013_Project1.entity.config_entity import DataValidationConfig
from AIML_1013_Project1.constants import SCHEMA_FILE_PATH


class DataValidation:
    """
    Orchestrates dataset validation for the Telco Churn pipeline.

    This class:
    - Loads the schema configuration from `SCHEMA_FILE_PATH`.
    - Validates the number of columns and presence of required numerical/categorical columns.
    - Detects data drift between training and testing dataframes using Evidently.
    - Produces a `DataValidationArtifact` summarizing the validation outcome.

    Parameters
    ----------
    data_ingestion_artifact : DataIngestionArtifact
        Output reference of the data ingestion stage. Expected to provide file paths such as
        `trained_file_path` and `test_file_path`.
    data_validation_config : DataValidationConfig
        Configuration object for data validation (e.g., paths to write drift reports).

    Attributes
    ----------
    data_ingestion_artifact : DataIngestionArtifact
        Stored ingestion artifact used to locate the training/testing datasets.
    data_validation_config : DataValidationConfig
        Stored configuration used to control validation outputs (e.g., drift report path).
    _schema_config : dict
        Parsed YAML schema loaded from `SCHEMA_FILE_PATH`. Expected to include keys such as
        "columns", "numerical_columns", and "categorical_columns".

    Raises
    ------
    custom_exception
        Wraps and re-raises any underlying exceptions with project-specific context.
    """

    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        Initialize DataValidation by storing artifacts/config and loading the schema.

        Parameters
        ----------
        data_ingestion_artifact : DataIngestionArtifact
            Ingestion artifact providing dataset file paths for validation.
        data_validation_config : DataValidationConfig
            Validation configuration (e.g., drift report output path).

        Raises
        ------
        custom_exception
            If schema loading fails or any unexpected error occurs.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            # Load schema that specifies expected columns and groupings.
            self._schema_config =read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise custom_exception(e,sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Validate that the dataframe has the exact number of columns specified by the schema.

        Method Name
        -----------
        validate_number_of_columns

        Description
        -----------
        Compares the count of columns in the provided dataframe against the schema's
        "columns" specification. Logs the result and returns a boolean status.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            The dataframe to validate.

        Returns
        -------
        bool
            True if the column counts match; False otherwise.

        Raises
        ------
        custom_exception
            If validation fails due to unexpected errors (e.g., schema/key issues).
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise custom_exception(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Validate existence of all expected numerical and categorical columns.

        Method Name
        -----------
        is_column_exist

        Description
        -----------
        Ensures every column listed under "numerical_columns" and "categorical_columns"
        in the schema is present in the provided dataframe. Logs any missing columns.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe to validate.

        Returns
        -------
        bool
            True if all required columns are present; False otherwise.

        Raises
        ------
        custom_exception
            If an unexpected error occurs during validation.
        """
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            # Check required numerical columns
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")

            # Check required categorical columns
            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns)>0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            # Return True only if no required columns are missing
            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True
        except Exception as e:
            raise custom_exception(e, sys) from e

    @staticmethod
    def read_data(file_path) -> DataFrame:
        """
        Read a CSV file from disk into a pandas DataFrame.

        Parameters
        ----------
        file_path : str
            Path to the CSV file to load.

        Returns
        -------
        pandas.DataFrame
            Loaded dataframe.

        Raises
        ------
        custom_exception
            If the file cannot be read for any reason.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise custom_exception(e, sys)

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame, ) -> bool:
        """
        Detect dataset drift between a reference (training) and current (testing) dataframe.

        Method Name
        -----------
        detect_dataset_drift

        Description
        -----------
        Uses Evidently's `Profile` with `DataDriftProfileSection` to compute drift metrics.
        Writes the full drift report (JSON) to a YAML file path provided by the
        `data_validation_config`. Logs the number of drifted features and returns the
        dataset-level drift status.

        Parameters
        ----------
        reference_df : pandas.DataFrame
            The reference dataframe (typically training data).
        current_df : pandas.DataFrame
            The current dataframe to compare against the reference (typically testing data).

        Returns
        -------
        bool
            True if Evidently determines dataset-level drift; False otherwise.

        Raises
        ------
        custom_exception
            If Evidently computation fails or report writing encounters an error.
        """
        try:
            # Build and compute Evidently drift profile
            data_drift_profile = Profile(sections=[DataDriftProfileSection()])

            data_drift_profile.calculate(reference_df, current_df)

            # Extract JSON and persist via YAML writer for downstream consumption
            report = data_drift_profile.json()
            json_report = json.loads(report)

            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content=json_report)

            # Log summary metrics
            n_features = json_report["data_drift"]["data"]["metrics"]["n_features"]
            n_drifted_features = json_report["data_drift"]["data"]["metrics"]["n_drifted_features"]

            logging.info(f"{n_drifted_features}/{n_features} drift detected.")
            drift_status = json_report["data_drift"]["data"]["metrics"]["dataset_drift"]
            return drift_status
        except Exception as e:
            raise custom_exception(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Run all validation steps and produce a `DataValidationArtifact`.

        Method Name
        -----------
        initiate_data_validation

        Description
        -----------
        Pipeline:
        1) Load training and testing dataframes from ingestion artifact paths.
        2) Validate column counts for both dataframes.
        3) Validate presence of all required numerical and categorical columns.
        4) If the dataset passes schema checks, detect dataset drift.
        5) Build and return a `DataValidationArtifact` summarizing results and report paths.

        Returns
        -------
        DataValidationArtifact
            An artifact including:
            - `validation_status` (bool),
            - `message` (str) describing validation/drift outcome,
            - `drift_report_file_path` (str) pointing to the written drift report.

        Raises
        ------
        custom_exception
            If any step in validation or drift detection fails unexpectedly.
        """

        try:
            validation_error_msg = ""
            logging.info("Starting data validation")

            # Read training and testing datasets using paths from the ingestion artifact
            train_df, test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                                 DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))

            # Validate number of columns for training dataframe
            status = self.validate_number_of_columns(dataframe=train_df)
            logging.info(f"All required columns present in training dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."

            # Validate number of columns for testing dataframe
            status = self.validate_number_of_columns(dataframe=test_df)
            logging.info(f"All required columns present in testing dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."

            # Validate required columns for training dataframe
            status = self.is_column_exist(df=train_df)
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."

            # Validate required columns for testing dataframe
            status = self.is_column_exist(df=test_df)
            if not status:
                validation_error_msg += f"columns are missing in test dataframe."

            # Determine overall validation status based on accumulated messages
            validation_status = len(validation_error_msg) == 0

            # If schema checks pass, proceed to drift detection
            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                if drift_status:
                    logging.info(f"Drift detected.")
                    validation_error_msg = "Drift detected"
                else:
                    validation_error_msg = "Drift not detected"
            else:
                logging.info(f"Validation_error: {validation_error_msg}")

            # Build the DataValidationArtifact with status, message, and drift report path
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise custom_exception(e, sys) from e