"""
Module: telco_prediction.py

Purpose:
    Provides two classes for inference in the Telco Churn project:
      • TelcoData – wraps raw feature inputs into a structured format
        (dictionary or DataFrame) for model consumption.
      • TelcoClassifier – loads the trained Telco model from S3 and generates predictions.

Notes:
    - This module assumes preprocessing (encoding, scaling) matches the training pipeline.
    - The TelcoEstimator handles model retrieval from S3 and prediction logic.
"""

import sys
from pandas import DataFrame
from telco_churn.entity.config_entity import TelcoPredictorConfig
from telco_churn.entity.s3_estimator import TelcoEstimator
from telco_churn.exceptions import custom_exception
from telco_churn.logger import logging


class TelcoData:
    def __init__(self,
                 SeniorCitizen: int,
                 Dependents: str,
                 tenure: float,
                 MultipleLines: str,
                 InternetService: str,
                 OnlineSecurity: str,
                 TechSupport: str,
                 StreamingTV: str,
                 StreamingMovies: str,
                 Contract: str,
                 PaperlessBilling: str,
                 PaymentMethod: str,
                 MonthlyCharges: float,
                 TotalCharges: float):
        """
        Initialize TelcoData with raw feature values.

        Parameters
        ----------
        SeniorCitizen : int
            Indicates if the customer is a senior citizen (0 or 1).
        Dependents : str
            'Yes' or 'No' depending on dependent status.
        tenure : float
            Number of months the customer has stayed with the company.
        MultipleLines : str
            Indicates if the customer has multiple phone lines.
        InternetService : str
            Type of internet service (e.g., 'DSL', 'Fiber optic', 'No').
        OnlineSecurity, TechSupport, StreamingTV, StreamingMovies : str
            Customer service options ('Yes', 'No', 'No internet service').
        Contract : str
            Contract term ('Month-to-month', 'One year', 'Two year').
        PaperlessBilling : str
            'Yes' or 'No' for paperless billing.
        PaymentMethod : str
            Payment method used by the customer.
        MonthlyCharges, TotalCharges : float
            Monthly and total charges for the customer.
        """
        try:
            self.SeniorCitizen = SeniorCitizen
            self.Dependents = Dependents
            self.tenure = tenure
            self.MultipleLines = MultipleLines
            self.InternetService = InternetService
            self.OnlineSecurity = OnlineSecurity
            self.TechSupport = TechSupport
            self.StreamingTV = StreamingTV
            self.StreamingMovies = StreamingMovies
            self.Contract = Contract
            self.PaperlessBilling = PaperlessBilling
            self.PaymentMethod = PaymentMethod
            self.MonthlyCharges = MonthlyCharges
            self.TotalCharges = TotalCharges

            # Basic numeric validation
            for attr_name in ["tenure", "MonthlyCharges", "TotalCharges"]:
                val = getattr(self, attr_name)
                if not isinstance(val, (int, float)):
                    raise ValueError(f"{attr_name} must be numeric, got {type(val)}")

        except Exception as e:
            raise custom_exception(e, sys) from e

    def get_telco_input_data_frame(self) -> DataFrame:
        """
        Convert TelcoData instance into a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            A single-row DataFrame with column names matching model input schema.
        """
        try:
            telco_input_dict = self.get_telco_data_as_dict()
            return DataFrame(telco_input_dict)
        except Exception as e:
            raise custom_exception(e, sys) from e

    def get_telco_data_as_dict(self) -> dict:
        """
        Convert the TelcoData instance to a dictionary.

        Returns
        -------
        dict
            Dictionary with feature names as keys and single-element lists as values.
        """
        logging.info("Entered get_telco_data_as_dict method of TelcoData class.")
        try:
            input_data = {
                "SeniorCitizen": [self.SeniorCitizen],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges],
            }
            logging.info("Created Telco data dictionary successfully.")
            return input_data
        except Exception as e:
            raise custom_exception(e, sys) from e


class TelcoClassifier:
    def __init__(self, prediction_pipeline_config: TelcoPredictorConfig = None) -> None:
        """
        Initialize the TelcoClassifier used for churn prediction.

        Parameters
        ----------
        prediction_pipeline_config : TelcoPredictorConfig, optional
            Configuration with S3 bucket and model file path.
            If not provided, a default instance will be created.
        """
        try:
            if prediction_pipeline_config is None:
                prediction_pipeline_config = TelcoPredictorConfig()
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise custom_exception(e, sys) from e

    def predict(self, dataframe: DataFrame):
        """
        Run churn prediction using the Telco model stored in S3.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Feature data structured like the training dataset.

        Returns
        -------
        np.ndarray or list
            Model predictions.

        Raises
        ------
        custom_exception
            If model loading or prediction fails.
        """
        try:
            logging.info("Entered predict method of TelcoClassifier class.")
            model = TelcoEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result = model.predict(dataframe)
            logging.info("Prediction completed successfully.")
            return result
        except Exception as e:
            raise custom_exception(e, sys) from e