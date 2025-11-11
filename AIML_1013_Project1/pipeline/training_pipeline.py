"""
Module: train_pipeline.py

Purpose
-------
Run the Telco Churn machine-learning pipeline end-to-end:
  1) Data Ingestion
  2) Data Validation
  3) Data Transformation
  4) Model Training
  5) Model Evaluation
  6) Model Pushing (conditional on acceptance)
"""

import sys
from AIML_1013_Project1.exceptions import custom_exception
from AIML_1013_Project1.logger import logging

from AIML_1013_Project1.components.data_ingestion import DataIngestion
from AIML_1013_Project1.components.data_validation import DataValidation
from AIML_1013_Project1.components.data_transformation import DataTransformation
from AIML_1013_Project1.components.model_trainer import ModelTrainer
from AIML_1013_Project1.components.model_evaluation import ModelEvaluation
from AIML_1013_Project1.components.model_pusher import ModelPusher

from AIML_1013_Project1.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
)

from AIML_1013_Project1.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
)


class TrainPipeline:
    """
    High-level orchestrator that wires together the core pipeline stages
    (ingestion → validation → transformation → training → evaluation → pushing).

    Attributes
    ----------
    data_ingestion_config : DataIngestionConfig
        Configuration for data sourcing and split locations.
    data_validation_config : DataValidationConfig
        Configuration for schema checks, drift checks, and validation outputs.
    data_transformation_config : DataTransformationConfig
        Configuration for feature engineering and preprocessing artifacts.
    model_trainer_config : ModelTrainerConfig
        Configuration for model search, training thresholds, and save paths.
    model_evaluation_config : ModelEvaluationConfig
        Configuration for evaluation criteria and production model lookup.
    model_pusher_config : ModelPusherConfig
        Configuration for deployment target (e.g., S3 bucket/key).
    """

    def __init__(self):
        """
        Initialize all stage configurations required by the pipeline.

        Notes
        -----
        Each configuration object encapsulates parameters, thresholds, and I/O paths
        that downstream components rely on. This constructor does not perform any I/O.
        """
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Start the data ingestion stage.

        Returns
        -------
        DataIngestionArtifact
            Artifact containing file paths to the ingested training and test datasets.

        Raises
        ------
        custom_exception
            If ingestion fails for any reason.
        """
        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            logging.info("Getting data from MongoDB")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from MongoDB")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise custom_exception(e, sys) from e

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """
        Start the data validation stage.

        Parameters
        ----------
        data_ingestion_artifact : DataIngestionArtifact
            Output from data ingestion, providing dataset paths to validate.

        Returns
        -------
        DataValidationArtifact
            Artifact capturing validation results, reports, and status flags.

        Raises
        ------
        custom_exception
            If validation fails for any reason.
        """
        logging.info("Entered the start_data_validation method of TrainPipeline class")
        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Performed the data validation operation")
            logging.info("Exited the start_data_validation method of TrainPipeline class")
            return data_validation_artifact
        except Exception as e:
            raise custom_exception(e, sys) from e

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        """
        Start the data transformation stage.

        Parameters
        ----------
        data_ingestion_artifact : DataIngestionArtifact
            Provides raw/train/test inputs needed for transformation.
        data_validation_artifact : DataValidationArtifact
            Provides schema and validation outputs to guide transformation.

        Returns
        -------
        DataTransformationArtifact
            Artifact with transformed arrays and the fitted preprocessing object path.

        Raises
        ------
        custom_exception
            If transformation fails for any reason.
        """
        try:
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
                data_validation_artifact=data_validation_artifact,
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise custom_exception(e, sys)

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        """
        Start the model training stage.

        Parameters
        ----------
        data_transformation_artifact : DataTransformationArtifact
            Supplies transformed arrays and the preprocessing artifact.

        Returns
        -------
        ModelTrainerArtifact
            Artifact containing the saved model path and classification metrics.

        Raises
        ------
        custom_exception
            If training fails for any reason.
        """
        try:
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise custom_exception(e, sys)

    def start_model_evaluation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        """
        Start the model evaluation stage.

        Parameters
        ----------
        data_ingestion_artifact : DataIngestionArtifact
            Provides the test dataset path for evaluation.
        model_trainer_artifact : ModelTrainerArtifact
            Provides the candidate model path and its training metrics.

        Returns
        -------
        ModelEvaluationArtifact
            Artifact indicating acceptance decision, score deltas, and relevant paths.

        Raises
        ------
        custom_exception
            If evaluation fails for any reason.
        """
        try:
            model_evaluation = ModelEvaluation(
                model_eval_config=self.model_evaluation_config,
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        except Exception as e:
            raise custom_exception(e, sys)

    def start_model_pusher(
        self, model_evaluation_artifact: ModelEvaluationArtifact
    ) -> ModelPusherArtifact:
        """
        Start the model pushing (deployment) stage.

        Parameters
        ----------
        model_evaluation_artifact : ModelEvaluationArtifact
            Must indicate that the candidate model is accepted and provide its path.

        Returns
        -------
        ModelPusherArtifact
            Artifact describing the destination (e.g., bucket/key) of the pushed model.

        Raises
        ------
        custom_exception
            If pushing fails for any reason.
        """
        try:
            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=self.model_pusher_config,
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except Exception as e:
            raise custom_exception(e, sys)

    def run_pipeline(self) -> None:
        """
        Execute the complete pipeline.

        Workflow
        --------
        Ingestion → Validation → Transformation → Training → Evaluation → (conditional) Pushing

        Behavior
        --------
        If the newly trained model is not accepted during evaluation, the method exits early.
        Otherwise, it proceeds to push the accepted model to the configured destination.

        Raises
        ------
        custom_exception
            If any stage fails with an unhandled exception.
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
            )
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )
            model_evaluation_artifact = self.start_model_evaluation(
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )

            if not model_evaluation_artifact.is_model_accepted:
                logging.info("Model not accepted.")
                return None

            _ = self.start_model_pusher(
                model_evaluation_artifact=model_evaluation_artifact
            )

        except Exception as e:
            raise custom_exception(e, sys)