"""
Module: model_pusher.py

Purpose:
    Upload the newly trained Telco Churn model to the configured production storage (S3),
    and create a ModelPusherArtifact summarizing the S3 upload details.

"""

import sys
from telco_churn.cloud_storage.aws_storage import SimpleStorageService
from telco_churn.exceptions import custom_exception
from telco_churn.logger import logging
from telco_churn.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from telco_churn.entity.config_entity import ModelPusherConfig
from telco_churn.entity.s3_estimator import TelcoEstimator


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        Initialize the ModelPusher with evaluation artifacts and configuration.

        Parameters
        ----------
        model_evaluation_artifact : ModelEvaluationArtifact
            Provides the path to the newly trained model ready for deployment.
        model_pusher_config : ModelPusherConfig
            Configuration object containing S3 bucket details and the destination key path.

        Notes
        -----
        - The SimpleStorageService instance (`self.s3`) is available if direct AWS operations
          are needed in future extensions, though not currently used.
        - `telco_estimator` wraps the logic for saving/loading models to/from S3.
        """
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.telco_estimator = TelcoEstimator(
            bucket_name=model_pusher_config.bucket_name,
            model_path=model_pusher_config.s3_model_key_path
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_pusher
        Description :   Upload the trained Telco Churn model to S3 and return
                        a ModelPusherArtifact documenting the deployment.

        Workflow
        --------
        1) Log entry into the process.
        2) Upload the trained model file (from evaluation artifact) to the configured S3 path.
        3) Create a ModelPusherArtifact summarizing the upload details.
        4) Log successful completion and return the artifact.

        Returns
        -------
        ModelPusherArtifact
            Object containing:
            - bucket_name : str
                The target S3 bucket name where the model was uploaded.
            - s3_model_path : str
                The S3 key or path for the deployed model file.

        Raises
        ------
        custom_exception
            Raised if any failure occurs during the upload process or artifact creation.
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            logging.info("Starting upload of the trained model file to the S3 bucket")

            # Upload the trained model from local path to S3 using the TelcoEstimator interface.
            self.telco_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)

            # Construct the artifact summarizing the model push.
            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path
            )

            logging.info("Model successfully uploaded to S3 storage")
            logging.info(f"Model pusher artifact created: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact

        except Exception as e:
            # Wrap and re-raise the exception for consistent project-level error handling.
            raise custom_exception(e, sys) from e