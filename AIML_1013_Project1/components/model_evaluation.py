"""
Module: model_evaluation.py

Purpose:
    Compare a newly trained model against the current production model (if one exists in S3)
    using F1 score on the held-out test data, decide whether to accept the new model, and
    produce a ModelEvaluationArtifact summarizing the decision and related paths/scores.


"""

from AIML_1013_Project1.entity.config_entity import ModelEvaluationConfig
from AIML_1013_Project1.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from AIML_1013_Project1.exceptions import custom_exception
from AIML_1013_Project1.constants import TARGET_COLUMN
from AIML_1013_Project1.logger import logging
import sys
import pandas as pd
from typing import Optional
from AIML_1013_Project1.entity.s3_estimator import project1Estimator
from dataclasses import dataclass
from AIML_1013_Project1.entity.estimator import project1Model
from AIML_1013_Project1.entity.estimator import TargetValueMapping


@dataclass
class EvaluateModelResponse:
    """
    Container for model comparison results.

    Attributes
    ----------
    trained_model_f1_score : float
        F1 score of the newly trained model evaluated on the test set.
    best_model_f1_score : float
        F1 score of the current production model on the same test set. May be None if no model exists.
    is_model_accepted : bool
        True if the trained model's F1 score strictly exceeds the production model's F1 score (or 0 if none).
    difference : float
        trained_model_f1_score - (best_model_f1_score or 0). Positive implies an improvement.
    """
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    """
    Orchestrates evaluation of a newly trained model against the production model.

    Parameters
    ----------
    model_eval_config : ModelEvaluationConfig
        Configuration containing S3 bucket and model key path for production model lookup.
    data_ingestion_artifact : DataIngestionArtifact
        Artifact containing paths to the ingested datasets (e.g., test_file_path).
    model_trainer_artifact : ModelTrainerArtifact
        Artifact containing the trained model path and its computed metric(s).

    Raises
    ------
    custom_exception
        Wraps any initialization error for standardized upstream handling.
    """

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config      # Store evaluation configuration (e.g., S3 info).
            self.data_ingestion_artifact = data_ingestion_artifact  # Access to test data path.
            self.model_trainer_artifact = model_trainer_artifact    # Access to trained model metrics/paths.
        except Exception as e:
            raise custom_exception(e, sys) from e

    def get_best_model(self) -> Optional[project1Estimator]:
        """
        Retrieve a handle to the current production model stored in S3, if present.

        Returns
        -------
        Optional[TelcoEstimator]
            TelcoEstimator instance wrapping the production model if present; otherwise None.

        Raises
        ------
        custom_exception
            If any error occurs during S3 lookup/initialization.
        """
        try:
            bucket_name = self.model_eval_config.bucket_name          # S3 bucket containing the model artifact(s).
            model_path = self.model_eval_config.s3_model_key_path     # S3 key/path to the production model.
            project1_estimator = project1Estimator(bucket_name=bucket_name,
                                             model_path=model_path)   # Initialize estimator wrapper for S3 model.

            if project1_estimator.is_model_present(model_path=model_path):  # Check model presence at specified path.
                return project1_estimator
            return None                                                 # No production model available.
        except Exception as e:
            raise custom_exception(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Evaluate the newly trained model and (if available) the production model on the same test set.

        Workflow
        --------
        1) Load test data from DataIngestionArtifact.
        2) Split into features X and target y using TARGET_COLUMN.
        3) Map human-readable target labels to numeric codes via TargetValueMapping()._asdict().
        4) Obtain the trained model's F1 score from ModelTrainerArtifact.metric_artifact.
        5) If a production model exists in S3, compute its predictions and F1 score.
        6) Compare scores and compute acceptance decision and difference.

        Returns
        -------
        EvaluateModelResponse
            Dataclass summarizing trained vs production F1, acceptance decision, and score difference.

        Raises
        ------
        custom_exception
            If any step fails (I/O errors, metric computation, etc.).
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)   # Load held-out test dataset.

            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]   # Separate features/target.
            y = y.replace(
                TargetValueMapping()._asdict()                                   # Map string labels to ints.
            )

            # trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score  # Use stored metric.

            best_model_f1_score = None
            best_model = self.get_best_model()                                    # Attempt to fetch prod model.
            if best_model is not None:
                y_hat_best_model = best_model.predict(x)                          # Predict with prod model.
                best_model_f1_score = f1_score(y, y_hat_best_model)               # Compute prod F1 on test set.
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score  # Baseline for compare.
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")                                     # Log comparison summary.
            return result

        except Exception as e:
            raise custom_exception(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Run the end-to-end evaluation and build a ModelEvaluationArtifact summarizing the outcome.

        Returns
        -------
        ModelEvaluationArtifact
            Artifact containing:
              - is_model_accepted : bool
              - s3_model_path : str (production model path)
              - trained_model_path : str (path to the newly trained model)
              - changed_accuracy : float (trained - baseline best score)

        Raises
        ------
        custom_exception
            If evaluation or artifact construction fails.
        """  
        try:
            evaluate_model_response = self.evaluate_model()                        # Perform comparison.
            s3_model_path = self.model_eval_config.s3_model_key_path              # Record prod model path.

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,      # Decision flag.
                s3_model_path=s3_model_path,                                      # Where prod model lives.
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,  # Trained model path.
                changed_accuracy=evaluate_model_response.difference)              # Score delta for auditing.

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")  # Persist log.
            return model_evaluation_artifact
        except Exception as e:
            raise custom_exception(e, sys) from e