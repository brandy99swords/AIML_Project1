"""
Module: model_trainer.py

Purpose:
    Train/ select the best classifier using `neuro_mf.ModelFactory` on transformed arrays,
    compute evaluation metrics on the test split, and persist a composite `TelcoModel`
    (preprocessor + estimator). Produces a `ModelTrainerArtifact` capturing the saved model
    path and the test-set metrics.
"""

import sys
from typing import Tuple

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from neuro_mf import ModelFactory

from telco_churn.exceptions import custom_exception
from telco_churn.logger import logging
from telco_churn.utils.main_utils import load_numpy_array_data, load_object, save_object
from telco_churn.entity.config_entity import ModelTrainerConfig
from telco_churn.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)
from telco_churn.entity.estimator import TelcoModel


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        Initialize the ModelTrainer with artifacts and configuration.

        Parameters
        ----------
        data_transformation_artifact : DataTransformationArtifact
            Carries paths to transformed train/test arrays and the fitted preprocessing object.
        model_trainer_config : ModelTrainerConfig
            Holds trainer parameters such as expected_accuracy, model_config_file_path,
            and the output path for the trained model file.
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.ndarray, test: np.ndarray) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function uses neuro_mf to get the best model object and report of the best model
        
        Output      :   Returns metric artifact object and best model object
        On Failure  :   Write an exception log and then raise an exception

        Detailed behavior
        -----------------
        - Initializes a ModelFactory with the provided YAML model config.
        - Splits the provided numpy arrays into (X, y) for train and test.
        - Calls `get_best_model` with a base accuracy threshold.
        - Uses the selected best model to predict on X_test.
        - Computes F1, precision, and recall with explicit binary settings (pos_label=1).
        - Returns the best model detail (from ModelFactory) and the metrics artifact.

        Parameters
        ----------
        train : np.ndarray
            Transformed training array with features in all columns except the last,
            and the target as the final column.
        test : np.ndarray
            Transformed test array with the same column structure as `train`.

        Returns
        -------
        Tuple[object, object]
            (best_model_detail, metric_artifact), where:
            - best_model_detail: object returned by neuro_mf containing `.best_model` and `.best_score`
            - metric_artifact: ClassificationMetricArtifact with f1, precision, and recall
        """
        try:
            logging.info("Using neuro_mf to get best model object and report")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
            
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

            best_model_detail = model_factory.get_best_model(
                X=x_train, y=y_train, base_accuracy=self.model_trainer_config.expected_accuracy
            )
            model_obj = best_model_detail.best_model  # Selected estimator from the model factory.

            y_pred = model_obj.predict(x_test)  # Predictions on the test features.
            
            f1 = f1_score(y_test, y_pred, pos_label=1)
            precision = precision_score(y_test, y_pred, pos_label=1)
            recall = recall_score(y_test, y_pred, pos_label=1)
            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1, precision_score=precision, recall_score=recall
            )
            
            return best_model_detail, metric_artifact
        
        except Exception as e:
            raise custom_exception(e, sys) from e
        

    def initiate_model_trainer(self, ) -> ModelTrainerArtifact:
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception

        Detailed behavior
        -----------------
        - Loads transformed train/test arrays from disk.
        - Calls `get_model_object_and_report` to select the best model and compute metrics.
        - Loads the fitted preprocessing object.
        - Checks that the best model score meets the expected accuracy threshold.
        - Wraps the preprocessor and best model in `TelcoModel`.
        - Persists the composite model to the configured path.
        - Builds and returns a `ModelTrainerArtifact` with the model path and metrics.
        """
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            
            best_model_detail, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            # Guardrail: ensure the selected model meets or exceeds the expected baseline.
            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")

            # Compose a deployable object that applies preprocessing prior to inference.
            telco_model = TelcoModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model_detail.best_model
            )
            logging.info("Created telco model object with preprocessor and model")
            logging.info("Created best model file path.")
            save_object(self.model_trainer_config.trained_model_file_path, telco_model)

            # Package outputs into an artifact for downstream stages.
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise custom_exception(e, sys) from e