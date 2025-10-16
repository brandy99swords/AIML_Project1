from dataclasses import dataclass

@dataclass
class DataIngestionArtifact: 
    trained_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact: 
    validation_status: bool
    message: str
    drift_report_path: str


@dataclass
class DataTransformationArtifact: 
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str


@dataclass
class DataClassificationMetricArtifact: 
    f1_score: float 
    prediction_score: float 
    rmse_score: float 
    recall_score: float 


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    metric_artifact: DataClassificationMetricArtifact 
    

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    changed_accuracy: float 
    changed_rmse: float
    s3_model_path: str
    trained_model_path: str  

@dataclass
class ModelPusherArtifact:
    bucket_name: str
    s3_model_path: str   
