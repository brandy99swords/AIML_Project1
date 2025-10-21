import os 
from pathlib import Path

project_name = "telco_churn"


list_files = [
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/components/model_pusher.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/articfact_entity.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/estimator.py",
    f"{project_name}/entity/s3_estimator.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/prediction_pipeline.py",
    f"{project_name}/pipeline/training_pipeline.py", 
    f"{project_name}/configuration/__init__.py",
    f"{project_name}/configuration/aws_connection.py",
    f"{project_name}/configuration/mongo_db_connect.py",
    f"{project_name}/configuration/pine_cone_connect.py"
    f"{project_name}/database_access/__init__.py",
    f"{project_name}/database_access/mongo_extract",
    f"{project_name}/database_access/pinecone_extract",
    f"{project_name}/cloud_storage/__init__.py",
    f"{project_name}/cloud_storage/aws_storage.py", 
    f"{project_name}/exceptions/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/utils/__init__.py",
    "app.py", 
    "requirements.txt",
    "DockerFile",
    ".dockerignore",
    "demo.py",
    "setup.py", 
    "config/model.yaml",
    "config/schema.yaml"
]

for filepath in list_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir:
        os.makedirs(filedir, exist_ok=True)
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w"):
            pass
    else:
        print(f'File is already present at: {filepath}')    