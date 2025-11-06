import sys
from pandas import DataFrame

from AIML_1013_Project1.cloud_storage.aws_storage import SimpleStorageService
from AIML_1013_Project1.exceptions import custom_exception
from AIML_1013_Project1.entity.estimator import project1Model


class project1Estimator:
    """
    Manage persistence and inference for a Telco churn model stored in S3.

    Notes
    -----
    - This class does not alter or wrap the model's API; it simply defers to
      `TelcoModel.predict(dataframe=...)` for inference once the model is loaded.
    - The first call to `predict(...)` triggers a lazy load of the model if it
      has not already been loaded via `load_model()`.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket containing the model artifact.
    model_path : str
        S3 key (path) to the serialized model within `bucket_name`.

    Attributes
    ----------
    bucket_name : str
        Target S3 bucket for model storage.
    s3 : SimpleStorageService
        Helper client for S3 interactions (upload, load, existence checks).
    model_path : str
        S3 key to the model file.
    loaded_model : TelcoModel or None
        Cached in-memory model instance. Loaded on demand.
    """

    def __init__(self, bucket_name, model_path,):
        """
        Initialize the estimator with S3 location details.

        Parameters
        ----------
        bucket_name : str
            Name of your model bucket.
        model_path : str
            Location (S3 key) of your model in the bucket.
        """
        self.bucket_name = bucket_name
        self.s3 = SimpleStorageService()
        self.model_path = model_path
        self.loaded_model: project1Model = None

    def is_model_present(self, model_path):
        """
        Check whether a model artifact exists at the given S3 key.

        Parameters
        ----------
        model_path : str
            S3 key to check for existence within `self.bucket_name`.

        Returns
        -------
        bool
            True if the key exists in the bucket; otherwise False.

        Notes
        -----
        - Catches `custom_exception` raised by the storage layer, prints the error,
          and returns False to indicate absence or error.
        """
        try:
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=model_path)
        except custom_exception as e:
            print(e)
            return False

    def load_model(self,) -> project1Model:
        """
        Load the model from S3 using the configured bucket and key.

        Returns
        -------
        TelcoModel
            The deserialized model object.

        Notes
        -----
        - This method does not cache the model on its own. The `predict(...)`
          method will cache the loaded instance in `self.loaded_model` when first used.
        """
        return self.s3.load_model(self.model_path, bucket_name=self.bucket_name)

    def save_model(self, from_file, remove: bool = False) -> None:
        """
        Upload a local model artifact to S3 at `self.model_path`.

        Parameters
        ----------
        from_file : str
            Local filesystem path to the model file to upload.
        remove : bool, optional (default=False)
            If True, remove the local file after successful upload.

        Raises
        ------
        custom_exception
            Wraps and re-raises any underlying exceptions thrown during upload.
        """
        try:
            self.s3.upload_file(
                from_file,
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove,
            )
        except Exception as e:
            raise custom_exception(e, sys)

    def predict(self, dataframe: DataFrame):
        """
        Generate predictions for the provided dataframe using the Telco model.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Input features in tabular form compatible with the trained model's expectations.

        Returns
        -------
        Any
            The prediction output of `TelcoModel.predict(dataframe=...)`. The exact
            type depends on the underlying model implementation (e.g., numpy array,
            pandas Series, list).

        Raises
        ------
        custom_exception
            Wraps and re-raises any exceptions during model loading or prediction.

        Notes
        -----
        - If `self.loaded_model` is None, the model is loaded from S3 via `load_model()`
          before prediction. The loaded instance is cached for subsequent calls.
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise custom_exception(e, sys)