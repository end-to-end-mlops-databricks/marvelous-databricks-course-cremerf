# SDK
from packages.config import ProjectConfig
from packages.paths import AllPaths

# Built-in
import os
import dotenv
import json

# Overall
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

# MLFlow
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env

ALLPATHS = AllPaths()
config = ProjectConfig.from_yaml(config_path=ALLPATHS.filename_config)
spark = SparkSession.builder.getOrCreate()

num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name


mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")
client = MlflowClient()

run_id = mlflow.search_runs(
    experiment_names=["/Shared/hotel-reservations-cremerf"],
    filter_string="tags.branch='week2'",
).run_id[0]

model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")


class CancellatioModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, model_input, return_proba=False):
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        # Prediction based on specified mode
        if return_proba:
            # Predict probabilities for each class
            probabilities = self.model.predict_proba(model_input)
            predictions = {
                "Probabilities": probabilities.tolist(),
                "Predicted Class": probabilities.argmax(axis=1).tolist(),
            }
        else:
            # Predict class labels directly
            predicted_classes = self.model.predict(model_input)
            predictions = {"Predicted Class": predicted_classes.tolist()}

        return predictions