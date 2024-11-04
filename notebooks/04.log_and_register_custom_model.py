# Databricks notebook source
# MAGIC %pip install python-dotenv

# COMMAND ----------

# Databricks notebook source
import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from mlflow.models import infer_signature
from packages.config import ProjectConfig
from packages.paths import AllPaths
from packages.classifier import CancellationModel
import json
from mlflow import MlflowClient
from mlflow.utils.environment import _mlflow_conda_env

# COMMAND ----------

ALLPATHS = AllPaths()

# COMMAND ----------

config = ProjectConfig.from_yaml(config_path=ALLPATHS.filename_config)


# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")
client = MlflowClient()

# COMMAND ----------

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name


spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

run_id = mlflow.search_runs(
    experiment_names=["/Shared/hotel-reservations-cremerf"],
    filter_string="tags.branch='week2'",
).run_id[0]

model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")


# COMMAND ----------



# COMMAND ----------

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

X_train = train_set[num_features + cat_features].toPandas()
y_train = train_set[[target]].toPandas()

X_test = test_set[num_features + cat_features].toPandas()
y_test = test_set[[target]].toPandas()

# COMMAND ----------

wrapped_model = HotelReservationseModelWrapper(model)  # we pass the loaded model to the wrapper
example_input = X_test.iloc[0:1]  # Select the first row for prediction as example
example_prediction = wrapped_model.predict(context=None, model_input=example_input)
print("Example Prediction:", example_prediction)
