# Databricks notebook source
# MAGIC %restart_python

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
from mlflow_train import CancellatioModelWrapper
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

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

X_train = train_set[num_features + cat_features].toPandas()
y_train = train_set[[target]].toPandas()

X_test = test_set[num_features + cat_features].toPandas()
y_test = test_set[[target]].toPandas()

# COMMAND ----------

run_id

# COMMAND ----------

wrapped_model = CancellatioModelWrapper(model.pipeline)  # we pass the loaded model to the wrapper
example_input = X_test.iloc[0:1]  # Select the first row for prediction as example
example_prediction = wrapped_model.predict(model_input=example_input)
print("Example Prediction:", example_prediction)

# COMMAND ----------

wrapped_model

# COMMAND ----------

mlflow.set_experiment(experiment_name="/Shared/hotel-reservations-cremerf-pyfunc")
git_sha = "b3fd67dfc8595127c3f8787b59116a064195df53"

with mlflow.start_run(tags={"branch": "week2", "git_sha": f"{git_sha}"}) as run:
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output={"Prediction": example_prediction})
    dataset = mlflow.data.from_spark(train_set, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[
            "code/marvelmlops-0.0.1-py3-none-any.whl",
        ],
        additional_conda_channels=None,
    )
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc-hotel-reservations-cremerf-model",
        #code_paths=["../Volumes/mlops_students/cremerfederico29/packages/"],
        signature=signature,
    )


# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(f'runs:/{run_id}/pyfunc-hotel-reservations-cremerf-model')
loaded_model.unwrap_python_model()

# COMMAND ----------

model_name = f"{catalog_name}.{schema_name}.hotel_reservation_cremerf_pyfunc"

model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/pyfunc-hotel-reservations-cremerf-model',
    name=model_name,
    tags={"git_sha": f"{git_sha}"})

# COMMAND ----------

with open("model_version.json", "w") as json_file:
    json.dump(model_version.__dict__, json_file, indent=4)

# COMMAND ----------

model_name

# COMMAND ----------

model_version_alias = "the_best_model"
client.set_registered_model_alias(model_name, model_version_alias, "1")

model_uri = f"models:/{model_name}@{model_version_alias}"
model = mlflow.pyfunc.load_model(model_uri)


# COMMAND ----------

client.get_model_version_by_alias(model_name, model_version_alias)

# COMMAND ----------

model
