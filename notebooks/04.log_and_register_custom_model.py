# Databricks notebook source
# Databricks notebook source
import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from mlflow.models import infer_signature
from packages.config import ProjectConfig
from packages.paths import AllPaths
import json
from mlflow import MlflowClient
from mlflow.utils.environment import _mlflow_conda_env

# COMMAND ----------

from pathlib import Path
import os
import yaml

class AllPaths:
    def __init__(self) -> None:
        if self.is_databricks():
            # Running in Databricks
            self.filename_config = Path('/Workspace/Users/cremerfederico29@gmail.com/marvelmlops-cremerf/project-config.yml')
        else:
            # Running in VSCode or other local environment
            try:
                self.BASE_DIR = Path(__file__).resolve().parent.parent.parent
            except NameError:
                # __file__ is not defined, use current working directory
                self.BASE_DIR = Path.cwd()
            self.filename_config = self.BASE_DIR / 'project-config.yml'
        
        self.config = self.get_config_file()
        self.cremerf_catalog = f'/Volumes/{self.config["catalog_name"]}/{self.config["schema_name"]}/'
        self.data_volume = f"{self.cremerf_catalog}" + "data/"
    
    def is_databricks(self):
        # Check if the /databricks directory exists
        return os.path.exists('/databricks')
    
    def get_config_file(self):
        # Load configuration
        try:
            with open(self.filename_config, "r") as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            print(f"Configuration file not found at {self.filename_config}")
            raise

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

model
