# Databricks notebook source
# Databricks notebook source
from packages.config import ProjectConfig
from packages.paths import AllPaths
from pyspark.sql import SparkSession
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
from mlflow.models import infer_signature
import os
from pathlib import Path

# COMMAND ----------

from pathlib import Path
import yaml

class AllPaths:
    def __init__(self):
        # Assuming your base directory is correctly set up
        self.BASE_DIR = Path("/Workspace/Users/cremerfederico29@gmail.com/marvelmlops-cremerf")
        self.filename_config = self.BASE_DIR / 'project-config.yml'
        self.config = self.get_config_file()
        # Update the paths below as necessary
        self.cremerf_catalog = f'/Volumes/{self.config["catalog_name"]}/{self.config["schema_name"]}/'
        self.data_volume = f"{self.cremerf_catalog}" + "data/"

    def get_config_file(self):
        # Load configuration
        try:
            with open(self.filename_config, "r") as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None

# Initialize the class
ALLPATHS = AllPaths()


# COMMAND ----------

import yaml
from pathlib import Path
import os

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

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# Load training and testing sets from Databricks tables
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# COMMAND ----------

X_train = train_set[num_features + cat_features]
y_train = train_set[target]

X_test = test_set[num_features + cat_features]
y_test = test_set[target]

# COMMAND ----------

# Define the preprocessor for categorical features
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)], remainder="passthrough"
)

# Create the pipeline with preprocessing and the LightGBM regressor
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**parameters))])

# COMMAND ----------

mlflow.set_experiment(experiment_name="/Shared/hotel-reservations-cremerf")
git_sha = "50a9297454e49cbec3c6b681981b38f1485b3c10"

# Start an MLflow run to track the training process
with mlflow.start_run(
    tags={"git_sha": f"{git_sha}", "branch": "week2"},
) as run:
    run_id = run.info.run_id

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate the model performance using classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("accuracy", accuracy)

    # Log classification report metrics
    for class_label, metrics in report.items():
        if isinstance(metrics, dict):
            mlflow.log_metric(f"precision_{class_label}", metrics["precision"])
            mlflow.log_metric(f"recall_{class_label}", metrics["recall"])
            mlflow.log_metric(f"f1-score_{class_label}", metrics["f1-score"])

    signature = infer_signature(model_input=X_train, model_output=y_pred)
    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")

    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="lightgbm-pipeline-model", signature=signature)


# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")  # It must be -uc for registering models to Unity Catalog
