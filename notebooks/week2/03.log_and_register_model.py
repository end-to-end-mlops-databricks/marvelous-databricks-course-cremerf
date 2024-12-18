# Databricks notebook source
# MAGIC %restart_python

# COMMAND ----------


import mlflow
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder

from hotel_reservation.classifier import CancellationModel
from hotel_reservation.config import ProjectConfig
from hotel_reservation.paths import AllPaths

# COMMAND ----------

# Initialize the class
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
# pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**parameters))])

# COMMAND ----------

model_pipeline = CancellationModel(config=config, preprocessor=preprocessor, classifier=LGBMClassifier)

# COMMAND ----------

mlflow.set_experiment(experiment_name="/Shared/hotel-reservations-cremerf")
git_sha = "f6564c4210596362360ac94671e3c2621330bac2"

# Start an MLflow run to track the training process
with mlflow.start_run(
    tags={"git_sha": f"{git_sha}", "branch": "week2"},
) as run:
    run_id = run.info.run_id

    model_pipeline.pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.pipeline.predict(X_test)

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

    mlflow.sklearn.log_model(sk_model=model_pipeline, artifact_path="lightgbm-pipeline-model", signature=signature)


# COMMAND ----------

model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model",
    name=f"{catalog_name}.{schema_name}.hotel_reservations_model_basic",
    tags={"git_sha": f"{git_sha}"},
)

# COMMAND ----------

run = mlflow.get_run(run_id)
dataset_info = run.inputs.dataset_inputs[0].dataset
dataset_source = mlflow.data.get_source(dataset_info)
dataset_source.load()
