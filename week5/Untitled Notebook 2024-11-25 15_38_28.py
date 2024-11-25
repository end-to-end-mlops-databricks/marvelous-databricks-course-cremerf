# Databricks notebook source
from databricks import feature_engineering
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
import mlflow
import argparse
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from lightgbm import LGBMClassifier
from hotel_reservation.classifier import CancellationModel
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from hotel_reservation.config import ProjectConfig
from hotel_reservation.paths import AllPaths

# COMMAND ----------

ALLPATHS = AllPaths()

config = ProjectConfig.from_yaml(config_path=ALLPATHS.filename_config)

# COMMAND ----------

# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# Define the UDF using your provided function
def calculate_loyalty_score(no_of_previous_bookings_not_canceled, no_of_previous_cancellations):
    # Define weightings
    w1 = 1.5     # Weight the number of times a previous booking was NOT cancelled
    w2 = 1.0     # Weight the number of times a previous booking was cancelled

    # Calculate loyalty score
    loyalty_score = (w1 * no_of_previous_bookings_not_canceled) - (w2 * no_of_previous_cancellations)
    return loyalty_score

# Register the UDF
calculate_loyalty_score_udf = udf(calculate_loyalty_score, DoubleType())

# COMMAND ----------

# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.hotel_features"
loyalty_function_name = f"{catalog_name}.{schema_name}.calculate_loyalty_score"

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").drop("lead_time", "no_of_special_requests", "avg_price_per_room")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

# COMMAND ----------

# Cast necessary columns in the training set
train_set = train_set.withColumn(
    'no_of_previous_bookings_not_canceled',
    F.col('no_of_previous_bookings_not_canceled').cast('double')
).withColumn(
    'no_of_previous_cancellations',
    F.col('no_of_previous_cancellations').cast('double')
)

test_set = test_set.withColumn(
    'no_of_previous_bookings_not_canceled',
    F.col('no_of_previous_bookings_not_canceled').cast('double')
).withColumn(
    'no_of_previous_cancellations',
    F.col('no_of_previous_cancellations').cast('double')
)

# COMMAND ----------

# Feature engineering setup
training_set = fe.create_training_set(
    df=train_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["lead_time", "no_of_special_requests", "avg_price_per_room"],
            lookup_key="Booking_ID",
        ),
        FeatureFunction(
            udf_name=loyalty_function_name,
            output_name="loyalty_score",
            input_bindings={"no_of_previous_bookings_not_canceled": "no_of_previous_bookings_not_canceled",
                            "no_of_previous_cancellations": "no_of_previous_cancellations"},
        ),
    ],
    exclude_columns=["update_timestamp_utc"]
)

# COMMAND ----------

test_set = test_set.withColumn(
    'loyalty_score',
    calculate_loyalty_score_udf(
        F.col('no_of_previous_bookings_not_canceled'),
        F.col('no_of_previous_cancellations')
    )
)

# COMMAND ----------

train_df = training_set.load_df().toPandas()
test_df = test_set.toPandas()

X_train = train_df[num_features + cat_features + ["loyalty_score"]]
y_train = train_df[target]
# Ensure 'loyalty_score' is included in the features
X_test = test_df[num_features + cat_features + ["loyalty_score"]]
y_test = test_df[target]

# COMMAND ----------

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)], remainder="passthrough"
)

model_pipeline = CancellationModel(config=config, preprocessor=preprocessor, classifier=LGBMClassifier)

# COMMAND ----------

mlflow.set_experiment(experiment_name="/Shared/hotel-reservations-fe-cremerf")

with mlflow.start_run(tags={"branch": "week5",
                            "git_sha": f"{git_sha}",
                            "job_run_id": job_run_id}) as run:
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
    mlflow.log_input(dataset, context="training")

    mlflow.sklearn.log_model(sk_model=model_pipeline, artifact_path="lightgbm-pipeline-model", signature=signature)
