# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from pyspark.sql.functions import col
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import DoubleType
from datetime import datetime
import mlflow
import argparse
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import RegressionEvaluator

from hotel_reservation.config import ProjectConfig
from hotel_reservation.paths import AllPaths

ALLPATHS = AllPaths()

# COMMAND ----------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--new_model_uri",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)


args = parser.parse_args()
root_path = args.root_path
new_model_uri = args.new_model_uri
job_run_id = args.job_run_id
git_sha = args.git_sha

# COMMAND ----------

#config_path = (f"{root_path}/project_config.yml")
# config_path = ("/Volumes/mlops_test/house_prices/data/project_config.yml")
config = ProjectConfig.from_yaml(config_path=ALLPATHS.filename_config)

spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = FeatureEngineeringClient()

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# Define the serving endpoint
serving_endpoint_name = "hotel-reservations-cremerf-model-serving-fe"
serving_endpoint = workspace.serving_endpoints.get(serving_endpoint_name)
model_name = serving_endpoint.config.served_models[0].model_name
model_version = serving_endpoint.config.served_models[0].model_version
previous_model_uri = f"models:/{model_name}/{model_version}"

# COMMAND ----------

model_name

# COMMAND ----------

test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

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

test_set = test_set.withColumn(
    'no_of_previous_bookings_not_canceled',
    F.col('no_of_previous_bookings_not_canceled').cast('double')
).withColumn(
    'no_of_previous_cancellations',
    F.col('no_of_previous_cancellations').cast('double')
)

test_set = test_set.withColumn(
    'loyalty_score',
    calculate_loyalty_score_udf(
        F.col('no_of_previous_bookings_not_canceled'),
        F.col('no_of_previous_cancellations')
    )
)

test_set = test_set.withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))

# COMMAND ----------

test_set.toPandas()

# COMMAND ----------

X_test_spark = test_set.select(num_features + cat_features + ["loyalty_score", "Booking_ID", "update_timestamp_utc"])
y_test_spark = test_set.select("Booking_ID", target)

# COMMAND ----------

# Prepare feature columns
feature_columns = num_features + cat_features + ["loyalty_score"] + ["update_timestamp_utc"]

# Ensure all required columns are present
missing_columns = [col for col in feature_columns if col not in test_set.columns]
if missing_columns:
    raise ValueError(f"Missing columns in test_set: {missing_columns}")

# Select the features in the correct order
X_test_spark = test_set.select(*feature_columns, "Booking_ID")

# COMMAND ----------

display(X_test_spark)

# COMMAND ----------

# Generate predictions from both models
predictions_previous = fe.score_batch(model_uri=previous_model_uri, df=X_test_spark)
predictions_new = fe.score_batch(model_uri=new_model_uri, df=X_test_spark)

predictions_new = predictions_new.withColumnRenamed("prediction", "prediction_new")
predictions_old = predictions_previous.withColumnRenamed("prediction", "prediction_old")
test_set = test_set.select("Booking_ID", "loyalty_score")

# Join the DataFrames on the 'id' column
df = test_set \
    .join(predictions_new, on="Booking_ID") \
    .join(predictions_old, on="Booking_ID")