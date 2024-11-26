import argparse

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from pyspark.sql.types import DoubleType, FloatType

from hotel_reservation.config import ProjectConfig
from hotel_reservation.paths import AllPaths

ALLPATHS = AllPaths()

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


config_path = f"{root_path}/project-config.yml"
config = ProjectConfig.from_yaml(config_path=config_path)

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

# Define the serving endpoint
serving_endpoint_name = "hotel-reservations-cremerf-model-serving-fe"
serving_endpoint = workspace.serving_endpoints.get(serving_endpoint_name)
model_name = serving_endpoint.config.served_models[0].model_name
model_version = serving_endpoint.config.served_models[0].model_version
previous_model_uri = f"models:/{model_name}/{model_version}"

test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")


# Define the UDF using your provided function
def calculate_loyalty_score(no_of_previous_bookings_not_canceled, no_of_previous_cancellations):
    # Define weightings
    w1 = 1.5  # Weight the number of times a previous booking was NOT cancelled
    w2 = 1.0  # Weight the number of times a previous booking was cancelled

    # Calculate loyalty score
    loyalty_score = (w1 * no_of_previous_bookings_not_canceled) - (w2 * no_of_previous_cancellations)
    return loyalty_score


# Register the UDF
calculate_loyalty_score_udf = F.udf(calculate_loyalty_score, DoubleType())


test_set = test_set.withColumn(
    "no_of_previous_bookings_not_canceled", F.col("no_of_previous_bookings_not_canceled").cast("double")
).withColumn("no_of_previous_cancellations", F.col("no_of_previous_cancellations").cast("double"))

test_set = test_set.withColumn(
    "loyalty_score",
    calculate_loyalty_score_udf(F.col("no_of_previous_bookings_not_canceled"), F.col("no_of_previous_cancellations")),
)

test_set = test_set.withColumn("lead_time", F.col("lead_time").cast("integer"))
test_set = test_set.withColumn("no_of_special_requests", F.col("no_of_special_requests").cast("integer"))
test_set = test_set.withColumn('avg_price_per_room', F.col('avg_price_per_room').cast(FloatType()))
test_set = test_set.withColumn("update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))

X_test_spark = test_set.select(num_features + cat_features + ["loyalty_score", "Booking_ID", "update_timestamp_utc"])
y_test_spark = test_set.select("Booking_ID", target)

# Prepare feature columns
feature_columns = num_features + cat_features + ["loyalty_score"] + ["update_timestamp_utc"]

# Ensure all required columns are present
missing_columns = [col for col in feature_columns if col not in test_set.columns]
if missing_columns:
    raise ValueError(f"Missing columns in test_set: {missing_columns}")

# Select the features in the correct order
X_test_spark = test_set.select(*feature_columns, "Booking_ID")

# Generate predictions from both models
predictions_previous = fe.score_batch(model_uri=previous_model_uri, df=X_test_spark)
predictions_new = fe.score_batch(model_uri=new_model_uri, df=X_test_spark)

predictions_new = predictions_new.withColumnRenamed("prediction", "prediction_new")
predictions_old = predictions_previous.withColumnRenamed("prediction", "prediction_old")
# Ensure you have the target variable in test_set
test_set_labels = test_set.select("Booking_ID", target)

# Join the DataFrames on 'Booking_ID'
df = test_set_labels.join(predictions_new.select("Booking_ID", "prediction_new"), on="Booking_ID").join(
    predictions_old.select("Booking_ID", "prediction_old"), on="Booking_ID"
)

# Now 'df' contains: Booking_ID, target (true label), prediction_new, prediction_old

# Ensure the predictions and labels are of the same data type
df = df.withColumn(target, F.col(target).cast("double"))
df = df.withColumn("prediction_new", F.col("prediction_new").cast("double"))
df = df.withColumn("prediction_old", F.col("prediction_old").cast("double"))

# Create evaluators for F1 Score and Accuracy
evaluator_f1 = MulticlassClassificationEvaluator(labelCol=target, metricName="f1")
evaluator_acc = MulticlassClassificationEvaluator(labelCol=target, metricName="accuracy")

# Evaluate the new model
evaluator_f1.setPredictionCol("prediction_new")
evaluator_acc.setPredictionCol("prediction_new")
f1_new = evaluator_f1.evaluate(df)
accuracy_new = evaluator_acc.evaluate(df)

# Evaluate the old model
evaluator_f1.setPredictionCol("prediction_old")
evaluator_acc.setPredictionCol("prediction_old")
f1_old = evaluator_f1.evaluate(df)
accuracy_old = evaluator_acc.evaluate(df)

# Compare models based on F1 Score and Accuracy
print(f"F1 Score for New Model: {f1_new}")
print(f"F1 Score for Old Model: {f1_old}")
print(f"Accuracy for New Model: {accuracy_new}")
print(f"Accuracy for Old Model: {accuracy_old}")

# Decide which model is better based on F1 Score
if f1_new > f1_old:
    print("New model is better based on F1 Score.")
    # Register the new model
    model_version = mlflow.register_model(
        model_uri=new_model_uri,
        name=f"{catalog_name}.{schema_name}.hotel_reservations_model_fe",
        tags={"git_sha": f"{git_sha}", "job_run_id": job_run_id},
    )

    print("New model registered with version:", model_version.version)
    dbutils.jobs.taskValues.set(key="model_version", value=model_version.version)
    dbutils.jobs.taskValues.set(key="model_update", value=1)
else:
    print("Old model is better based on F1 Score.")
    dbutils.jobs.taskValues.set(key="model_update", value=0)
