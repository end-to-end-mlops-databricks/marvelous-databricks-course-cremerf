"""
This script trains a LightGBM model for house price prediction with feature engineering.
Key functionality:
- Loads training and test data from Databricks tables
- Performs feature engineering using Databricks Feature Store
- Creates a pipeline with preprocessing and LightGBM regressor
- Tracks the experiment using MLflow
- Logs model metrics, parameters and artifacts
- Handles feature lookups and custom feature functions
- Outputs model URI for downstream tasks

The model uses both numerical and categorical features, including a custom calculated house age feature.
"""

import argparse

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder

from hotel_reservation.classifier import CancellationModel
from hotel_reservation.config import ProjectConfig
from hotel_reservation.paths import AllPaths

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
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
parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
git_sha = args.git_sha
job_run_id = args.job_run_id

ALLPATHS = AllPaths()

config = ProjectConfig.from_yaml(config_path=ALLPATHS.filename_config)

# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name


# Define the UDF using your provided function
def calculate_loyalty_score(no_of_previous_bookings_not_canceled, no_of_previous_cancellations):
    # Define weightings
    w1 = 1.5  # Weight the number of times a previous booking was NOT cancelled
    w2 = 1.0  # Weight the number of times a previous booking was cancelled

    # Calculate loyalty score
    loyalty_score = (w1 * no_of_previous_bookings_not_canceled) - (w2 * no_of_previous_cancellations)
    return loyalty_score


# Register the UDF
calculate_loyalty_score_udf = udf(calculate_loyalty_score, DoubleType())

# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.hotel_features"
loyalty_function_name = f"{catalog_name}.{schema_name}.calculate_loyalty_score"

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").drop(
    "lead_time", "no_of_special_requests", "avg_price_per_room"
)
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

# Cast necessary columns in the training set
train_set = train_set.withColumn(
    "no_of_previous_bookings_not_canceled", F.col("no_of_previous_bookings_not_canceled").cast("double")
).withColumn("no_of_previous_cancellations", F.col("no_of_previous_cancellations").cast("double"))

test_set = test_set.withColumn(
    "no_of_previous_bookings_not_canceled", F.col("no_of_previous_bookings_not_canceled").cast("double")
).withColumn("no_of_previous_cancellations", F.col("no_of_previous_cancellations").cast("double"))

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
            input_bindings={
                "no_of_previous_bookings_not_canceled": "no_of_previous_bookings_not_canceled",
                "no_of_previous_cancellations": "no_of_previous_cancellations",
            },
        ),
    ],
    exclude_columns=["update_timestamp_utc"],
)

test_set = test_set.withColumn(
    "loyalty_score",
    calculate_loyalty_score_udf(F.col("no_of_previous_bookings_not_canceled"), F.col("no_of_previous_cancellations")),
)

train_df = training_set.load_df().toPandas()
test_df = test_set.toPandas()

X_train = train_df[num_features + cat_features + ["loyalty_score"]]
y_train = train_df[target]
# Ensure 'loyalty_score' is included in the features
X_test = test_df[num_features + cat_features + ["loyalty_score"]]
y_test = test_df[target]


preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)], remainder="passthrough"
)

model_pipeline = CancellationModel(config=config, preprocessor=preprocessor, classifier=LGBMClassifier)

mlflow.set_experiment(experiment_name="/Shared/hotel-reservations-fe-cremerf")

with mlflow.start_run(tags={"branch": "week5", "git_sha": f"{git_sha}", "job_run_id": job_run_id}) as run:
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

    # Log model with feature engineering
    fe.log_model(
        model=model_pipeline,
        flavor=mlflow.sklearn,
        artifact_path="lightgbm-pipeline-model-fe",
        training_set=training_set,
        signature=signature,
    )

model_uri = f"runs:/{run_id}/lightgbm-pipeline-model-fe"
dbutils.jobs.taskValues.set(key="new_model_uri", value=model_uri)
