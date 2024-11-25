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

from databricks import feature_engineering
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
import mlflow
import argparse
from pyspark.sql import functions as F
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from databricks.feature_engineering import FeatureFunction, FeatureLookup
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


config_path = (f"{root_path}/project_config.yml")
# config_path = ("/Volumes/mlops_test/house_prices/data/project_config.yml")
config = ProjectConfig.from_yaml(config_path=config_path)

# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name


# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.hotel_features"
loyalty_function_name = f"{catalog_name}.{schema_name}.calculate_loyalty_score_week5"

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").drop("lead_time", "no_of_special_requests", "avg_price_per_room")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

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
            output_name="",
            input_bindings={"": ""},
        ),
    ],
    exclude_columns=["update_timestamp_utc"]
)





