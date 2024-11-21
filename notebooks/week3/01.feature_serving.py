# Databricks notebook source
"""
Create feature table in unity catalog, it will be a delta table
Create online table which uses the feature delta table created in the previous step
Create a feature spec. When you create a feature spec,
you specify the source Delta table.
This allows the feature spec to be used in both offline and online scenarios.
For online lookups, the serving endpoint automatically uses the online table to perform low-latency feature lookups.
The source Delta table and the online table must use the same primary key.

"""

# COMMAND ----------
# Standard library imports
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
import mlflow
import pandas as pd
import requests
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

# Local application imports
from hotel_reservation.config import ProjectConfig
from hotel_reservation.paths import AllPaths

spark = SparkSession.builder.getOrCreate()

# Initialize Databricks clients
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# Set the MLflow registry URI
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

ALLPATHS = AllPaths()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Configurations and Data
# MAGIC We load the configuration for the project, including the feature and target columns. We also load training and test datasets from the catalog.

# COMMAND ----------

# Load config
config = ProjectConfig.from_yaml(config_path=ALLPATHS.filename_config)

# COMMAND ----------

# Get feature columns details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
catalog_name = config.catalog_name
schema_name = config.schema_name

# Define table names
feature_table_name = f"{catalog_name}.{schema_name}.hotel_reservations_preds"
online_table_name = f"{catalog_name}.{schema_name}.hotel_reservations_preds_online"

# Load training and test sets from Catalog
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

df = pd.concat([train_set, test_set])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load a Registered Model
# MAGIC In this step, we load a registered MLflow model to be used for making predictions.

# COMMAND ----------

# Load the MLflow model for predictions
pipeline = mlflow.sklearn.load_model(f"models:/{catalog_name}.{schema_name}.hotel_reservations_model_basic/5")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Prepare DataFrame for Feature Table
# MAGIC We prepare a DataFrame containing the features needed for predictions and then create a feature table.

# COMMAND ----------

# Prepare the DataFrame for predictions and feature table creation - these features are the ones we want to serve.
preds_df = df[["Booking_ID", "lead_time", "no_of_special_requests", "avg_price_per_room"]]
preds_df["Predicted_BookingStatus"] = pipeline.predict(df[cat_features + num_features])

# COMMAND ----------

preds_df = spark.createDataFrame(preds_df)

# COMMAND ----------

# 1. Create the feature table in Databricks

# COMMAND ----------

fe.create_table(
    name=feature_table_name,
    primary_keys=["Booking_ID"],
    df=preds_df,
    description="Hotel Reservations predictions feature table",
)

# Enable Change Data Feed
spark.sql(f"""
    ALTER TABLE {feature_table_name}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Create Online Table
# MAGIC In this step, we create an online table that uses the feature table created earlier

# COMMAND ----------

# 2. Create the online table using feature table

spec = OnlineTableSpec(
    primary_key_columns=["Booking_ID"],
    source_table_full_name=feature_table_name,
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    perform_full_copy=False,
)

# Create the online table in Databricks
online_table_pipeline = workspace.online_tables.create(name=online_table_name, spec=spec)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Create Feature Lookup and Feature Spec
# MAGIC We define the features to look up from the feature table and create the feature specification for serving.

# COMMAND ----------

# 3. Create feture look up and feature spec table feature table

# Define features to look up from the feature table
features = [
    FeatureLookup(
        table_name=feature_table_name,
        lookup_key="Booking_ID",
        feature_names=["lead_time", "no_of_special_requests", "avg_price_per_room"],
    )
]

# Create the feature spec for serving
feature_spec_name = f"{catalog_name}.{schema_name}.return_predictions"

fe.create_feature_spec(name=feature_spec_name, features=features, exclude_columns=None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Deploy Feature Serving Endpoint
# MAGIC We deploy a feature serving endpoint using the feature specification.
# MAGIC

# COMMAND ----------

# 4. Create endpoing using feature spec

# Create a serving endpoint for the house booking predictions
workspace.serving_endpoints.create(
    name="hotel-reservations-cremerf-feature-serving",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=feature_spec_name,  # feature spec name defined in the previous step
                scale_to_zero_enabled=True,
                workload_size="Small",  # Define the workload size (Small, Medium, Large)
            )
        ]
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Call the Endpoint
# MAGIC We make a sample request to the feature serving endpoint to validate the setup.

# COMMAND ----------

dbutils = DBUtils(spark)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

id_list = preds_df["Booking_ID"]

start_time = time.time()
serving_endpoint = f"https://{host}/serving-endpoints/hotel-reservations-cremerf-feature-serving/invocations"
response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": [{"Booking_ID": "182"}]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------

# another way to call the endpoint

response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_split": {"columns": ["Booking_ID"], "data": [["182"]]}},
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Test

# COMMAND ----------

# Initialize variables
serving_endpoint = f"https://{host}/serving-endpoints/hotel-reservations-cremerf-feature-serving/invocations"
id_list = preds_df.select("Booking_ID").rdd.flatMap(lambda x: x).collect()
headers = {"Authorization": f"Bearer {token}"}
num_requests = 10

# COMMAND ----------


def send_request():
    random_id = random.choice(id_list)
    start_time = time.time()
    response = requests.post(
        serving_endpoint,
        headers=headers,
        json={"dataframe_records": [{"booking_id": random_id}]},
    )
    end_time = time.time()
    latency = end_time - start_time  # Calculate latency for this request
    return response.status_code, latency


# Measure total execution time
total_start_time = time.time()
latencies = []

# Send requests concurrently
with ThreadPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(send_request) for _ in range(num_requests)]

    for future in as_completed(futures):
        status_code, latency = future.result()
        latencies.append(latency)

total_end_time = time.time()
total_execution_time = total_end_time - total_start_time

# Calculate the average latency
average_latency = sum(latencies) / len(latencies)

print("\nTotal execution time:", total_execution_time, "seconds")
print("Average latency per request:", average_latency, "seconds")
