# Databricks notebook source
# MAGIC %md
# MAGIC ## Create Online Table for house features
# MAGIC We already created house_features table as feature look up table.

# COMMAND ----------

import time

import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from pyspark.sql import SparkSession

from hotel_reservation.config import ProjectConfig
from hotel_reservation.paths import AllPaths
from pyspark.dbutils import DBUtils


spark = SparkSession.builder.getOrCreate()

# Initialize Databricks clients
workspace = WorkspaceClient()

# COMMAND ----------

ALLPATHS = AllPaths()

# COMMAND ----------

# Load config
config = ProjectConfig.from_yaml(config_path=ALLPATHS.filename_config)
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

online_table_name = f"{catalog_name}.{schema_name}.hotel_features_online"
spec = OnlineTableSpec(
    primary_key_columns=["Booking_ID"],
    source_table_full_name=f"{catalog_name}.{schema_name}.hotel_features",
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    perform_full_copy=False,
)

online_table_pipeline = workspace.online_tables.create(name=online_table_name, spec=spec)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create endpoint

# COMMAND ----------

workspace.serving_endpoints.create(
    name="hotel-reservations-cremerf-model-serving-fe",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.hotel_reservations_model_fe",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=2,
            )
        ]
    ),
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Call the Endpoint
# MAGIC This section demonstrates how to call the model endpoint to make predictions, including the use of the feature lookup table.


# COMMAND ----------

dbutils = DBUtils(spark)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

required_columns = [
    "type_of_meal_plan",
    "required_car_parking_space",
    "room_type_reserved",
    "arrival_year",
    "arrival_month",
    "arrival_date",
    "market_segment_type",
    "repeated_guest",
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "Booking_ID",
    "update_timestamp_utc",
    "lead_time",
    "no_of_special_requests",
    "avg_price_per_room",
    #"loyalty_score"
]

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()

# COMMAND ----------

# Assuming 'train_set' already has the 'update_timestamp_utc' column as Timestamp objects
import pandas as pd
train_set['update_timestamp_utc'] = train_set['update_timestamp_utc'].apply(lambda x: x.isoformat() if pd.notnull(x) else None)

# If you're preparing sampled_records as previously described
sampled_records = train_set[required_columns + ['update_timestamp_utc']].sample(n=1000, replace=True).to_dict(orient='records')
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.2 Call the Model Endpoint for Prediction
# MAGIC In this step, we make a request to the model serving endpoint with the sampled data.

# COMMAND ----------

start_time = time.time()

model_serving_endpoint = f"https://{host}/serving-endpoints/hotel-reservations-cremerf-model-serving-fe/invocations"

response = requests.post(
    f"{model_serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": dataframe_records[0]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Load the Feature Lookup Table
# MAGIC We load the `hotel_features` table, which is used for feature lookup.


# COMMAND ----------

house_features = spark.table(f"{catalog_name}.{schema_name}.hotel_features").toPandas()


# COMMAND ----------

house_features.dtypes
