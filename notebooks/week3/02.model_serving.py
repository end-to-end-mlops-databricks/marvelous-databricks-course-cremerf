# Databricks notebook source
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    Route,
    ServedEntityInput,
    TrafficConfig,
)
from pyspark.sql import SparkSession

from hotel_reservation.config import ProjectConfig
from hotel_reservation.paths import AllPaths

workspace = WorkspaceClient()
spark = SparkSession.builder.getOrCreate()

ALLPATHS = AllPaths()

config = ProjectConfig.from_yaml(config_path=ALLPATHS.filename_config)

catalog_name = config.catalog_name
schema_name = config.schema_name

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Deploy Model Serving Endpoint
# MAGIC Here, we deploy a model serving endpoint for the `hotel_reservation_cremerf_pyfunc`.

# COMMAND ----------

workspace.serving_endpoints.create(
    name="hotel-reservations-cremerf-model-serving",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.hotel_reservation_cremerf_pyfunc",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=21,
            )
        ],
        # Optional if only 1 entity is served
        traffic_config=TrafficConfig(
            routes=[Route(served_model_name="hotel_reservation_cremerf_pyfunc-21", traffic_percentage=100)]
        ),
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Call the endpoint

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Create Sample Request Body
# MAGIC Here, we sample records from the training set to create the request body for calling the model endpoint.

# COMMAND ----------

required_columns = [
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "required_car_parking_space",
    "lead_time",
    "arrival_year",
    "arrival_month",
    "arrival_date",
    "repeated_guest",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "avg_price_per_room",
    "no_of_special_requests",
    "type_of_meal_plan",
    "room_type_reserved",
    "market_segment_type",
]

sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

start_time = time.time()

model_serving_endpoint = f"https://{host}/serving-endpoints/hotel-reservations-cremerf-model-serving/invocations"
response = requests.post(
    f"{model_serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": dataframe_records[4]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load Test
# MAGIC We send multiple concurrent requests to the endpoint to measure latency and evaluate performance.
# MAGIC

# COMMAND ----------

# Initialize variables
model_serving_endpoint = f"https://{host}/serving-endpoints/hotel-reservations-cremerf-model-serving/invocations"

headers = {"Authorization": f"Bearer {token}"}
num_requests = 1000

# COMMAND ----------


# Function to make a request and record latency
def send_request():
    random_record = random.choice(dataframe_records)
    start_time = time.time()
    response = requests.post(
        model_serving_endpoint,
        headers=headers,
        json={"dataframe_records": random_record},
    )
    end_time = time.time()
    latency = end_time - start_time
    return response.status_code, latency


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
