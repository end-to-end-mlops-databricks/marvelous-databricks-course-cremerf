# Databricks notebook source
# MAGIC %md
# MAGIC ## 1. Set up MLflow for tracking and model registry

# COMMAND ----------

import time

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from lightgbm import LGBMClassifier
from mlflow import MlflowClient
from pyspark.dbutils import DBUtils
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import OneHotEncoder
import hashlib
import requests

from hotel_reservation.config import ProjectConfig
from hotel_reservation.paths import AllPaths

ALLPATHS = AllPaths()

# Set up MLflow for tracking and model registry
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Initialize the MLflow client for model management
client = MlflowClient()

# Load configuration
config = ProjectConfig.from_yaml(config_path=ALLPATHS.filename_config)

# Extract key configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
catalog_name = config.catalog_name
schema_name = config.schema_name
ab_test_params = config.ab_test
id_field = config.id_field

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load and Prepare Training and Testing Datasets

# COMMAND ----------

# Initialize a Databricks session for Spark operations
spark = SparkSession.builder.getOrCreate()

# Load the training and testing sets from Databricks tables
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = train_set_spark.toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# Define features and target variables
X_train = train_set[num_features + cat_features]
y_train = train_set[target]
X_test = test_set[num_features + cat_features]
y_test = test_set[target]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Train Model A and Log with MLflow
# MAGIC

# COMMAND ----------

# Set up specific parameters for model A as part of the A/B test
parameters_a = {
    "learning_rate": ab_test_params["learning_rate_a"],
    "n_estimators": ab_test_params["n_estimators_a"],
    "max_depth": ab_test_params["max_depth_a"],
}

# Define a preprocessor for categorical features, which will one-hot encode categorical variables
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)], remainder="passthrough"
)

# Build a pipeline combining preprocessing and model training steps
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**parameters_a))])

# COMMAND ----------

# Set the MLflow experiment to track this A/B testing project
mlflow.set_experiment(experiment_name="/Shared/hotel-reservations-cremerf-ab")
model_name = f"{catalog_name}.{schema_name}.hotel_reservations_model_ab"

# Git commit hash for tracking model version
git_sha = "9a9fe8a84a990e25056962cfc6269fa74b60638f"

# Start MLflow run to track training of Model A
with mlflow.start_run(tags={"model_class": "A", "git_sha": git_sha}) as run:
    run_id = run.info.run_id

    # Train the model
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    log_loss_value = log_loss(y_test, pipeline.predict_proba(X_test))

    # Log model parameters, metrics, and other artifacts in MLflow
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters_a)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("log_loss", log_loss_value)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log the input dataset for tracking reproducibility
    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")

    # Log the pipeline model in MLflow with a unique artifact path
    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="lightgbm-pipeline-model", signature=signature)

model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model", name=model_name, tags={"git_sha": f"{git_sha}"}
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Register Model A and Assign Alias

# COMMAND ----------

# Assign alias for easy reference in future A/B tests
model_version_alias = "model_A"

client.set_registered_model_alias(model_name, model_version_alias, f"{model_version.version}")
model_uri = f"models:/{model_name}@{model_version_alias}"
model_A = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Train Model B and Log with MLflow

# COMMAND ----------

# Set up specific parameters for model B as part of the A/B test
parameters_b = {
    "learning_rate": ab_test_params["learning_rate_b"],
    "n_estimators": ab_test_params["n_estimators_b"],
    "max_depth": ab_test_params["max_depth_b"],
}

# Repeat the training and logging steps for Model B using parameters for B
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**parameters_b))])

# Start MLflow run for Model B
with mlflow.start_run(tags={"model_class": "B", "git_sha": git_sha}) as run:
    run_id = run.info.run_id

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    log_loss_value = log_loss(y_test, pipeline.predict_proba(X_test))

    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters_b)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("log_loss", log_loss_value)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")
    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="lightgbm-pipeline-model", signature=signature)

model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model", name=model_name, tags={"git_sha": f"{git_sha}"}
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Register Model B and Assign Alias

# COMMAND ----------

# Assign alias for Model B
model_version_alias = "model_B"

client.set_registered_model_alias(model_name, model_version_alias, f"{model_version.version}")
model_uri = f"models:/{model_name}@{model_version_alias}"
model_B = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Define Custom A/B Test Model

# COMMAND ----------

# Define a custom A/B test model that selects the model based on hashed booking ID
class HotelReservationModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, models):
        self.models = models
        self.model_a = models[0]
        self.model_b = models[1]

    def predict(self, context, model_input):
        # Ensure model_input is a DataFrame and contains booking_id
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if "Booking_ID" not in model_input.columns:
            raise ValueError("The input DataFrame must contain a 'Booking_ID' column.")

        reservation_id = str(model_input["Booking_ID"].values[0])
        if not reservation_id:
            raise ValueError("The 'Booking_ID' value is missing.")

        hashed_id = hashlib.md5(reservation_id.encode(encoding="UTF-8")).hexdigest()

        # Determine which model to use based on hashed_id
        if int(hashed_id, 16) % 2 == 0:
            predictions = self.model_a.predict(model_input.drop(["Booking_ID"], axis=1))
            return {"Prediction": predictions[0], "model": "Model A"}
        else:
            predictions = self.model_b.predict(model_input.drop(["Booking_ID"], axis=1))
            return {"Prediction": predictions[0], "model": "Model B"}

# COMMAND ----------

X_train = train_set[num_features + cat_features + ["Booking_ID"]]
X_test = test_set[num_features + cat_features + ["Booking_ID"]]

# COMMAND ----------

models = [model_A, model_B]
wrapped_model = HotelReservationModelWrapper(models)  # pass loaded models to the wrapper
example_input = X_test.iloc[0:1]  # Select the first row for prediction as example
example_prediction = wrapped_model.predict(context=None, model_input=example_input)
print("Example Prediction:", example_prediction)

# COMMAND ----------

mlflow.set_experiment(experiment_name="/Shared/hotel-reservations-cremerf-ab-testing")
model_name = f"{catalog_name}.{schema_name}.hotel_reservations_model_pyfunc_ab_test"

with mlflow.start_run() as run:
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output={"Prediction": 0, "model": "Model B"})
    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")
    mlflow.pyfunc.log_model(
        python_model=wrapped_model, artifact_path="pyfunc-hotel-reservations-model-ab", signature=signature
    )
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/pyfunc-hotel-reservations-model-ab", name=model_name, tags={"git_sha": f"{git_sha}"}
)
# COMMAND ----------
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version.version}")

# Run prediction
predictions = model.predict(X_test.iloc[0:1])

# Display predictions
predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create serving endpoint

# COMMAND ----------

workspace = WorkspaceClient()

workspace.serving_endpoints.create(
    name="hotel-reservations-cremerf-model-serving-ab-test",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.hotel_reservations_model_pyfunc_ab_test",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=1,
            )
        ]
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Call the endpoint

# COMMAND ----------

dbutils = DBUtils(spark)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

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
    "Booking_ID",
]

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

start_time = time.time()

model_serving_endpoint = f"https://{host}/serving-endpoints/hotel-reservations-cremerf-model-serving-ab-test/invocations"

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
