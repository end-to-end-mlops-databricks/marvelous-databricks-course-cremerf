# COMMAND ----------

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from hotel_reservation.config import ProjectConfig
from hotel_reservation.paths import AllPaths

# COMMAND ----------

# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# COMMAND ----------

ALLPATHS = AllPaths()

config = ProjectConfig.from_yaml(config_path=ALLPATHS.filename_config)

# COMMAND ----------

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
id_field = config.id_field
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.hotel_features"

loyalty_function_name = f"{catalog_name}.{schema_name}.calculate_loyalty_score"
# days_until_arrival_function_name = f"{catalog_name}.{schema_name}.calculate_days_until_arrival"
# holiday_season_function_name = f"{catalog_name}.{schema_name}.is_holiday_season"

# COMMAND ----------

# Load training and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

# COMMAND ----------

# Create or replace the hotel_features table
spark.sql(f"""
    CREATE OR REPLACE TABLE {feature_table_name}(
        Booking_ID STRING NOT NULL,
        lead_time INT,
        no_of_special_requests INT,
        avg_price_per_room FLOAT);
    """)

# COMMAND ----------

spark.sql(f"""ALTER TABLE {feature_table_name}
          ADD CONSTRAINT hotel_pk PRIMARY KEY(Booking_ID);""")

spark.sql(f"""ALTER TABLE {feature_table_name}
          SET TBLPROPERTIES (delta.enableChangeDataFeed = true);""")


# COMMAND ----------

# Insert data into the feature table from both train and test sets
spark.sql(f"""
        INSERT INTO {feature_table_name}
        SELECT
            booking_id, lead_time, no_of_special_requests, avg_price_per_room
        FROM {catalog_name}.{schema_name}.train_set
        """)
spark.sql(f"""
        INSERT INTO {feature_table_name}
        SELECT
            booking_id, lead_time, no_of_special_requests, avg_price_per_room
        FROM {catalog_name}.{schema_name}.test_set""")

# COMMAND ----------

# COMMAND ----------
spark.sql(f"""

CREATE OR REPLACE FUNCTION {loyalty_function_name}(
    no_of_previous_cancellations DOUBLE,
    no_of_previous_bookings_not_canceled DOUBLE
)
RETURNS DOUBLE
LANGUAGE PYTHON AS
$$
    # Define weightings
    w1 = 1.5     # Weight the number of times a previous booking was NOT cancelled
    w2 = 1.0     # Weight the number of times a previous booking was cancelled

    # Calculate loyalty score
    loyalty_score = (w1 * no_of_previous_bookings_not_canceled) - (w2 * no_of_previous_cancellations)
    return loyalty_score
$$
""")

# COMMAND ----------

# Load training and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").drop(
    "lead_time", "no_of_special_requests", "avg_price_per_room"
)
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# Cast YearBuilt to int for the function input
# Cast relevant columns to double for the function input
train_set = train_set.withColumn(
    "no_of_previous_bookings_not_canceled", train_set["no_of_previous_bookings_not_canceled"].cast("double")
)
train_set = train_set.withColumn(
    "no_of_previous_cancellations", train_set["no_of_previous_cancellations"].cast("double")
)
train_set = train_set.withColumn("Booking_ID", train_set["Booking_ID"].cast("string"))

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
            input_bindings={
                "no_of_previous_cancellations": "no_of_previous_cancellations",
                "no_of_previous_bookings_not_canceled": "no_of_previous_bookings_not_canceled",
            },
        ),
    ],
)

# COMMAND ----------

# Load feature-engineered DataFrame
training_df = training_set.load_df().toPandas()

# COMMAND ----------

test_set["loyalty_score"] = (test_set["no_of_previous_bookings_not_canceled"] * 1.5) + (
    test_set["no_of_week_nights"] * 1.0
)

# COMMAND ----------

# Split features and target
X_train = training_df[num_features + cat_features + ["loyalty_score"]]
y_train = training_df[target]
X_test = test_set[num_features + cat_features + ["loyalty_score"]]  # Ensure test set has the same features
y_test = test_set[target]

# COMMAND ----------

# Setup preprocessing and model pipeline
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)], remainder="passthrough"
)
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**parameters))])

# COMMAND ----------

# Set and start MLflow experiment
mlflow.set_experiment(experiment_name="/Shared/hotel-reservations-cremerf")
git_sha = "e0ec1a5902a2d8c60c434766c0013d47c3879237"


with mlflow.start_run(tags={"branch": "week2", "git_sha": f"{git_sha}"}) as run:
    run_id = run.info.run_id
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate and print metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    # Log model parameters, metrics, and model
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("accuracy", accuracy)

    # Log classification report metrics
    for class_label, metrics in report.items():
        if isinstance(metrics, dict):
            mlflow.log_metric(f"precision_{class_label}", metrics["precision"])
            mlflow.log_metric(f"recall_{class_label}", metrics["recall"])
            mlflow.log_metric(f"f1-score_{class_label}", metrics["f1-score"])

    # Log model with feature engineering
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    fe.log_model(
        model=pipeline,
        flavor=mlflow.sklearn,
        artifact_path="lightgbm-pipeline-model-fe",
        training_set=training_set,
        signature=signature,
        code_paths=['/Workspace/Users/cremerfederico29@gmail.com/.bundle/marvelmlops/dev/files/dist/marvelmlops-0.0.1-py3-none-any.whl'],
    )
mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model-fe",
    name=f"{catalog_name}.{schema_name}.hotel_reservations_model_fe",
)
