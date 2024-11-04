# Databricks notebook source
dbutils.library.restartPython()

# COMMAND ----------

import yaml
from databricks import feature_engineering
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
from packages.config import ProjectConfig
from packages.paths import AllPaths
from packages.classifier import CancellationModel
from mlflow_train import CancellatioModelWrapper
import mlflow
from pyspark.sql import functions as F
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from databricks.feature_engineering import FeatureFunction, FeatureLookup

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
target = config.id_field
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.hotel_features"
function_name = f"{catalog_name}.{schema_name}.calculate_total_guests"


# COMMAND ----------

# Load training and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")


# COMMAND ----------

from pandas.api.types import CategoricalDtype
import numpy as np
import pandas as pd

df = spark.read.csv(
    f'{ALLPATHS.data_volume}/hotel_reservations.csv',
    header=True,
    inferSchema=True).toPandas()

non_zero_values = df['avg_price_per_room'][(df['avg_price_per_room'] != 0) & (~df['avg_price_per_room'].isna())]
median_value = non_zero_values.median()
df['avg_price_per_room'] = df['avg_price_per_room'].replace(0, np.nan)
df['avg_price_per_room'] = df['avg_price_per_room'].fillna(median_value)

df[config.target] = df[config.target].map({'Not_Canceled': 0, 'Canceled': 1})

# Handle numeric features
num_features = config.num_features
for col in num_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle categorical features
cat_features = config.cat_features
for cat_col in cat_features:
    df[cat_col] = df[cat_col].astype('category')

for col in cat_features:
    # Ensure the column is of type 'category'
    if not isinstance(df[col].dtype, CategoricalDtype):
        df[col] = df[col].astype('category')
    
    # Add 'Unknown' to categories if not already present
    if 'Unknown' not in df[col].cat.categories:
        df[col] = df[col].cat.add_categories(['Unknown'])
    
    # Fill NaN values with 'Unknown'
    df[col] = df[col].fillna('Unknown')

# Extract target and relevant features
        # Extract target and relevant features
id_field = config.id_field
target = config.target
relevant_columns = cat_features + num_features + [target] + [id_field]
df = df[relevant_columns]

# COMMAND ----------

target = config.target
target

# COMMAND ----------

spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas().dtypes

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE TABLE {feature_table_name}(
        booking_id STRING NOT NULL,
        lead_time INT,
        no_of_special_requests INT,
        avg_price_per_room FLOAT);
    """)

spark.sql(f"""ALTER TABLE {feature_table_name}
          ADD CONSTRAINT hotel_pk PRIMARY KEY(booking_id);""")

spark.sql(f"""ALTER TABLE {feature_table_name}
          SET TBLPROPERTIES (delta.enableChangeDataFeed = true);""")

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
