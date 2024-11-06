import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from hotel_reservation.config import ProjectConfig


class Preprocessor:
    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig) -> None:
        self.config = config
        self.df = pandas_df

    def preprocess_raw_data(self):
        non_zero_values = self.df["avg_price_per_room"][
            (self.df["avg_price_per_room"] != 0) & (~self.df["avg_price_per_room"].isna())
        ]
        median_value = non_zero_values.median()
        self.df["avg_price_per_room"] = self.df["avg_price_per_room"].replace(0, np.nan)
        self.df["avg_price_per_room"] = self.df["avg_price_per_room"].fillna(median_value)

        self.df[self.config.target] = self.df[self.config.target].map({"Not_Canceled": 0, "Canceled": 1})

        # Handle numeric features
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Handle categorical features
        cat_features = self.config.cat_features
        for cat_col in cat_features:
            self.df[cat_col] = self.df[cat_col].astype("category")

        for col in cat_features:
            # Ensure the column is of type 'category'
            if not isinstance(self.df[col].dtype, CategoricalDtype):
                self.df[col] = self.df[col].astype("category")

            # Add 'Unknown' to categories if not already present
            if "Unknown" not in self.df[col].cat.categories:
                self.df[col] = self.df[col].cat.add_categories(["Unknown"])

            # Fill NaN values with 'Unknown'
            self.df[col] = self.df[col].fillna("Unknown")

        # Extract target and relevant features
        id_field = self.config.id_field
        target = self.config.target
        relevant_columns = cat_features + num_features + [target] + [id_field]
        self.df = self.df[relevant_columns]

    def split_data(self, test_size=0.2, random_state=42):
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)

        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
