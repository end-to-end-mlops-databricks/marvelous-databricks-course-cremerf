import pandas as pd
import numpy as np
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from hotel_reservation.config import ProjectConfig
from hotel_reservation.paths import AllPaths

ALLPATHS = AllPaths()

config = ProjectConfig.from_yaml(config_path=ALLPATHS.filename_config)


catalog_name = config.catalog_name
schema_name = config.schema_name

# Load training and test sets from Catalog
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
combined_set = pd.concat([train_set, test_set], ignore_index=True)

import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype

def create_synthetic_data(df, num_rows=100):
    synthetic_data = pd.DataFrame()

    for column in df.columns:
        if column == 'Booking_ID':
            continue  # We'll handle Booking_ID separately at the end

        if pd.api.types.is_numeric_dtype(df[column]):
            if column in ['arrival_year', 'arrival_month', 'arrival_date']:
                # Generate integer values within the existing range
                synthetic_data[column] = np.random.randint(df[column].min(), df[column].max() + 1, num_rows)
            else:
                mean, std = df[column].mean(), df[column].std()
                synthetic_data[column] = np.random.normal(mean, std, num_rows)
                if pd.api.types.is_integer_dtype(df[column]):
                    synthetic_data[column] = synthetic_data[column].round().astype(int)
                else:
                    synthetic_data[column] = synthetic_data[column].astype(df[column].dtype)

        elif isinstance(df[column].dtype, CategoricalDtype) or pd.api.types.is_object_dtype(df[column]):
            synthetic_data[column] = np.random.choice(
                df[column].unique(), num_rows, p=df[column].value_counts(normalize=True)
            )
        else:
            synthetic_data[column] = np.random.choice(df[column], num_rows)

    # Ensure no negative values for counts and other logical constraints
    for col in ['required_car_parking_space', 'no_of_adults', 'no_of_children',
                'no_of_weekend_nights', 'no_of_week_nights', 'lead_time',
                'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
                'no_of_special_requests']:
        if col in synthetic_data.columns:
            synthetic_data[col] = synthetic_data[col].abs()
            synthetic_data[col] = synthetic_data[col].round().astype(int)

    # Handle 'avg_price_per_room' to ensure it's positive
    if 'avg_price_per_room' in synthetic_data.columns:
        synthetic_data['avg_price_per_room'] = synthetic_data['avg_price_per_room'].abs()

    # Generate unique Booking_IDs
    existing_ids = set(df['Booking_ID'])
    new_ids = []
    while len(new_ids) < num_rows:
        new_id = 'Synthetic_' + str(np.random.randint(1e6, 1e7))
        if new_id not in existing_ids:
            new_ids.append(new_id)
            existing_ids.add(new_id)
    synthetic_data['Booking_ID'] = new_ids

    # Reorder columns to match the original DataFrame
    synthetic_data = synthetic_data[df.columns]

    return synthetic_data

synthetic_df = create_synthetic_data(combined_set)

existing_schema = spark.table(f"{catalog_name}.{schema_name}.source_data").schema

synthetic_spark_df = spark.createDataFrame(synthetic_df, schema=existing_schema)

train_set_with_timestamp = synthetic_spark_df.withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

# Append synthetic data as new data to source_data table
train_set_with_timestamp.write.mode("append").saveAsTable(
    f"{catalog_name}.{schema_name}.source_data"
)



