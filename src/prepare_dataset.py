from packages.paths import AllPaths
from packages.preprocessing import Preprocessor
from packages.config import ProjectConfig
from datetime import datetime
import pandas as pd
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

ALLPATHS = AllPaths()

# COMMAND ----------

config = ProjectConfig.from_yaml(config_path=ALLPATHS.filename_config)

# COMMAND ----------
# Load the house prices dataset
df = spark.read.csv(
    f'{ALLPATHS.data_volume}/hotel_reservations.csv',
    header=True,
    inferSchema=True).toPandas()

# COMMAND ----------
data_processor = Preprocessor(pandas_df=df, config=config)
data_processor.preprocess_raw_data()
train_set, test_set = data_processor.split_data()
data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)