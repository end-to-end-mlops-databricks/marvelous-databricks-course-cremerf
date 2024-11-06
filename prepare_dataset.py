# SDK
# Built-in
import logging

# Overall
from pyspark.sql import SparkSession

from src.packages.paths import AllPaths
from src.packages.preprocessing import Preprocessor

ALLPATHS = AllPaths()
spark = SparkSession.builder.getOrCreate()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_preprocessing(config):
    # Load the reservations dataset
    df = spark.read.csv(f"{ALLPATHS.data_volume}/hotel_reservations.csv", header=True, inferSchema=True).toPandas()

    data_processor = Preprocessor(pandas_df=df, config=config)
    data_processor.preprocess_raw_data()
    logger.info("Data preprocessed.")
    train_set, test_set = data_processor.split_data()
    logger.info("Data split into training and test sets.")
    data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)
