import logging

import yaml

from src.packages.preprocessing import Preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run():
    # Load configuration
    with open("project-config.yml", "r") as file:
        config = yaml.safe_load(file)

    logger.info("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))

    # initialize Preprocessor
    data_processor = Preprocessor(filename="hotel_reservations", config=config)
    logger.info("DataProcessor initialized.")

    # execute sci-kit learn pipeline to perform preprocessing tasks
    data_processor.preprocess_raw_data()
    logger.info("Data preprocessed.")

    # Split the data
    X_train, X_test, y_train, y_test = data_processor.split_data()
    logger.info("Data split into training and test sets.")
    logger.debug(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")


if __name__ == "__main__":
    run()
