import logging

import yaml

from prepare_dataset import run_preprocessing
from src.packages.config import ProjectConfig
from src.packages.paths import AllPaths

ALLPATHS = AllPaths()


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run():
    config = ProjectConfig.from_yaml(config_path=ALLPATHS.filename_config)
    logger.info("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))

    # run_preprocessing(): context transforms / splits / save to UC
    run_preprocessing(config)
    logger.info("Preprocessing raw data finished.")


if __name__ == "__main__":
    run()
