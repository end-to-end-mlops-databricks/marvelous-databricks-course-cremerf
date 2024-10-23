from pathlib import Path

import yaml


class AllPaths:
    def __init__(self, config) -> None:
        self.BASE_DIR = Path(__file__).resolve().parent.parent.parent

        self.config = config
        self.cremerf_catalog = f'/Volumes/{self.config["catalog_name"]}/{self.config["schema_name"]}/'
        self.data_volume = f"{self.cremerf_catalog}" + "data/"


# Load configuration
with open("/Users/fcremer29/focus/marvelmlops/project-config.yml", "r") as file:
    config = yaml.safe_load(file)

x = AllPaths(config=config)


print(x.BASE_DIR)
