import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

class AllPaths:
    def __init__(self) -> None:
        load_dotenv()  # Load environment variables from .env file if present

        if self.is_databricks():
            # Running in Databricks
            self.filename_config = Path(
                os.environ.get(
                    "DATABRICKS_CONFIG_PATH",
                    "/Workspace/Users/cremerfederico29@gmail.com/marvelmlops-cremerf/project-config.yml",
                )
            )
        else:
            # Running in local environment
            self.BASE_DIR = Path(__file__).resolve().parent.parent.parent
            self.filename_config = self.BASE_DIR / "project-config.yml"

        self.config = self.get_config_file()
        self.cremerf_catalog = f'/Volumes/{self.config["catalog_name"]}/{self.config["schema_name"]}/'
        self.data_volume = f"{self.cremerf_catalog}data/"

    def is_databricks(self):
        is_db_env = os.environ.get("IS_DATABRICKS")
        if is_db_env is not None:
            # Use the value from the environment variable
            return is_db_env.lower() == "true"
        else:
            # Fallback to detecting Databricks environment
            return "DATABRICKS_RUNTIME_VERSION" in os.environ

    def get_config_file(self):
        # Load configuration
        try:
            with open(self.filename_config, "r") as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            print(f"Configuration file not found at {self.filename_config}")
            raise
