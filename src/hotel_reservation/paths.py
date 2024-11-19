import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

class AllPaths:
    def __init__(self) -> None:
        load_dotenv()  # Load environment variables from .env file

        self.base_dir = self.get_base_dir()
        self.filename_config = self.base_dir / "project-config.yml"

        self.config = self.get_config_file()
        self.cremerf_catalog = f'/Volumes/{self.config["catalog_name"]}/{self.config["schema_name"]}/'
        self.data_volume = f"{self.cremerf_catalog}data/"

    def get_base_dir(self):
        if self.is_databricks():
            # In Databricks, the current working directory is accessible via os.getcwd()
            return Path(os.getcwd())
        else:
            # In local environment, use the parent of the script's directory
            return Path(__file__).resolve().parent.parent.parent

    def is_databricks(self):
        # Use the IS_DATABRICKS environment variable from .env
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
