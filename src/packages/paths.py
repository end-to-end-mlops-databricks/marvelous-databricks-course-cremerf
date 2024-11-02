import yaml
from pathlib import Path
import os

class AllPaths:
    def __init__(self) -> None:
        if self.is_databricks():
            # Running in Databricks
            self.filename_config = Path('/Workspace/Users/cremerfederico29@gmail.com/marvelmlops-cremerf/project-config.yml')
        else:
            # Running in VSCode or other local environment
            try:
                self.BASE_DIR = Path(__file__).resolve().parent.parent.parent
            except NameError:
                # __file__ is not defined, use current working directory
                self.BASE_DIR = Path.cwd()
            self.filename_config = self.BASE_DIR / 'project-config.yml'
        
        self.config = self.get_config_file()
        self.cremerf_catalog = f'/Volumes/{self.config["catalog_name"]}/{self.config["schema_name"]}/'
        self.data_volume = f"{self.cremerf_catalog}" + "data/"
    
    def is_databricks(self):
        # Check if the /databricks directory exists
        return os.path.exists('/databricks')
    
    def get_config_file(self):
        # Load configuration
        try:
            with open(self.filename_config, "r") as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            print(f"Configuration file not found at {self.filename_config}")
            raise

print('SIIIII')

