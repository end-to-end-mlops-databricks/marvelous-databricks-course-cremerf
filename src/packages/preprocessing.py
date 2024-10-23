from packages.paths import AllPaths
from databricks.connect import DatabricksSession

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


class Preprocessor():
    def __init__(self, filename, config) -> None:

        self.config = config
        self.all_paths = AllPaths(config= self.config)
        self.df_input = self.csv_loader_raw_data()
        self.filename = filename
        self.X = None
        self.y = None
        self.preprocessor = None

    def csv_loader_raw_data(self, filename):
        return pd.read_csv(f'{self.all_paths.data_volume}/{filename}.csv')
    
    def preprocess_raw_data(self):

        target = self.config['target']
        self.df_input = self.df.dropna(subset=[target])

        self.X = self.df[self.config['num_features'] + self.config['cat_features']]
        self.y = self.df[target]

        # Preprocessing for numeric data: convert data types and scale
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', StandardScaler()) 
        ])

        # Preprocessing for categorical data: fill missing values and apply one-hot encoding
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.config['num_features']),
                ('cat', categorical_transformer, self.config['cat_features'])
            ],
            remainder='drop'  # This will drop other columns not listed explicitly
        )

    def split_data(self, test_size=0.2, random_state=42):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)




