import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from mlflow.models import infer_signature
from packages.config import ProjectConfig
from packages.paths import AllPaths
import json
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from mlflow import MlflowClient
from mlflow.utils.environment import _mlflow_conda_env


ALLPATHS = AllPaths()

config = ProjectConfig.from_yaml(config_path=ALLPATHS.filename_config)

spark = SparkSession.builder.getOrCreate()

class CancellationModel():
        def __init__(self, config, preprocessor, classifier) -> None:
            self.config = config
            self.config_dict = config.dict()
            self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier(**self.config_dict['parameters']))
        ])
        
        def train(self, X_train, y_train):
            """
            Train the model using the training data.

            Parameters:
            X_train (pandas.DataFrame): The training features.
            y_train (pandas.Series): The target variable for training.
            """
            self.pipeline.fit(X_train, y_train)  # Fit the model

        def predict(self, X):
            """
            Make predictions using the trained model.

            Parameters:
            X (pandas.DataFrame): The input features for prediction.

            Returns:
            numpy.ndarray: Predicted labels for the input features.
            """
            return self.pipeline.predict(X)  # Return predictions

        def evaluate(self, X_test, y_test):
            """
            Evaluate the model using test data and return accuracy and classification report.

            Parameters:
            X_test (pandas.DataFrame): The test features.
            y_test (pandas.Series): The true labels for the test set.

            Returns:
            tuple: (accuracy (float), class_report (str))
                accuracy: Accuracy score of the model.
                class_report: Classification report as a string.
            """
            # Encode the test set target variable
            # y_test_encoded = self.label_encoder.transform(y_test)  # Optional: Encode target variable

            y_pred = self.predict(X_test)  # Predict on test set
            accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
            class_report = classification_report(y_test, y_pred)  # Generate classification report
            return accuracy, class_report  # Return results

        def get_feature_importance(self):
            """
            Retrieve feature importances from the trained model.

            Returns:
            tuple: (feature_importance (numpy.ndarray), feature_names (numpy.ndarray))
                feature_importance: Array of feature importances.
                feature_names: Names of the features used in the model.
            """
            feature_importance = self.pipeline.named_steps["classifier"].feature_importances_  # Get feature importances
            feature_names = self.pipeline.named_steps["preprocessor"].get_feature_names_out()  # Get feature names
            return feature_importance, feature_names  # Return importances and names

        def get_confusion_matrix(self, y_test, y_pred):
            """
            Generate and return confusion matrix and display.

            Parameters:
            y_test (pandas.Series): The true labels for the test set.
            y_pred (numpy.ndarray): The predicted labels from the model.

            Returns:
            tuple: (disp (ConfusionMatrixDisplay), cm (numpy.ndarray))
                disp: ConfusionMatrixDisplay object for visualizing the confusion matrix.
                cm: Confusion matrix as a 2D array.
            """
            cm = confusion_matrix(y_test, y_pred)  # Compute confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.pipeline.classes_)  # Prepare display
            return disp, cm  # Return display and confusion matrix
