# Week 5 materials - Code Overview
Hello students,

Before diving into the Week 5 materials and code, let's clarify the key concepts and implementations covered in this lecture.

This file provides an overview of the scripts used in our house price prediction pipeline. These scripts manage data ingestion, model training, evaluation, and deployment in Databricks.
In this example, we’re simulating a real-world scenario where a machine learning model is served based on a dataset that receives new rows each week. To maintain high performance, we need to regularly evaluate the model’s effectiveness on the latest data and decide if an update is necessary.


## Overview

We implement a house price prediction pipeline using Databricks workflows, with different steps for data preprocessing, model training, evaluation, and deployment. Each task is executed sequentially, and certain conditions determine whether a task will trigger the next. In this lecture, we used the model with feature look up.

## Code Structure and Implementations

### 1. Data Ingestion and Updating Tables
- **Script**: `preprocess.py`
- **Description**: Handles data ingestion, filtering new records, and updates train/test datasets.
- **Key Steps**:
  1. Loads the source dataset and retrieves recent records by comparing timestamps.
  2. Splits new data into train and test sets (80-20 split).
  3. Appends the processed train and test data to existing tables.
  4. Updates the feature table with the latest data for serving.
  5. Triggers an online feature refresh pipeline and monitors its completion.
  6. Sets task values to indicate whether new data was processed.
- **Purpose**: Ensures the training and test datasets are up-to-date and that the feature table is refreshed with the latest values. We assume source_data has processed data, not raw data.

### 2. Model Training
- **Script**: `train_model.py`
- **Description**: Trains a LightGBM model on the house price data with engineered features.
- **Key Steps**:
  1. Loads the train and test datasets from Databricks.
  2. Performs feature engineering using the Databricks Feature Store, including calculating house age.
  3. Creates a training pipeline with LightGBM regressor.
  4. Tracks the training process and parameters in MLflow, logging metrics, artifacts, and model parameters.
  5. Outputs the model_uri for downstream tasks.
- **Purpose**: Builds and logs a new model for house prices using feature-engineered data.

### 3. Model Evaluation
- **Script**: `evaluate_model.py`
- **Description**: Evaluates the new model and compares it against the currently deployed model.
- **Key Steps**:
  1. Loads test data and applies feature engineering.
  2. Generates predictions using both the new and existing models.
  3. Calculates performance metrics, specifically Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).
  4. Compares metrics and decides whether to register the new model if it performs better.
  5. Sets task values to communicate results for downstream steps.
- **Purpose**: Ensures the new model performs better than the current model before it can be registered for production use.

### 4. Model Deployment
- **Script**: `deploy_model.py`
- **Description**: Deploys the selected model to a Databricks serving endpoint.
- **Key Steps**:
  1. Retrieves the model version determined by the evaluation step.
  2. Updates the serving endpoint with the specified model version.
- **Purpose**: Makes the new model available for real-time predictions via a Databricks serving endpoint.
