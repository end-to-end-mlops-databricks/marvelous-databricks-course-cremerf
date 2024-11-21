# Week 3 Materials and Code Overview

Hello students,

Before diving into the Week 3 materials and code, let's clarify the key concepts and implementations covered in this lecture.

## Overview

Last week we demonstrated model training and registering for different use cases.
This week, we show three different serving endpoint creation for different scenarios. Feature serving, model serving and model serving with feature look up.

## Code Structure and Implementations

### 1. Feature Serving
```
01.feature_serving.py
```
Steps:
* The process begins by loading both the training and testing datasets, which are then concatenated into a single DataFrame. Subsequently, we load a pre-registered Scikit-learn model for generating predictions and select the features to be used for serving.

* Using the loaded model, we generate predictions for our dataset, resulting in a final DataFrame that includes 4 features, one of which is the predicted column. This DataFrame is then utilized to create a feature table in the form of a Delta table.

* Next, we establish an online feature table by using the previously created offline feature table as the source, which is also a Delta table. This setup enables the creation of an online table that relies on the feature Delta table crafted in the preceding steps.

* To create serving endpoints, it's essential to create a feature spec based on the feature table. This specification defines the source feature Delta table, allowing the feature spec to support both offline and online scenarios. For online lookups, the serving endpoint automatically utilizes the online table to execute low-latency feature retrievals. Both the source Delta table and the online table share the same primary key.

* Finally, we create a serving endpoint by using the feature spec as serving entity. This endpoint can be used for online feature lookups. The subsequent code examples shows how to invoke this endpoint and get responses.

### 2. Model Serving
```
02.model_serving.py
```
Model serving is a process of creating a model serving endpoint that can be used for inference. Endpoint creation process is similar to feature serving, with the exception that we don't need to create a feature table. Instead, we simply create a model serving endpoint that relies on the model we trained.

Steps:
* We start with loading the trained and registered model.
* Then we create a model serving endpoint using the model. It's important to note that entity name we pass is a registered model name and the version is an existing model version.
* We also show an example of traffic split, which is a feature of model serving that allows us to split traffic between multiple model versions.
* Finally, we invoke the endpoint and get the predictions. The payload should be a JSON object that includes the same features used for training and values. We need to provide all the features required for prediction.
* We also added an example piece of code for simple load test to get average latency.

### 3. Model Serving with Feature Look Up
```
03.model_serving_feature_lookup.py
```

This is a combination of the previous two examples. We load a pre-trained model and create a feature table for look up. Then we create a model serving endpoint that uses the feature table. Last week, we trained a model with feature lookup and feature func. Now we will create a serving endpoint for that model.

Steps:
- We start with creating an online table for existing offline feature table, house_features. This is the table we created last week on *week 2 - 05.log_and_register_fe_model.py* notebook.
- This online table is required for our model to look up features at serving endpoint.
- Next is the same as in the previous notebook, we create an endpoint using the model we registred in the same notebook *week 2 - 05.log_and_register_fe_model.p*. This is the model we registred using feature lookup and feature func.
- When we send request to the model endpoint, this time, we won't need to provide all the features. 3 features will be taken from the feature lookup table, also one feature "house_age" will be calculated by the feature function.


### 4. A/B Testing
```
04.AB_test_model_serving.py
```
In this notebook, we show the setup of A/B testing for two different model versions using Pyfunc. We'll train, register, and implement a serving endpoint that uses a Pyfunc model as a wrapper for these versions.

Steps:
- We start with loading the configurations, parameters for model A and model B, training and testing datasets.
- We use the same approach as we did in *week 2 - 03.log_and_register_model.py*.
- We train the model A and model B, and register them.
- After training, we create aliases for each model version, referred to as `model_A` and `model_B`.
- These registered model versions are then loaded and used within a wrapper class.
- The wrapper class is where we define the A/B testing logic, controlling which data points receive predictions from which model. For this, we use hash function.
- We run an MLflow experiment with this wrapper model and register.
- The next steps involve creating a serving endpoint, similar to our previous examples, and shows how to invoke it.
