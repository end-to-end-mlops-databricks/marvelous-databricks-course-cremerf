# MLFlow
import mlflow

# Overall
import pandas as pd

class CancellatioModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, model_input, return_proba=False):
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        # Prediction based on specified mode
        if return_proba:
            # Predict probabilities for each class
            probabilities = self.model.predict_proba(model_input)
            predictions = {
                "Probabilities": probabilities.tolist(),
                "Predicted Class": probabilities.argmax(axis=1).tolist(),
            }
        else:
            # Predict class labels directly
            predicted_classes = self.model.predict(model_input)
            predictions = {"Predicted Class": predicted_classes.tolist()}

        return predictions
