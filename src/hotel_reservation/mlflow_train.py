# MLFlow
import mlflow

# Overall
import pandas as pd


class CancellatioModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        # Set 'return_proba' to a default value
        return_proba = False  # Change to True if you want probabilities by default

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
