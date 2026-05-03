import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

class Model:  
    def __init__(self,
                 model_path: str = None,
                 model: Pipeline = None):
        self.model = self.load(model_path) if model is None else model
        
    def predict(self, X: pd.DataFrame):
        """Prediction."""
        return self.model.predict(X)

    def save(self, path: str):
        """Save model to directory.

        Args:
            path (str): save path for model.
        """
        if self.model is None:
            raise ValueError("No model for saving!")
        joblib.dump(self.model, path)

    def load(self, path: str):
        """Load model from path.

        Args:
            path (str): path for loading model.

        Returns:
            Pipeline: model.
        """
        self.model = joblib.load(path)
        return self