from models.base_model import AutoAudioBaseModel
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np


class AudioGB(AutoAudioBaseModel):
    def __init__(self):
        self.model = GradientBoostingClassifier()

    def fit(self, features: pd.DataFrame, labels: np.ndarray):
        self.model.fit(features, labels)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        predictions = self.model.predict(features)

        predictions = np.array(predictions)
        return predictions

    def __str__(self) -> str:
        return "Gradient Boosting"
