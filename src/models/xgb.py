from models.base_model import AutoAudioBaseModel
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np


class AudioGB(AutoAudioBaseModel):
    def __init__(self, random_state: int):
        self.model = GradientBoostingClassifier(random_state=random_state)

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame):
        self.model.fit(features, labels)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        predictions = self.model.predict(features)

        predictions = np.array(predictions)
        return predictions

    def __str__(self) -> str:
        return "Gradient Boosting"
