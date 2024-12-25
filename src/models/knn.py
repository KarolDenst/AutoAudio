from models.base_model import AutoAudioBaseModel
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np


class AudioKNN(AutoAudioBaseModel):
    def __init__(self):
        self.model = None

    def fit(self, features: pd.DataFrame, labels: np.ndarray):
        if self.model is None:
            n_neighbors = len(np.unique(labels))
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model.fit(features, labels)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained")
        predictions = self.model.predict(features)

        return predictions

    def __str__(self) -> str:
        return "KNN"
