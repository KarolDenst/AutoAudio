from models.base_model import AutoAudioBaseModel
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np


class AudioKNN(AutoAudioBaseModel):
    def __init__(self, n_neighbors: int):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame):
        self.model.fit(features, labels)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        predictions = self.model.predict(features)

        return predictions

    def __str__(self) -> str:
        return "KNN"
