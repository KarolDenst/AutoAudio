from models.base_model import AutoAudioBaseModel
from sklearn.svm import SVC
import pandas as pd
import numpy as np


class AudioSVM(AutoAudioBaseModel):
    def __init__(self):
        self.model = SVC()

    def fit(self, features: pd.DataFrame, labels: np.ndarray):
        self.model.fit(features, labels)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        predictions = self.model.predict(features)

        return predictions

    def __str__(self) -> str:
        return "SVM"
