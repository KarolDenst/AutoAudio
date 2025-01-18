from models.base_model import AutoAudioBaseModel
from sklearn.svm import SVC
import pandas as pd
import numpy as np


class AudioSVM(AutoAudioBaseModel):
    def __init__(self, random_state: int):
        self.model = SVC(random_state=random_state)

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame):
        self.model.fit(features, labels)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        predictions = self.model.predict(features)

        return predictions

    def __str__(self) -> str:
        return "SVM"
