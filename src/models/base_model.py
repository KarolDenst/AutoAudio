from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class AutoAudioBaseModel(ABC):
    @abstractmethod
    def fit(self, features: pd.DataFrame, labels: np.ndarray):
        """Fit the model to the training data."""
        pass

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions on the test data."""
        pass

    def __str__(self) -> str:
        return "Base Model"
