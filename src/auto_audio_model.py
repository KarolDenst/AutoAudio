import pandas as pd
import numpy as np
from models.base_model import AutoAudioBaseModel
import preprocessing as pre
from models.svm import AudioSVM
from models.knn import AudioKNN
from models.xgb import AudioGB
from models.transformer import AudioTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch


class AutoAudioModel:
    def __init__(self):
        self.models = []
        self.best_model = None

    def _get_models(self, labels, random_state) -> list[AutoAudioBaseModel]:
        unique = np.unique(labels)
        n_unique = len(unique)

        label2id = {}
        id2label = {}
        for i, label in enumerate(unique):
            label2id[label] = i
            id2label[i] = label
        models = [
            AudioSVM(random_state),
            AudioKNN(n_unique),
            AudioGB(random_state),
        ]
        if torch.cuda.is_available():
            models.append(AudioTransformer(n_unique, label2id, id2label, random_state))
        else:
            print("Cuda not available. Not training transformer model.")
        return models

    # TODO: Make full use of the random state
    def fit(self, data: pd.DataFrame, random_state: int = 42):
        """
        Fit the model to the training data.

        Parameters:
        data (pd.DataFrame): A DataFrame containing two columns:
            - 'file_path': The full path to the audio file.
            - 'label': The label associated with the audio file.
        random_state (int): The random state for reproducibility.

        Raises:
        ValueError: If the DataFrame does not contain the required columns.
        """

        if not {"file_path", "label"}.issubset(data.columns):
            raise ValueError("DataFrame must contain 'file_path' and 'label' columns")

        (
            features_train,
            labels_train,
            audios_train,
            features_test,
            labels_test,
            audios_test,
        ) = self.prepare_datesets(data, random_state)

        best_accuracy = -1
        for model in self.models:
            print(f"Training {model}")
            if model.get_data_type() == "audio":
                model.fit_from_audio(audios_train)
                predictions = model.predict_from_audio(audios_test)
            else:
                model.fit(features_train, labels_train)
                predictions = model.predict(features_test)
            accuracy = accuracy_score(labels_test, predictions)
            print(f"{model} achieved {accuracy * 100}% accuracy.")

            if accuracy > best_accuracy:
                self.best_model = model
                best_accuracy = accuracy

        print("Finished training.")
        print(f"Best model is: {str(self.best_model)}")
        # TODO: Fine tune best model
        # TODO: Maybe use ensamble of models

    def prepare_datesets(self, data, random_state):
        data.reset_index(drop=True, inplace=True)
        # TODO: maybe get multiple sets of features and test on each one
        print("Preprocessing audio files.")
        features, audios, labels = pre.aggregate_audio_features(data)
        print("Finished preprocessing files.")
        features.reset_index(drop=True, inplace=True)
        audios.reset_index(drop=True, inplace=True)
        labels.reset_index(drop=True, inplace=True)
        # TODO: add some feature reduction maybe

        self.models = self._get_models(labels, random_state)

        test_size = 0.2
        indices = labels.index
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state, shuffle=True
        )
        features_train = features.loc[train_indices]
        labels_train = labels.loc[train_indices].values.reshape(-1)
        audios_train = audios.loc[train_indices]
        features_test = features.loc[test_indices]
        labels_test = labels.loc[test_indices].values.reshape(-1)
        audios_test = audios.loc[test_indices]

        return (
            features_train,
            labels_train,
            audios_train,
            features_test,
            labels_test,
            audios_test,
        )

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on the test data.

        Parameters:
        data (pd.DataFrame): A DataFrame containing one column:
            - 'file_path': The full path to the audio file.

        Returns:
        np.ndarray: The predicted labels.

        Raises:
        ValueError: If the model has not been trained or if the DataFrame does not contain the required column.
        """

        if self.best_model is None:
            raise ValueError("Model has not been trained")
        if "file_path" not in data.columns:
            raise ValueError("DataFrame must contain 'file_path' column")
        features, audios, _ = pre.aggregate_audio_features(data)
        if self.best_model.get_data_type() == "audio":
            return self.best_model.predict(audios)
        return self.best_model.predict(features)
