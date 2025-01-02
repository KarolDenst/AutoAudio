from models.base_model import AutoAudioBaseModel
import evaluate
import torch
from transformers import (
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
    AutoFeatureExtractor,
)
import pandas as pd
import numpy as np
import uuid


# https://huggingface.co/docs/transformers/en/tasks/audio_classification
class AudioTransformer(AutoAudioBaseModel):
    def __init__(self, num_labels: int, label2id: dict, id2label: dict):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base"
        )
        self.model = AutoModelForAudioClassification.from_pretrained(
            "facebook/wav2vec2-base",
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
        )
        self.id = str(uuid.uuid4())
        self.path = ("outputs/transformer" + self.id,)

    def fit(self, train_dataset, test_dataset):
        training_args = TrainingArguments(
            output_dir=self.path,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=32,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=32,
            num_train_epochs=10,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
        )

        accuracy = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            predictions = np.argmax(eval_pred.predictions, axis=1)
            return accuracy.compute(
                predictions=predictions, references=eval_pred.label_ids
            )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            processing_class=self.feature_extractor,
            compute_metrics=compute_metrics,
        )

        trainer.train()

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        # TODO: get features into correct format
        with torch.no_grad():
            logits = self.model(features["file_path"]).logits
            predicted_class_ids = torch.argmax(
                logits, dim=1
            ).item()  # TODO: check if dim is correct
            predicted_labels = self.model.config.id2label[predicted_class_ids]
        return predicted_labels

    def __str__(self) -> str:
        return "Transformer"
