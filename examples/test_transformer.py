import os
import subprocess
import pandas as pd

dataset_path = "data/gtzan-dataset-music-genre-classification"

if not os.path.exists(dataset_path):
    print("Dataset not found. Downloading...")
    os.makedirs(dataset_path, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "andradaolteanu/gtzan-dataset-music-genre-classification",
            "-p",
            dataset_path,
            "--unzip",
        ]
    )
    print("Download complete.")
else:
    print("Dataset already exists.")

genres_path = os.path.join(dataset_path, "Data/genres_original")
paths = []
labels = []
for genre in os.listdir(genres_path):
    folder_path = os.path.join(genres_path, genre)
    for filename in os.listdir(folder_path):
        paths.append(os.path.join(folder_path, filename))
        labels.append(genre)
df = pd.DataFrame({"file_path": paths, "label": labels})

import sys
import os

sys.path.insert(0, os.path.abspath("../src"))


from models.transformer import AudioTransformer
import preprocessing as pre
from sklearn.model_selection import train_test_split
import numpy as np

df_train = df.sample(100, random_state=42)
df_test = df.sample(100, random_state=42)
data = df_train
    
data.reset_index(drop=True, inplace=True)
features, audios = pre.aggregate_audio_features(data)
features.reset_index(drop=True, inplace=True)
audios.reset_index(drop=True, inplace=True)

labels = data["label"]
unique = np.unique(labels)
n_unique = len(unique)

label2id = {}
id2label = {}
for i, label in enumerate(unique):
    label2id[label] = str(i)
    id2label[str(i)] = label

test_size = 0.2
indices = labels.index
train_indices, test_indices = train_test_split(
    indices, test_size=test_size, random_state=42, shuffle=True
)
labels_train = labels.loc[train_indices].values.reshape(-1)
audios_train = audios.loc[train_indices]
labels_test = labels.loc[test_indices].values.reshape(-1)
audios_test = audios.loc[test_indices]

model = AudioTransformer(n_unique, label2id, id2label)
train_dataset = pd.DataFrame({"audio": audios_train, "label": labels_train})
test_dataset = pd.DataFrame({"audio": audios_test, "label": labels_test})
model.fit(train_dataset, test_dataset)


# Test

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

y_test = df_test["label"]
y_pred = model.predict(df_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
disp.plot()
plt.xticks(rotation=90)
plt.show()
