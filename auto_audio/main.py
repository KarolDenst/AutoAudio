import preprocessing as pre
from auto_audio_model import AutoAudioModel
import os
import pandas as pd

directory = "data/"
file_paths = []
labels = []
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        file_path = os.path.join(directory, filename)
        label = int(filename[0])
        file_paths.append(file_path)
        labels.append(str(label))

data = pd.DataFrame({"file_path": file_paths, "label": labels})

features = pre.aggregate_audio_features(data)

model = AutoAudioModel()
model.fit(data, time_limit=500)
predictions = model.predict(data)
print(predictions)
