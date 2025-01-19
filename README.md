# Description

This package offers a way to automatically train audio classification models.
To see examples of how to use the package, please refer to the examples folder.

# Installation

To install the package, run:
`pip install .` in the root directory of the package.

The package was tested with Python 3.12.4.

# Usage

To use the package first set up a dataframe with the following columns: file_path, label. The file_path column should contain the path to the audio file and the label column should contain the label of the audio file. Then, run the following code:

```python
from auto_audio.auto_audio_model import AutoAudioModel
model = AutoAudioModel()
model.fit(df_train, time_limit=600, random_state=42)
y_pred = model.predict(df_test)
```

## Parameters

The `AutoAudioModel` constructor only takes one paraemter, `log` which if set to `False` will stop logging.
The fit method takes the following parameters:

- time_limit which is the time limit in seconds for the model to train.
- random_state which is the random state for the model.
