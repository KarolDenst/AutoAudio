import librosa
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


# TODO: This fails for corrupted files. Fix it
def aggregate_audio_features(
    file_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_mfcc = 13
    sr = 16000
    filenames = file_df["file_path"].tolist()

    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(lambda f: extract_audio_features(f, n_mfcc, sr), filenames)
        )

    features, ys = zip(*results)

    df = pd.DataFrame(features, index=pd.Index(filenames))

    audios = pd.DataFrame(
        {
            "audio": ys,
            "label": file_df["label"].tolist() if "label" in file_df.columns else None,
            "sampling_rate": sr,
        },
        index=pd.Index(filenames),
    ).dropna(axis=1, how="all")  # Drop 'label' if it wasn't included

    return df, audios


def extract_audio_features(
    file_path: str, n_mfcc: int = 13, sr: int = 16000
) -> tuple[dict, np.ndarray]:
    y, sr = librosa.load(file_path, sr=sr)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    spectral_centroid_mean = np.mean(spectral_centroid, axis=1)
    spectral_centroid_std = np.std(spectral_centroid, axis=1)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth, axis=1)
    spectral_bandwidth_std = np.std(spectral_bandwidth, axis=1)
    spectral_rolloff_mean = np.mean(spectral_rolloff, axis=1)
    spectral_rolloff_std = np.std(spectral_rolloff, axis=1)
    zcr_mean = np.mean(zcr, axis=1)
    zcr_std = np.std(zcr, axis=1)

    features = {
        "spectral_centroid_mean": spectral_centroid_mean[0],
        "spectral_centroid_std": spectral_centroid_std[0],
        "spectral_bandwidth_mean": spectral_bandwidth_mean[0],
        "spectral_bandwidth_std": spectral_bandwidth_std[0],
        "spectral_rolloff_mean": spectral_rolloff_mean[0],
        "spectral_rolloff_std": spectral_rolloff_std[0],
        "zcr_mean": zcr_mean[0],
        "zcr_std": zcr_std[0],
        **{f"mfccs_{i}": mfccs_mean[i] for i in range(len(mfccs_mean))},
        **{f"mfccs_{i}": mfccs_std[i] for i in range(len(mfccs_std))},
    }

    return features, y
