import librosa
import numpy as np
import pandas as pd


def aggregate_audio_features(file_df: pd.DataFrame) -> pd.DataFrame:
    n_mfcc = 13

    features = []
    filenames = file_df["file_path"].tolist()
    for file_path in filenames:
        feature = extract_audio_features(file_path, n_mfcc)
        features.append(feature)

    df = pd.DataFrame(features, index=pd.Index(filenames))
    return df


def extract_audio_features(file_path: str, n_mfcc: int = 13) -> dict:
    y, sr = librosa.load(file_path)

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

    return features
