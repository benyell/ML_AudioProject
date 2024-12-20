import deeplake
import librosa
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

PROCESSED_DIR = "data/processed/"
HEATMAP_DIR = os.path.join(PROCESSED_DIR, "heatmaps/")
os.makedirs(HEATMAP_DIR, exist_ok=True)

def extract_features(audio_waveform, sample_rate=16000):
    """Extract features from raw audio."""
    mfcc = librosa.feature.mfcc(y=audio_waveform, sr=sample_rate, n_mfcc=13).mean(axis=1)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_waveform, sr=sample_rate).mean()
    chroma = librosa.feature.chroma_stft(y=audio_waveform, sr=sample_rate).mean()
    return np.hstack((mfcc, spectral_centroid, chroma))

def create_heatmaps(audio_waveform, sample_name):
    """Generate and save a spectrogram as a heatmap."""
    S = librosa.feature.melspectrogram(y=audio_waveform, sr=16000, n_mels=128)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=16000, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    heatmap_path = os.path.join(HEATMAP_DIR, f"{sample_name}.png")
    plt.savefig(heatmap_path)
    plt.close()

def preprocess_nsynth():
    """Load and preprocess NSynth dataset."""
    # Load NSynth dataset from ActiveLoop
    dataset = deeplake.load("hub://activeloop/nsynth-train")
    processed_data = []

    for i, sample in enumerate(dataset[:1000]):  # Limit to 1000 samples for testing
        # Extract audio waveform and metadata
        audio_waveform = sample["audios"].numpy()
        instrument_family = sample["instrument_family"].numpy()  # Label
        instrument_source = sample["instrument_source"].numpy()  # Acoustic, electronic, or synthetic
        sample_name = f"sample_{i}"

        # Extract features
        features = extract_features(audio_waveform)

        # Save heatmap
        create_heatmaps(audio_waveform, sample_name)

        # Append processed data
        processed_data.append({
            "features": features.tolist(),
            "label": int(instrument_family),  # Convert to int for CSV
            "source": int(instrument_source),  # Convert to int for CSV
        })

        if i % 100 == 0:
            print(f"Processed {i} samples...")

    # Save processed data to a CSV file
    df = pd.DataFrame(processed_data)
    df.to_csv(os.path.join(PROCESSED_DIR, "nsynth_features.csv"), index=False)

if __name__ == "__main__":
    preprocess_nsynth()
