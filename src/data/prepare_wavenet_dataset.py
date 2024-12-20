import os
import numpy as np
import pandas as pd
import deeplake
import librosa

# Directories
PROCESSED_DIR = "data/processed/wavenet_data/"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Constants
SAMPLE_RATE = 16000
CHUNK_SIZE = SAMPLE_RATE // 2  # 0.5-second chunks for better handling
CONDITIONING_FEATURES_FILE = "src/notebooks/processed_features.csv"
BATCH_SIZE = 500  # Process 500 samples per batch

# Load the NSynth dataset
ds = deeplake.load("hub://activeloop/nsynth-train")

# Load conditioning features
features_df = pd.read_csv(CONDITIONING_FEATURES_FILE)
features_df["features"] = features_df["features"].apply(eval)

# Batch processing
batch_data = []
batch_count = 0

for i, sample in enumerate(ds[:len(features_df)]):  # Process samples based on feature dataframe size
    try:
        # Load raw audio waveform
        audio_waveform = sample["audios"].numpy()
        
        # Normalize waveform between -1 and 1
        audio_waveform = audio_waveform / np.max(np.abs(audio_waveform))
        
        # Split into chunks
        num_chunks = len(audio_waveform) // CHUNK_SIZE
        for j in range(num_chunks):
            start = j * CHUNK_SIZE
            end = start + CHUNK_SIZE
            chunk = audio_waveform[start:end]
            
            if len(chunk) < CHUNK_SIZE:
                continue
            
            conditioning_features = np.array(features_df.iloc[i].features)
            
            batch_data.append({
                "audio_chunk": chunk.tolist(),
                "conditioning_features": conditioning_features.tolist(),
                "label": features_df.iloc[i].instrument_family
            })

            # Save batch to CSV when BATCH_SIZE is reached
            if len(batch_data) == BATCH_SIZE:
                output_file = os.path.join(PROCESSED_DIR, f"wavenet_dataset_batch_{i // BATCH_SIZE + 1}.csv")
                pd.DataFrame(batch_data).to_csv(output_file, index=False)
                batch_data = []  # Clear the batch

        # Log progress
        if i % 500 == 0:
            print(f"Processed {i} audio samples...")
    
    except Exception as e:
        print(f"Error processing sample {i}: {e}")

# Save any remaining data after the loop
if batch_data:
    output_file = os.path.join(PROCESSED_DIR, f"wavenet_dataset_batch_final.csv")
    pd.DataFrame(batch_data).to_csv(output_file, index=False)
    print(f"Final batch saved with {len(batch_data)} samples.")