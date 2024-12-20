import pandas as pd
import numpy as np

# Path to the merged CSV file with null values
merged_file_path = "data/processed/wavenet_data/wavenet_dataset_merged.csv"

# Load the merged dataset
merged_data = pd.read_csv(merged_file_path)

# Convert the `audio_chunk` column to lists for filtering
merged_data["audio_chunk"] = merged_data["audio_chunk"].apply(eval)

# Remove rows where `audio_chunk` is null or contains all zeros
filtered_data = merged_data[
    merged_data["audio_chunk"].apply(lambda chunk: isinstance(chunk, list) and not np.all(np.array(chunk) == 0))
]

# Path to save the filtered dataset
filtered_file_path = "data/processed/wavenet_data/wavenet_dataset_merged_filtered.csv"

# Save the filtered dataset
filtered_data.to_csv(filtered_file_path, index=False)

# Get the number of entries in the filtered dataset
num_entries = len(filtered_data)

print(f"Filtered dataset saved to: {filtered_file_path}")
print(f"Number of entries in the filtered dataset: {num_entries}")
