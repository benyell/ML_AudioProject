import pandas as pd
import os

# Directory containing CSV files
PROCESSED_DIR = "data/processed/wavenet_data/"

# List all batch files
batch_files = [os.path.join(PROCESSED_DIR, f) for f in os.listdir(PROCESSED_DIR) if f.endswith(".csv")]

# Merge all files
merged_data = pd.concat([pd.read_csv(file) for file in batch_files], ignore_index=True)

# Save the merged dataset (optional)
merged_data.to_csv(os.path.join(PROCESSED_DIR, "wavenet_dataset_merged.csv"), index=False)
print(f"Merged dataset saved with {len(merged_data)} samples.")