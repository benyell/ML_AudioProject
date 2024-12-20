import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the merged dataset
data_file = "data/processed/wavenet_data/wavenet_dataset_merged_filtered.csv"
data = pd.read_csv(data_file)

# Extract features and labels
X_audio = data["audio_chunk"].apply(eval).tolist()  # Convert audio chunks back to arrays
X_features = data["conditioning_features"].apply(eval).tolist()  # Convert features back to arrays
y = data["label"].values  # Labels

# Split into training, validation, and test sets
X_audio_train, X_audio_temp, X_features_train, X_features_temp, y_train, y_temp = train_test_split(
    X_audio, X_features, y, test_size=0.3, random_state=42
)
X_audio_val, X_audio_test, X_features_val, X_features_test, y_val, y_test = train_test_split(
    X_audio_temp, X_features_temp, y_temp, test_size=0.5, random_state=42
)

print(f"Training samples: {len(X_audio_train)}")
print(f"Validation samples: {len(X_audio_val)}")
print(f"Test samples: {len(X_audio_test)}")

# Save splits as numpy arrays
np.save("data/processed/wavenet_data/X_audio_train.npy", X_audio_train)
np.save("data/processed/wavenet_data/X_audio_val.npy", X_audio_val)
np.save("data/processed/wavenet_data/X_audio_test.npy", X_audio_test)
np.save("data/processed/wavenet_data/X_features_train.npy", X_features_train)
np.save("data/processed/wavenet_data/X_features_val.npy", X_features_val)
np.save("data/processed/wavenet_data/X_features_test.npy", X_features_test)
np.save("data/processed/wavenet_data/y_train.npy", y_train)
np.save("data/processed/wavenet_data/y_val.npy", y_val)
np.save("data/processed/wavenet_data/y_test.npy", y_test)

print("Data splits saved successfully!")