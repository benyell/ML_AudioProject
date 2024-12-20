import numpy as np
import pandas as pd
from transformers import pipeline

# Metadata File
METADATA_FILE = "src/notebooks/processed_features.csv"

def load_nlp_model():
    """Load a pre-trained NLP model for text classification."""
    print("Loading NLP model...")
    return pipeline("text-classification", model="distilbert-base-uncased", return_all_scores=True)

def generate_conditioning_features(text, nlp_model):
    """
    Generate conditioning features for WaveNet based on user input text.

    Parameters:
    - text: User input (e.g., "play a soft piano sound").
    - nlp_model: Pre-trained NLP model.

    Returns:
    - conditioning_features: Array of features for WaveNet.
    """
    # Load metadata
    metadata_df = pd.read_csv(METADATA_FILE)
    metadata_df["features"] = metadata_df["features"].apply(eval)  # Convert stringified features to lists
    instrument_classes = metadata_df["instrument_family"].unique()

    # Map text to instrument family
    classification = nlp_model(text)
    scores = [score["score"] for score in classification[0]]
    instrument_family_idx = np.argmax(scores)
    instrument_family = instrument_classes[instrument_family_idx]
    print(f"Predicted Instrument Family: {instrument_family}")

    # Extract representative features for the predicted instrument family
    family_features = metadata_df[metadata_df["instrument_family"] == instrument_family].iloc[0]["features"]
    family_features = np.array(family_features)

    return family_features
