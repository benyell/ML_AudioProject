import streamlit as st
from text_to_features import load_nlp_model
from generate_audio import generate_audio_from_text
import os
import numpy as np
import librosa
import soundfile as sf

# Load the NLP model once during app initialization
nlp_model = load_nlp_model()

# Directories
PROCESSED_AUDIO_DIR = "data/processed/processed_audio"
os.makedirs(PROCESSED_AUDIO_DIR, exist_ok=True)

# Streamlit UI
st.title("Text-to-Audio Generator with Virtual Keyboard")
st.subheader("Describe the sound you want, and we'll generate it for you!")

# Text input for user description
user_input = st.text_input("Enter a sound description (e.g., 'play a soothing violin melody'):").strip()

# Placeholder for generated audio
generated_audio_file = None

# Button to trigger audio generation
if st.button("Generate Sound"):
    if user_input:
        st.write("Processing your request... Please wait.")

        try:
            # Generate audio using the text input
            generated_audio_file = generate_audio_from_text(
                user_input,
                nlp_model,
                model_path="data/models/wavenet_model1.h5",
                num_samples=16000,
                output_dir=PROCESSED_AUDIO_DIR
            )

            # Display success message and audio player
            st.success("Audio generated successfully!")
            st.audio(generated_audio_file, format="audio/wav")
        except Exception as e:
            st.error(f"An error occurred while generating audio: {e}")
    else:
        st.warning("Please enter a valid sound description.")

# Virtual Keyboard UI
if generated_audio_file:
    st.subheader("Virtual Keyboard")

    # Load the generated audio
    base_audio, sr = librosa.load(generated_audio_file, sr=16000)

    # Function to pitch shift the audio for different keys
    def pitch_shift(audio, semitones, sr):
        return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=semitones)

    # Key layout for one octave
    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Generate pitch-shifted audio files for each key
    key_audio_files = {}
    for i, key in enumerate(keys):
        try:
            shifted_audio = pitch_shift(base_audio, i, sr)
            key_audio_path = os.path.join(PROCESSED_AUDIO_DIR, f"{key}_key.wav")
            sf.write(key_audio_path, shifted_audio, sr)
            key_audio_files[key] = key_audio_path
        except Exception as e:
            st.error(f"Error generating audio for {key} key: {e}")

    # Display the virtual keyboard
    cols = st.columns(len(keys))
    for i, key in enumerate(keys):
        with cols[i]:
            if st.button(key):
                st.audio(key_audio_files[key], format="audio/wav")
