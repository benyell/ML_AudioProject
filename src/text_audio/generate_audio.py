import numpy as np
import soundfile as sf
from wavenet_model import load_wavenet_model
from text_to_features import generate_conditioning_features

def generate_audio_from_text(text, nlp_model, model_path="wavenet_model.h5", num_samples=16000, output_dir="data/processed/processed_audio"):
    """
    Generate audio based on text input using the trained WaveNet model.

    Parameters:
    - text: User input text (e.g., "play a soft piano sound").
    - nlp_model: Pre-trained NLP model.
    - model_path: Path to the trained WaveNet model.
    - num_samples: Number of audio samples to generate.
    - output_dir: Directory to save the generated audio file.

    Returns:
    - audio_file_path: Path to the generated audio file.
    """
    # Convert text to conditioning features
    conditioning_features = generate_conditioning_features(text, nlp_model)
    print(f"Conditioning Features: {conditioning_features}")

    # Load the trained WaveNet model
    model = load_wavenet_model(model_path)

    # Initialize waveform with silence
    generated_waveform = [0.0]

    for _ in range(num_samples):
        # Prepare the input waveform
        input_waveform = np.array(generated_waveform[-8000:])  # Use the last 8000 samples
        input_waveform = np.pad(input_waveform, (8000 - len(input_waveform), 0))  # Pad if less than 8000 samples

        # Reshape for model input
        input_waveform = input_waveform.reshape(1, -1, 1)
        conditioning_input = np.array(conditioning_features).reshape(1, -1)

        # Predict the next sample
        prediction = model.predict([input_waveform, conditioning_input], verbose=0).flatten()
        prediction = np.clip(prediction, 0, 1)  # Ensure non-negative probabilities
        prediction /= np.sum(prediction)  # Normalize probabilities

        next_sample = np.random.choice(np.arange(len(prediction)), p=prediction)
        generated_waveform.append(next_sample)

    # Normalize the audio waveform
    audio_waveform = np.array(generated_waveform, dtype=np.float32)
    audio_waveform /= np.max(np.abs(audio_waveform))

    # Save the generated audio to a file
    audio_file_name = f"generated_audio_{hash(text)}.wav"
    audio_file_path = f"{output_dir}/{audio_file_name}"
    sf.write(audio_file_path, audio_waveform, 16000)
    print(f"Generated audio saved to: {audio_file_path}")

    return audio_file_path
