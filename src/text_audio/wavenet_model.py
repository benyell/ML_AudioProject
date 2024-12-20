from keras.models import load_model

def load_wavenet_model(model_path="data/models/wavenet_model.h5"):
    """Load the trained WaveNet model."""
    return load_model(model_path)
