import numpy as np
from scipy.io.wavfile import write

def generate_sample_audio(filename, stressed=True):
    """Generate synthetic audio sample."""
    sr = 22050
    duration = 3  # seconds
    if stressed:
        # Higher pitch, more energy
        freq = 300
        energy = 0.8
    else:
        freq = 200
        energy = 0.5

    t = np.linspace(0, duration, int(sr * duration), False)
    audio = energy * np.sin(freq * 2 * np.pi * t) + 0.1 * np.random.randn(len(t))
    audio = (audio * 32767).astype(np.int16)  # Convert to 16-bit
    write(filename, sr, audio)

if __name__ == "__main__":
    generate_sample_audio("data/raw/stressed.wav", stressed=True)
    generate_sample_audio("data/raw/not_stressed.wav", stressed=False)