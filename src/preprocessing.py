import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import librosa
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download nltk data if needed
for pkg in ['stopwords', 'wordnet', 'punkt', 'punkt_tab']:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass


class Preprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_voice(self, audio_file):
        """Preprocess audio: load, convert to mono, remove silence, normalize."""
        y, sr = librosa.load(audio_file, sr=None, mono=True)
        # Remove silence (trim) with default top_db
        y, _ = librosa.effects.trim(y, top_db=20)
        # Normalize amplitude to [-1, 1]
        if len(y) > 0 and np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        return y, sr

    def preprocess_voice_array(self, y, sr):
        """Preprocess audio array: convert to mono if needed, remove silence, normalize."""
        y = y.astype(float)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        # Remove silence
        y, _ = librosa.effects.trim(y, top_db=20)
        # Normalize amplitude to [-1, 1]
        if len(y) > 0 and np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        return y, sr

    def preprocess_text(self, text):
        """Preprocess text: tokenize, remove stop words, lemmatize."""
        if not text:
            return []
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in self.stop_words]
        return tokens


# Standalone functional helpers
def preprocess_audio_waveform(audio, sr=None):
    """Normalize and convert waveform to mono for feature extraction."""
    audio = audio.astype(float)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    max_abs = np.max(np.abs(audio))
    if max_abs > 0:
        audio = audio / max_abs
    return audio


# Example usage
if __name__ == "__main__":
    prep = Preprocessor()
    tokens = prep.preprocess_text("I am feeling stressed out today.")
    print("Processed text:", tokens)