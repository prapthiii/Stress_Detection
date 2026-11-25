import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import librosa
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure VADER lexicon is downloaded
try:
    nltk.download('vader_lexicon', quiet=True)
except Exception:
    pass


def extract_segment_energy_features(audio, sr=22050):
    """Extract 15-dimensional segment-energy baseline features.
    
    1. Normalize audio: x = audio / max(abs(audio))
    2. Compute 13 segment-energy coefficients: mean(x[i::13]) for i in 0..12
    3. Compute energy: sum(x**2) / len(x)
    4. Compute zero-crossing rate: mean(abs(diff(sign(x))))
    
    Returns 15-dimensional numpy array: [13 coefficients, energy, zcr].
    """
    if len(audio) == 0:
        return np.zeros(15, dtype=float)
        
    normalized = audio.astype(float)
    if len(normalized.shape) > 1:
        normalized = np.mean(normalized, axis=1)
        
    max_abs = np.max(np.abs(normalized))
    if max_abs > 0:
        normalized = normalized / max_abs
        
    coeffs = np.array([np.mean(normalized[i::13]) for i in range(13)], dtype=float)
    energy = float(np.sum(normalized ** 2) / len(normalized))
    zcr = float(np.mean(np.abs(np.diff(np.sign(normalized)))))
    return np.hstack([coeffs, energy, zcr])


def extract_standard_mfcc_features(audio, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512, window="hann"):
    """Extract 13-dimensional standard MFCC features averaged over time.
    
    Uses standard librosa.feature.mfcc with Hann window and mean aggregation.
    
    Returns 13-dimensional numpy array.
    """
    if len(audio) == 0:
        return np.zeros(n_mfcc, dtype=float)
        
    audio = audio.astype(float)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
        
    mel_spec = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window
    )
    return np.mean(mel_spec, axis=1)


class FeatureExtractor:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.tfidf_vectorizer = None

    def extract_segment_energy_features(self, y, sr=22050):
        """Extract 15-dim segment-energy features."""
        return extract_segment_energy_features(y, sr)

    def extract_standard_mfcc_features(self, y, sr=22050):
        """Extract 13-dim standard MFCC features."""
        return extract_standard_mfcc_features(y, sr)

    def extract_voice_features(self, y, sr):
        """Extract voice features for real-time application (backward compatibility)."""
        # Trim silence and normalize
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        if len(y_trimmed) > 0:
            y = y_trimmed

        if len(y) == 0:
            return np.zeros(16, dtype=float)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = float(np.mean(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0

        # Energy
        energy = float(np.mean(librosa.feature.rms(y=y)))

        # Speech rate (rough estimate)
        speech_rate = float(len(librosa.onset.onset_detect(y=y, sr=sr)) / (len(y)/sr)) if len(y) > 0 else 0.0

        features = np.concatenate([mfcc_mean, [pitch_mean, energy, speech_rate]])
        return features

    def extract_text_features(self, text, use_transformer=False):
        """Extract text features: VADER sentiment scores, text statistics, TF-IDF."""
        if not text:
            return np.zeros(14, dtype=float)
            
        # NLTK VADER sentiment scoring
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        sentiment_score = sentiment['compound']

        # Basic text features
        words = text.split()
        word_count = len(words)
        avg_word_length = np.mean([len(word) for word in words]) if words else 0.0
        unique_words = len(set(words))
        type_token_ratio = unique_words / word_count if word_count > 0 else 0.0

        # TF-IDF: For demo, fit on sample corpus if not initialized
        if self.tfidf_vectorizer is None:
            sample_texts = [
                "I feel stressed and overwhelmed",
                "I'm happy and relaxed",
                "This is neutral text",
                "I'm anxious about the future",
                "Everything is fine"
            ]
            self.tfidf_vectorizer = TfidfVectorizer(max_features=10)
            self.tfidf_vectorizer.fit(sample_texts)

        tfidf_features = self.tfidf_vectorizer.transform([text]).toarray().flatten()
        features = np.array([sentiment_score, word_count, avg_word_length, type_token_ratio])
        return np.concatenate([features, tfidf_features])


# Example usage
if __name__ == "__main__":
    extractor = FeatureExtractor()
    text_feat = extractor.extract_text_features("I am happy and relaxed today")
    print("Text features shape:", text_feat.shape)
    print("VADER sentiment polarity compound:", text_feat[0])