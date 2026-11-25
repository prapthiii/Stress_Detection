import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureFusion:
    def __init__(self):
        self.scaler = StandardScaler()

    def fuse_features(self, voice_features, text_features, method='concat'):
        """Fuse voice and text features with normalization."""
        # Normalize features
        voice_norm = self.scaler.fit_transform(voice_features.reshape(1, -1)).flatten()
        text_norm = self.scaler.fit_transform(text_features.reshape(1, -1)).flatten()

        if method == 'concat':
            return np.concatenate([voice_norm, text_norm])
        elif method == 'average':
            # Assuming same length after normalization
            return (voice_norm + text_norm) / 2
        else:
            return np.concatenate([voice_norm, text_norm])

# Example usage
if __name__ == "__main__":
    fusion = FeatureFusion()
    voice = np.random.rand(20)
    text = np.random.rand(102)
    fused = fusion.fuse_features(voice, text)
    print("Fused features shape:", fused.shape)