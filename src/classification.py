import numpy as np

class StressClassifier:
    def __init__(self, model):
        self.model = model
        self.labels = {0: 'No stress', 1: 'Mild stress', 2: 'Moderate stress', 3: 'High stress'}

    def classify(self, features):
        """Classify stress level with confidence."""
        pred = self.model.predict([features])[0]
        proba = self.model.predict_proba([features])[0]
        confidence = np.max(proba)
        return self.labels[pred], confidence

    def classify_with_details(self, features):
        """Return prediction, confidence, and probabilities."""
        pred = self.model.predict([features])[0]
        proba = self.model.predict_proba([features])[0]
        confidence = np.max(proba)
        return self.labels[pred], confidence, proba

# Example usage
if __name__ == "__main__":
    from model import StressModel
    model = StressModel()
    # Assume trained
    classifier = StressClassifier(model)
    # Dummy
    result = classifier.classify(np.random.rand(122))
    print("Classification:", result)