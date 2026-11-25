import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from src.preprocessing import Preprocessor
from src.feature_extraction import FeatureExtractor

def generate_text_data():
    """Generate synthetic text data for stress classification."""
    # Stress-related texts
    stressed_texts = [
        "I feel so overwhelmed with work today",
        "Everything is piling up and I can't handle it",
        "I'm so stressed about the deadline",
        "My anxiety is through the roof",
        "I can't sleep because I'm worried about everything",
        "This pressure is too much for me",
        "I feel like I'm going to break down",
        "Everything feels heavy and difficult",
        "I'm exhausted from all this stress",
        "My mind won't stop racing with worries",
        "I feel trapped and can't see a way out",
        "The tension is building up inside me",
        "I'm struggling to keep it together",
        "Everything seems impossible right now",
        "I feel so helpless and stressed",
        "The weight of everything is crushing me",
        "I'm on the edge of losing control",
        "This constant worry is draining me",
        "I feel like I'm drowning in stress",
        "Everything feels chaotic and overwhelming"
    ]

    # Mild stress texts
    mild_stressed_texts = [
        "I'm a bit worried about the meeting tomorrow",
        "This project is taking longer than expected",
        "I have a lot on my plate this week",
        "Feeling a little anxious about the presentation",
        "The schedule is tight but manageable",
        "Some things are stressing me out a bit",
        "I could use a break from all this",
        "Things are getting a bit hectic",
        "I'm feeling somewhat overwhelmed",
        "A few things are bothering me today",
        "The pressure is building a little",
        "I'm not as relaxed as I'd like to be",
        "Some concerns are on my mind",
        "Things feel a bit tense right now",
        "I'm dealing with some stress factors"
    ]

    # No stress texts
    no_stress_texts = [
        "I'm feeling good today",
        "Everything is going well",
        "I'm relaxed and enjoying my day",
        "Life feels balanced right now",
        "I'm content with how things are",
        "Feeling peaceful and calm",
        "Things are working out nicely",
        "I'm in a good mood today",
        "Everything feels manageable",
        "I'm feeling positive about life",
        "Things are going smoothly",
        "I'm enjoying the moment",
        "Feeling happy and relaxed",
        "Life is treating me well",
        "I'm in a good place right now",
        "Everything feels right",
        "I'm feeling optimistic",
        "Life is good",
        "I'm content and at peace",
        "Things are working out"
    ]

    # Moderate stress texts
    moderate_stressed_texts = [
        "I'm really struggling with this workload",
        "The stress is getting to me",
        "I feel overwhelmed by everything",
        "This is becoming too much to handle",
        "I'm under a lot of pressure",
        "The anxiety is building up",
        "I can't seem to catch a break",
        "Everything feels stressful right now",
        "I'm dealing with significant stress",
        "The tension is quite high",
        "I'm feeling very overwhelmed",
        "This stress level is concerning",
        "I'm struggling to cope",
        "The pressure is intense",
        "I'm experiencing high stress"
    ]

    # Combine all texts with labels
    texts = []
    labels = []

    # No stress (0)
    texts.extend(no_stress_texts)
    labels.extend([0] * len(no_stress_texts))

    # Mild stress (1)
    texts.extend(mild_stressed_texts)
    labels.extend([1] * len(mild_stressed_texts))

    # Moderate stress (2)
    texts.extend(moderate_stressed_texts)
    labels.extend([2] * len(moderate_stressed_texts))

    # High stress (3)
    texts.extend(stressed_texts)
    labels.extend([3] * len(stressed_texts))

    return texts, labels

def train_text_model():
    """Train a text-based stress classifier."""
    print("Generating text data...")
    texts, labels = generate_text_data()

    print(f"Generated {len(texts)} text samples")
    print(f"Label distribution: {np.bincount(labels)}")

    # Initialize preprocessors
    prep = Preprocessor()
    extractor = FeatureExtractor()

    # Extract features
    print("Extracting features...")
    features = []
    for text in texts:
        tokens = prep.preprocess_text(text)
        text_feat = extractor.extract_text_features(" ".join(tokens))
        features.append(text_feat)

    features = np.array(features)
    labels = np.array(labels)

    print(f"Feature shape: {features.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Train model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, "text_stress_model.pkl")
    print("Model saved as text_stress_model.pkl")

    return model

if __name__ == "__main__":
    train_text_model()