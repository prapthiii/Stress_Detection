import os
import numpy as np
from scipy.io import wavfile
import warnings
warnings.filterwarnings("ignore")

# ✅ ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ✅ Save model
import joblib

# 🎯 Emotion mapping
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# 🎯 Stress mapping
stress_map = {
    "neutral": 0,
    "calm": 0,
    "happy": 1,
    "surprised": 1,
    "sad": 2,
    "angry": 3,
    "fearful": 3,
    "disgust": 3
}

# ✅ Extract emotion
def extract_emotion_from_filename(filename):
    parts = filename.split("-")
    emotion_code = parts[2]
    return emotion_map.get(emotion_code, "unknown")

# ✅ Convert to stress
def get_stress_level(emotion):
    return stress_map.get(emotion, 0)

# ✅ Custom MFCC
def extract_mfcc_simple(audio, sr, n_mfcc=13):
    audio = audio / np.max(np.abs(audio))

    mfcc = []
    for i in range(n_mfcc):
        mfcc.append(np.mean(audio[i::n_mfcc]))
    return np.array(mfcc)

# ✅ Build dataset
def build_dataset():
    # Use workspace-relative dataset path if available
    base_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(base_dir, "ravdess-emotional-speech-audio")

    X, y = [], []

    print("\nBuilding dataset...\n")

    for actor in os.listdir(dataset_path):
        actor_path = os.path.join(dataset_path, actor)

        if not os.path.isdir(actor_path):
            continue

        for file in os.listdir(actor_path):
            if not file.endswith(".wav"):
                continue

            file_path = os.path.join(actor_path, file)

            try:
                sr, audio = wavfile.read(file_path)
                audio = audio.astype(float)

                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)

                # 🎧 Features
                mfcc = extract_mfcc_simple(audio, sr)
                energy = np.sum(audio**2) / len(audio)
                zcr = np.mean(np.abs(np.diff(np.sign(audio))))

                features = np.hstack([mfcc, energy, zcr])

                # 🎯 Labels
                emotion = extract_emotion_from_filename(file)
                stress = get_stress_level(emotion)

                X.append(features)
                y.append(stress)

            except Exception as e:
                print(f"Error processing {file}: {e}")

    X, y = np.array(X), np.array(y)

    print(f"\nTotal samples: {len(X)}")
    print(f"Feature shape: {X.shape}")

    return X, y

# ✅ Train model
def train_model(X, y):
    print("\nTraining model...\n")
    # Split before scaling to avoid data leakage; use stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Model Accuracy: {acc:.2f}")

    print("\nDetailed Report:\n")
    print(classification_report(y_test, y_pred))

    return model, scaler

# 🚀 Main
def main():
    X, y = build_dataset()

    print("\nSample labels:", y[:10])

    model, scaler = train_model(X, y)

    # 💾 Save model and scaler into results/
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    joblib.dump(model, os.path.join(results_dir, "stress_model.pkl"))
    joblib.dump(scaler, os.path.join(results_dir, "scaler.pkl"))

    print(f"\n✅ Model saved as {os.path.join(results_dir, 'stress_model.pkl')}")
    print(f"✅ Scaler saved as {os.path.join(results_dir, 'scaler.pkl')}")

# ✅ Entry point
if __name__ == "__main__":
    main()