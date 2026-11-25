import os
import json
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from scipy.io import wavfile
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

WORKSPACE = Path(__file__).resolve().parents[1]
DATASET_PATH = WORKSPACE / "ravdess-emotional-speech-audio"
RANDOM_STATE = 42
STRESS_LABELS = {0: "No Stress", 1: "Mild Stress", 2: "Moderate Stress", 3: "High Stress"}
RESULTS_DIR = WORKSPACE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

HEURISTIC_STRESS_MAPPING = {
    "neutral": 0,
    "calm": 0,
    "happy": 1,
    "surprised": 1,
    "sad": 2,
    "angry": 3,
    "fearful": 3,
    "disgust": 3,
}

# The current project uses 22,050 Hz RAVDESS audio processing as a common assumption.
# The implementation in preprocessing loads audio with librosa and the model currently expects that convention.
SAMPLE_RATE = 22050
WINDOW_LENGTH = 2048
HOP_LENGTH = 512
N_MFCC = 13


def get_ravdess_files(dataset_path):
    files = []
    for actor_dir in sorted(dataset_path.iterdir()):
        if not actor_dir.is_dir():
            continue
        for audio_file in sorted(actor_dir.iterdir()):
            if audio_file.suffix.lower() == ".wav":
                files.append(audio_file)
    return files


def emotion_from_ravdess_filename(filename):
    filename = os.path.basename(filename)
    parts = filename.split("-")
    emotion_code = parts[2] if len(parts) > 2 else "01"
    return EMOTION_MAP.get(emotion_code, "unknown")


def build_segment_energy_features(audio, sr):
    """Baseline representation used by the current Streamlit app.

    The published paper describes this as 'MFCC-like', but the implementation
    calculates a 13-coefficient segment-energy representation rather than
    standard MFCCs. The exact calculation in the current app.py is:

      1. Normalize audio with x = audio / max(abs(audio))
      2. For i in range(13): coefficient_i = mean(audio[i::13])
      3. energy = sum(audio ** 2) / len(audio)
      4. zcr = mean(abs(diff(sign(audio))))

    Therefore the returned feature vector is 15-dimensional:
      [13 segment-energy coefficients] + [energy] + [zcr]

    This function intentionally preserves that baseline logic and is used for
    comparison experiments only.
    """
    if len(audio) == 0:
        return np.zeros(15, dtype=float)
    normalized = audio.astype(float)
    max_abs = np.max(np.abs(normalized))
    if max_abs > 0:
        normalized = normalized / max_abs

    if len(normalized.shape) > 1:
        normalized = np.mean(normalized, axis=1)

    coeffs = []
    for i in range(13):
        coeffs.append(np.mean(normalized[i::13]))
    coeffs = np.array(coeffs)

    energy = float(np.sum(normalized ** 2) / len(normalized))
    zcr = float(np.mean(np.abs(np.diff(np.sign(normalized)))))
    return np.hstack([coeffs, energy, zcr])


def build_standard_mfcc_features(audio, sr):
    """True standard MFCC ablation used in the reviewer experiment.

    Configuration used for the MFCC feature extraction:
      - sampling rate: 22050 Hz
      - n_mfcc: 13
      - n_fft: 2048
      - hop_length: 512
      - window function: Hann (librosa default)
      - aggregation: mean across time axis
    """
    mel_spec = librosa.feature.mfcc(
        y=audio.astype(float),
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=WINDOW_LENGTH,
        hop_length=HOP_LENGTH,
        window="hann",
    )
    return np.mean(mel_spec, axis=1)


def build_extended_acoustic_features(audio, sr):
    """Reviewer-recognized richer acoustic ablation.

    Adds the standard 13 MFCC coefficients plus their delta and delta-delta
    summaries, F0 pitch mean/std from librosa.pyin, mean spectral centroid/
    bandwidth/rolloff, plus energy and zero-crossing-rate.

    All features are derived per file from the existing RAVDESS audio split and
    the current heuristic stress labels, preserving the label and evaluation
    protocol used elsewhere in the experiment driver.
    """
    if len(audio) == 0:
        return np.zeros(46, dtype=float)

    audio = audio.astype(float)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    max_abs = np.max(np.abs(audio))
    if max_abs > 0:
        audio = audio / max_abs

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=WINDOW_LENGTH,
        hop_length=HOP_LENGTH,
        window="hann",
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_delta = np.mean(librosa.feature.delta(mfcc, order=1), axis=1)
    mfcc_delta2 = np.mean(librosa.feature.delta(mfcc, order=2), axis=1)

    pitch_values, voiced_flags, _ = librosa.pyin(
        audio,
        sr=sr,
        fmin=50,
        fmax=1000,
    )
    voiced = pitch_values[voiced_flags]
    if voiced.size == 0:
        pitch_stats = np.array([0.0, 0.0], dtype=float)
    else:
        pitch_stats = np.array([
            float(np.nanmean(voiced)),
            float(np.nanstd(voiced)),
        ], dtype=float)

    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    spectral_stats = np.array([
        float(np.mean(spectral_centroid)),
        float(np.mean(spectral_bandwidth)),
        float(np.mean(spectral_rolloff)),
    ], dtype=float)

    energy = float(np.sum(audio ** 2) / len(audio))
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))))

    extended_vector = np.hstack([
        mfcc_mean,
        mfcc_delta,
        mfcc_delta2,
        pitch_stats,
        spectral_stats,
        energy,
        zcr,
    ])
    return extended_vector


def load_audio_file(path):
    sr, audio = wavfile.read(path)
    audio = audio.astype(float)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    return audio, float(sr)


def extract_dataset(feature_mode="segment_energy"):
    """Create the current stress-labeled RAVDESS dataset using the heuristic mapping."""
    X, y = [], []
    files = get_ravdess_files(DATASET_PATH)

    for idx, file_path in enumerate(files, start=1):
        try:
            audio, sr = load_audio_file(file_path)
            if feature_mode == "standard_mfcc":
                features = build_standard_mfcc_features(audio, sr)
            elif feature_mode == "extended_acoustic":
                features = build_extended_acoustic_features(audio, sr)
            else:
                features = build_segment_energy_features(audio, sr)

            emotion = emotion_from_ravdess_filename(file_path.name)
            stress_label = HEURISTIC_STRESS_MAPPING.get(emotion, 0)
            X.append(features)
            y.append(stress_label)
        except Exception as exc:
            print(f"Skipping {file_path}: {exc}")

        if idx % 200 == 0:
            print(f"extract_dataset({feature_mode}) processed {idx} / {len(files)} files")

    return np.array(X), np.array(y)


def build_classifier(config=None):
    if config is None:
        config = {
            "n_estimators": 300,
            "max_depth": 20,
            "min_samples_split": 5,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        }
    return RandomForestClassifier(**config)


def evaluate_classifier(X, y, config=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    # Fit scaler on train only to avoid leakage
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = build_classifier(config)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    labels = sorted(set(y))
    class_report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    for label in labels:
        metrics[f"class_{label}_precision"] = class_report[str(label)]['precision']
        metrics[f"class_{label}_recall"] = class_report[str(label)]['recall']
        metrics[f"class_{label}_f1"] = class_report[str(label)]['f1-score']

    return metrics, y_test, y_pred


def evaluate_mfcc_classifier_ablation():
    """Run GradientBoostingClassifier and RBF SVC on the same standard-MFCC representation.

    Uses the same stratified 80:20 split with random_state=42. This function writes a small
    set of row-wise metrics and a per-class F1 table comparable with the RF MFCC baseline.
    """
    X, y = extract_dataset("standard_mfcc")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results = []

    # Scale features for gradient boosting and SVC comparators
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    gbc = GradientBoostingClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
    )
    gbc.fit(X_train_s, y_train)
    y_pred = gbc.predict(X_test_s)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    results.append({
        "Classifier": "GradientBoostingClassifier",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "No_Stress_F1": report['0']['f1-score'] if '0' in report else None,
        "Mild_Stress_F1": report['1']['f1-score'] if '1' in report else None,
        "Moderate_Stress_F1": report['2']['f1-score'] if '2' in report else None,
        "High_Stress_F1": report['3']['f1-score'] if '3' in report else None,
    })

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    svc = SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    svc.fit(X_train_s, y_train)
    y_pred = svc.predict(X_test_s)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    results.append({
        "Classifier": "SVC",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "No_Stress_F1": report['0']['f1-score'] if '0' in report else None,
        "Mild_Stress_F1": report['1']['f1-score'] if '1' in report else None,
        "Moderate_Stress_F1": report['2']['f1-score'] if '2' in report else None,
        "High_Stress_F1": report['3']['f1-score'] if '3' in report else None,
    })

    mfcc_frame = pd.read_csv(RESULTS_DIR / "feature_ablation_results.csv")
    rf_row = mfcc_frame.loc[mfcc_frame['Feature Representation'] == 'Standard MFCC'].iloc[0]
    rf_row_dict = {
        "Classifier": "RandomForestClassifier (MFCC baseline)",
        "Accuracy": float(rf_row['Accuracy']),
        "Precision": float(rf_row['Precision']),
        "Recall": float(rf_row['Recall']),
        "F1": float(rf_row['F1']),
        "No_Stress_F1": None,
        "Mild_Stress_F1": None,
        "Moderate_Stress_F1": None,
        "High_Stress_F1": None,
    }
    results.insert(0, rf_row_dict)

    pd.DataFrame(results).to_csv(RESULTS_DIR / "classifier_ablation_results.csv", index=False)

    return results


def write_feature_ablation_results():
    X_segment, y = extract_dataset("segment_energy")
    X_mfcc, y_mfcc = extract_dataset("standard_mfcc")

    if len(y) == len(y_mfcc) and np.array_equal(y, y_mfcc):
        pass

    metrics_segment, *_ = evaluate_classifier(X_segment, y, config={
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    })

    metrics_mfcc, *_ = evaluate_classifier(X_mfcc, y, config={
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    })

    rows = [
        {
            "Feature Representation": "Segment-energy coefficients",
            "Accuracy": metrics_segment["accuracy"],
            "Precision": metrics_segment["precision"],
            "Recall": metrics_segment["recall"],
            "F1": metrics_segment["f1"],
        },
        {
            "Feature Representation": "Standard MFCC",
            "Accuracy": metrics_mfcc["accuracy"],
            "Precision": metrics_mfcc["precision"],
            "Recall": metrics_mfcc["recall"],
            "F1": metrics_mfcc["f1"],
        },
    ]

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "feature_ablation_results.csv", index=False)

    # Per-class support/std metrics for the same metric frame
    y_test_segment = None
    with open(RESULTS_DIR / "feature_ablation_results.csv", "w", newline="") as csvfile:
        fieldnames = ["Feature Representation", "Accuracy", "Precision", "Recall", "F1"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Add a companion JSON result record.
    with open(RESULTS_DIR / "feature_ablation_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "segment_energy": metrics_segment,
            "standard_mfcc": metrics_mfcc
        }, f, indent=2)


def write_extended_acoustic_results():
    X_ext, y = extract_dataset("extended_acoustic")
    metrics_ext, *_ = evaluate_classifier(X_ext, y, config={
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    })

    rows = [
        {
            "Feature Representation": "Extended acoustic features",
            "Accuracy": metrics_ext["accuracy"],
            "Precision": metrics_ext["precision"],
            "Recall": metrics_ext["recall"],
            "F1": metrics_ext["f1"],
        }
    ]

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "extended_features_results.csv", index=False)

    classes = sorted(set(y))
    class_rows = []
    for label in classes:
        class_rows.append({
            "Class": int(label),
            "Class_Label": STRESS_LABELS[int(label)] if label in STRESS_LABELS else str(label),
            "Precision": metrics_ext.get(f"class_{label}_precision", None),
            "Recall": metrics_ext.get(f"class_{label}_recall", None),
            "F1": metrics_ext.get(f"class_{label}_f1", None),
        })
    pd.DataFrame(class_rows).to_csv(RESULTS_DIR / "extended_features_per_class.csv", index=False)

    mfcc_frame = pd.read_csv(RESULTS_DIR / "feature_ablation_results.csv")
    standard_mfcc_row = mfcc_frame.loc[mfcc_frame['Feature Representation'] == 'Standard MFCC'].iloc[0]
    mfcc_acc = float(standard_mfcc_row['Accuracy'])
    ext_acc = float(metrics_ext['accuracy'])

    if ext_acc > mfcc_acc:
        direction = "improved"
    elif ext_acc < mfcc_acc:
        direction = "worsened"
    else:
        direction = "had no clear effect"

    note_lines = []
    note_lines.append('# Extended acoustic feature ablation note')
    note_lines.append('')
    note_lines.append('## Results')
    note_lines.append(f"Extended acoustic features accuracy: {ext_acc:.6f}; precision: {float(metrics_ext['precision']):.6f}; recall: {float(metrics_ext['recall']):.6f}; F1: {float(metrics_ext['f1']):.6f}.")
    note_lines.append(f"Standard MFCC accuracy from the existing ablation: {mfcc_acc:.6f}.")
    note_lines.append(f"Relative to those standard-MFCC results the extended acoustic representation {direction} on held-out evaluation accuracy.")
    note_lines.append('')
    note_lines.append('## Per-class results')
    for row in class_rows:
        note_lines.append(
            f"Class {row['Class']} ({row['Class_Label']}): precision={row['Precision']:.6f}, recall={row['Recall']:.6f}, F1={row['F1']:.6f}."
        )
    note_lines.append('')
    note_lines.append('## Interpretation')
    note_lines.append('This note follows the measured accuracy comparison only. It does not claim a clinical or externally validated improvement beyond the experiment output shown above.')
    (RESULTS_DIR / "extended_acoustic_ablation_note.md").write_text("\n".join(note_lines), encoding="utf-8")


def run_cross_validation():
    X, y = extract_dataset("segment_energy")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fold_rows = []
    cv_metric_rows = []
    acc_values, weighted_f1_values = [], []
    class_f1 = {c: [] for c in sorted(set(y))}
    class_recall = {c: [] for c in sorted(set(y))}

    for fold_index, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # Fit scaler inside each fold to avoid leakage
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = build_classifier({
            "n_estimators": 300,
            "max_depth": 20,
            "min_samples_split": 5,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        })
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        acc_values.append(acc)
        weighted_f1_values.append(weighted_f1)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        for c in sorted(set(y)):
            class_f1[c].append(report[str(c)]['f1-score'])
            class_recall[c].append(report[str(c)]['recall'])

        fold_rows.append({
            "Fold": fold_index,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "Weighted_F1": weighted_f1,
            "F1": f1,
            "No_Stress_F1": report['0']['f1-score'] if '0' in report else None,
            "Mild_Stress_F1": report['1']['f1-score'] if '1' in report else None,
            "Moderate_Stress_F1": report['2']['f1-score'] if '2' in report else None,
            "High_Stress_F1": report['3']['f1-score'] if '3' in report else None,
        })

    # Save fold-by-fold CSV.
    pd.DataFrame(fold_rows).to_csv(RESULTS_DIR / "cross_validation_folds.csv", index=False)

    summary = {
        "Metric": ["Accuracy", "Weighted_F1"],
        "Mean": [float(np.mean(acc_values)), float(np.mean(weighted_f1_values))],
        "Std": [float(np.std(acc_values, ddof=1)), float(np.std(weighted_f1_values, ddof=1))],
    }

    # per-class F1 and recall mean/std
    for c in sorted(set(y)):
        summary[f"Class_{STRESS_LABELS[c]}_F1_mean"] = [float(np.mean(class_f1[c]))]
        summary[f"Class_{STRESS_LABELS[c]}_F1_std"] = [float(np.std(class_f1[c], ddof=1))]
        summary[f"Class_{STRESS_LABELS[c]}_Recall_mean"] = [float(np.mean(class_recall[c]))]
        summary[f"Class_{STRESS_LABELS[c]}_Recall_std"] = [float(np.std(class_recall[c], ddof=1))]

    # Convert to table-like row.
    summary_rows = []
    summary_rows.append({
        "Metric": "Accuracy",
        "Mean": float(np.mean(acc_values)),
        "Std": float(np.std(acc_values, ddof=1)),
    })
    summary_rows.append({
        "Metric": "Weighted_F1",
        "Mean": float(np.mean(weighted_f1_values)),
        "Std": float(np.std(weighted_f1_values, ddof=1)),
    })
    for c in sorted(set(y)):
        summary_rows.append({
            "Metric": f"Class_{STRESS_LABELS[c]}_F1",
            "Mean": float(np.mean(class_f1[c])),
            "Std": float(np.std(class_f1[c], ddof=1)),
        })
        summary_rows.append({
            "Metric": f"Class_{STRESS_LABELS[c]}_Recall",
            "Mean": float(np.mean(class_recall[c])),
            "Std": float(np.std(class_recall[c], ddof=1)),
        })

    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "cross_validation_results.csv", index=False)


def write_random_forest_ablation():
    X, y = extract_dataset("segment_energy")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    configs = {
        "Model_A_simple_RF": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        },
        "Model_B_current_RF": {
            "n_estimators": 300,
            "max_depth": 20,
            "min_samples_split": 5,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        },
    }

    rows = []
    for name, cfg in configs.items():
        model = build_classifier(cfg)
        # scale for consistency
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        rows.append({
            "Configuration": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "No_Stress_F1": report['0']['f1-score'] if '0' in report else None,
            "Mild_Stress_F1": report['1']['f1-score'] if '1' in report else None,
            "Moderate_Stress_F1": report['2']['f1-score'] if '2' in report else None,
            "High_Stress_F1": report['3']['f1-score'] if '3' in report else None,
        })

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "random_forest_ablation.csv", index=False)


def write_majority_baseline_comparison():
    X, y = extract_dataset("segment_energy")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    values, counts = np.unique(y_test, return_counts=True)
    majority_class = int(values[np.argmax(counts)])
    majority_count = int(counts[np.argmax(counts)])
    majority_pct = float(majority_count / len(y_test) * 100.0)

    model = build_classifier({
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    })
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred)

    pd.DataFrame([
        {
            "Dataset": "RAVDESS stress mapping (actual 80:20 test split)",
            "Majority_Class": majority_class,
            "Majority_Class_Name": STRESS_LABELS[majority_class],
            "Number_of_Samples": majority_count,
            "Percentage_of_Test_Samples": majority_pct,
            "Random_Forest_Test_Accuracy": rf_accuracy,
            "Accuracy_vs_Majority_Baseline": rf_accuracy - majority_pct / 100.0,
        }
    ]).to_csv(RESULTS_DIR / "baseline_comparison.csv", index=False)


def write_research_summary():
    # This placeholder pulls actual experiment outputs from previously written CSVs when present.
    summary_lines = [
        "# Experiment Summary",
        "",
        "## Dataset",
        "The repository currently contains a RAVDESS speech subset under the workspace folder `ravdess-emotional-speech-audio`. The same dataset is used for the experiment scripts. The heuristic emotion-to-stress mapping is defined in `research_config.json`.",
        "",
        "## Label Mapping",
        "Heuristic emotion-to-stress mapping: Neutral -> No Stress; Calm -> No Stress; Happy -> Mild Stress; Surprised -> Mild Stress; Sad -> Moderate Stress; Angry -> High Stress; Fearful -> High Stress; Disgust -> High Stress.",
        "",
        "## Feature Extraction",
        "The baseline implementation in `app.py` calculates a 13-dimensional segment-energy coefficient vector through coefficient_i = mean(audio[i::13]), then appends energy and zero-crossing rate, yielding a 15-dimensional feature vector. This is documented as a segment-energy coefficient representation and must not be described as a standard MFCC implementation.",
        "",
        "## Evaluation Status",
        "The current repository supports text-only sentiment analysis and OCR, but the available RAVDESS dataset does not provide corresponding text labels. Text-only and multimodal evaluation therefore remain status-only reports rather than quantitative claims.",
        "",
        "## Current Outputs",
        "The following CSV artefacts are produced by the reproducible experiments when they are run: feature_ablation_results.csv, cross_validation_results.csv, cross_validation_folds.csv, baseline_comparison.csv, random_forest_ablation.csv.",
        "",
        "## Reproducibility",
        "The experiment configuration is saved in `research_config.json` with random_state = 42, 80:20 train/test methodology, 5-fold stratified CV and the Random Forest configuration used in the nominal baseline.",
    ]
    (WORKSPACE / "experiment_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


def main():
    write_feature_ablation_results()
    write_extended_acoustic_results()
    run_cross_validation()
    write_majority_baseline_comparison()
    write_random_forest_ablation()
    write_research_summary()

    # Status-only reports that cannot be interpreted as data.
    (WORKSPACE / "text_only_evaluation_status.txt").write_text(
        "Text-only classification could not be quantitatively evaluated because the available RAVDESS dataset does not contain corresponding text samples with stress labels.\n",
        encoding="utf-8",
    )

    (WORKSPACE / "modality_evaluation_status.txt").write_text(
        "Multimodal fusion is implemented as a system capability, but it is not quantitatively validated because the repository does not contain paired labelled audio/text samples with common stress labels.\n",
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
