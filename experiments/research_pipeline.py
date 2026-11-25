from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import sklearn
from scipy.io import wavfile
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "research_config.json"
DATASET_PATH = ROOT / "ravdess-emotional-speech-audio"
RESULTS_PATH = ROOT / "results"
FEATURE_NAMES = (
    "strided_mean",
    "mfcc",
    "mfcc_delta",
    "mfcc_delta_delta",
    "mfcc_pitch",
    "mfcc_spectral",
    "extended_acoustic",
)
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def read_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def parse_recording(path: Path):
    fields = path.stem.split("-")
    if len(fields) != 7 or fields[2] not in EMOTIONS:
        raise ValueError("filename does not contain a valid RAVDESS emotion and actor code")
    actor_id = int(fields[6])
    if not 1 <= actor_id <= 24:
        raise ValueError("actor ID is outside the RAVDESS speaker range")
    return actor_id, EMOTIONS[fields[2]]


def preprocess_waveform(audio, sample_rate, target_rate, top_db=20):
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if audio.ndim != 1 or audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError("audio is empty, non-mono, or contains non-finite samples")
    if sample_rate <= 0:
        raise ValueError("invalid sample rate")
    if sample_rate != target_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_rate)
    peak = float(np.max(np.abs(audio)))
    if peak == 0:
        raise ValueError("audio is silent")
    audio = audio / peak
    audio, _ = librosa.effects.trim(audio, top_db=top_db)
    if audio.size < 2 or not np.all(np.isfinite(audio)):
        raise ValueError("audio is empty after trimming")
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak
    return audio


def _pitch_summary(audio, sample_rate, hop_length):
    pitches = librosa.yin(audio, fmin=50, fmax=1000, sr=sample_rate, frame_length=2048, hop_length=hop_length)
    pitches = pitches[np.isfinite(pitches) & (pitches >= 50) & (pitches <= 1000)]
    if pitches.size == 0:
        return np.zeros(2, dtype=float)
    return np.asarray([np.mean(pitches), np.std(pitches)], dtype=float)


def extract_representations(audio, sample_rate):
    hop_length = 512
    n_fft = 2048
    started = time.perf_counter()
    coeffs = np.asarray([np.mean(audio[index::13]) for index in range(13)], dtype=float)
    energy = float(np.mean(audio ** 2))
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))))
    strided = np.concatenate([coeffs, [energy, zcr]])
    strided_seconds = time.perf_counter() - started

    mfcc_started = time.perf_counter()
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13, n_fft=n_fft, hop_length=hop_length, window="hann")
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_seconds = time.perf_counter() - mfcc_started
    delta_started = time.perf_counter()
    delta = librosa.feature.delta(mfcc, order=1)
    delta_delta = librosa.feature.delta(mfcc, order=2)
    delta_mean = np.mean(delta, axis=1)
    delta_delta_mean = np.mean(delta_delta, axis=1)
    delta_seconds = time.perf_counter() - delta_started
    pitch_started = time.perf_counter()
    pitch = _pitch_summary(audio, sample_rate, hop_length)
    pitch_seconds = time.perf_counter() - pitch_started

    spectral_started = time.perf_counter()
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)))
    spectral = np.asarray([centroid, bandwidth, rolloff, energy, zcr], dtype=float)
    spectral_seconds = time.perf_counter() - spectral_started

    representations = {
        "strided_mean": strided,
        "mfcc": mfcc_mean,
        "mfcc_delta": np.concatenate([mfcc_mean, delta_mean]),
        "mfcc_delta_delta": np.concatenate([mfcc_mean, delta_mean, delta_delta_mean]),
        "mfcc_pitch": np.concatenate([mfcc_mean, pitch]),
        "mfcc_spectral": np.concatenate([mfcc_mean, spectral]),
        "extended_acoustic": np.concatenate([mfcc_mean, delta_mean, delta_delta_mean, pitch, [centroid, bandwidth, rolloff, energy, zcr]]),
    }
    common_mfcc = mfcc_seconds
    timings = {
        "strided_mean": strided_seconds,
        "mfcc": common_mfcc,
        "mfcc_delta": common_mfcc + delta_seconds,
        "mfcc_delta_delta": common_mfcc + delta_seconds,
        "mfcc_pitch": common_mfcc + pitch_seconds,
        "mfcc_spectral": common_mfcc + spectral_seconds,
        "extended_acoustic": common_mfcc + delta_seconds + pitch_seconds + spectral_seconds,
    }
    return representations, timings


def extract_dataset(config, output_dir):
    dataset_path = ROOT / config["dataset"]["path"]
    target_rate = int(config["dataset"]["sampling_rate"])
    original_mapping = config["dataset"]["emotion_to_stress"]
    records = []
    skipped = []
    extraction_seconds = {name: 0.0 for name in FEATURE_NAMES}
    wav_paths = sorted(dataset_path.glob("Actor_*/*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No RAVDESS WAV files found under {dataset_path}")

    for path in wav_paths:
        try:
            actor_id, emotion = parse_recording(path)
            source_rate, raw_audio = wavfile.read(path)
            audio = preprocess_waveform(raw_audio, int(source_rate), target_rate)
            row = {
                "sample_id": path.relative_to(ROOT).as_posix(),
                "filename": path.name,
                "actor_id": actor_id,
                "emotion": emotion,
                "source_sample_rate": int(source_rate),
                "sample_rate": target_rate,
                "stress_label": int(original_mapping[emotion]),
            }
            representations, timings = extract_representations(audio, target_rate)
            for name, vector in representations.items():
                extraction_seconds[name] += timings[name]
                if vector.ndim != 1 or not np.all(np.isfinite(vector)):
                    raise ValueError(f"{name} feature vector is malformed")
                for index, value in enumerate(vector):
                    row[f"{name}_{index:02d}"] = float(value)
                row[f"{name}_dimensions"] = int(vector.size)
            records.append(row)
        except Exception as error:
            skipped.append({"filename": path.relative_to(ROOT).as_posix(), "error": f"{type(error).__name__}: {error}"})

    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("No valid samples were extracted")
    if frame["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate sample IDs found")
    frame.to_csv(output_dir / "sample_features.csv", index=False)
    pd.DataFrame(skipped, columns=["filename", "error"]).to_csv(output_dir / "skipped_files.csv", index=False)
    extraction_cost = [
        {
            "feature_representation": name,
            "feature_dimensionality": int(frame[f"{name}_dimensions"].iloc[0]),
            "extraction_seconds_total": extraction_seconds[name],
            "extraction_seconds_per_sample": extraction_seconds[name] / len(frame),
        }
        for name in FEATURE_NAMES
    ]
    return frame, extraction_cost, skipped


def feature_matrix(frame, representation):
    columns = [column for column in frame.columns if column.startswith(f"{representation}_") and column != f"{representation}_dimensions"]
    columns.sort()
    return frame[columns].to_numpy(dtype=float)


def make_estimator(name, config, seed):
    rf = config["modeling"]["random_forest"]
    if name == "logistic_regression":
        return LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    if name == "svm":
        return SVC(kernel="rbf", class_weight="balanced", random_state=seed)
    if name == "random_forest":
        params = dict(rf)
        params["random_state"] = seed
        params["class_weight"] = None
        return RandomForestClassifier(**params)
    if name == "random_forest_balanced":
        params = dict(rf)
        params["random_state"] = seed
        params["class_weight"] = "balanced"
        return RandomForestClassifier(**params)
    if name == "random_forest_oversampled":
        params = dict(rf)
        params["random_state"] = seed
        params["class_weight"] = None
        return RandomForestClassifier(**params)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed)
    raise ValueError(f"Unsupported model: {name}")


def fit_predict(name, X_train, y_train, X_test, config, seed):
    if name == "majority":
        values, counts = np.unique(y_train, return_counts=True)
        return np.full(len(X_test), values[np.argmax(counts)]), None

    estimator = make_estimator(name, config, seed)
    started = time.perf_counter()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if name == "random_forest_oversampled":
        rng = np.random.default_rng(seed)
        class_values, counts = np.unique(y_train, return_counts=True)
        target_count = int(counts.max())
        sampled_indices = np.concatenate(
            [rng.choice(np.flatnonzero(y_train == value), target_count, replace=True) for value in class_values]
        )
        rng.shuffle(sampled_indices)
        estimator.fit(X_train_scaled[sampled_indices], y_train[sampled_indices])
    else:
        estimator.fit(X_train_scaled, y_train)
    training_seconds = time.perf_counter() - started
    model = make_pipeline(scaler, estimator)
    # Reuse the fitted training-only scaler; pipeline prediction does not refit it.
    inference_started = time.perf_counter()
    predictions = estimator.predict(X_test_scaled)
    inference_seconds = (time.perf_counter() - inference_started) / max(len(X_test), 1)
    model_size = len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    return predictions, {
        "training_seconds": training_seconds,
        "inference_seconds_per_sample": inference_seconds,
        "model_size_bytes": model_size,
    }


def calculate_metrics(y_true, y_pred, class_labels):
    report = classification_report(y_true, y_pred, labels=class_labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=class_labels, average="weighted", zero_division=0)),
        "per_class": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_labels).tolist(),
    }


def model_names():
    return ["majority", "logistic_regression", "svm", "random_forest", "random_forest_balanced", "random_forest_oversampled", "gradient_boosting"]


def run_grouped_cv(frame, config, feature_modes, seeds, mapping_name, mapping, models=None):
    fold_rows = []
    class_rows = []
    prediction_rows = []
    cost_rows = []
    matrix_records = {}
    class_labels = sorted(set(int(value) for value in mapping.values()))
    groups = frame["actor_id"].to_numpy()
    for representation in feature_modes:
        X = feature_matrix(frame, representation)
        for model_name in models or model_names():
            condition_key = (mapping_name, representation, model_name)
            matrix_records[condition_key] = {}
            for seed in seeds:
                y = frame["emotion"].map(mapping).to_numpy(dtype=int)
                splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                seed_truth = []
                seed_predictions = []
                seed_groups = []
                seed_ids = []
                for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
                    train_actors = set(groups[train_idx])
                    test_actors = set(groups[test_idx])
                    if train_actors & test_actors:
                        raise AssertionError("Actor overlap detected in grouped cross-validation")
                    if set(y[train_idx]) != set(class_labels):
                        raise ValueError("A training fold is missing one or more mapped classes")
                    prediction, cost = fit_predict(model_name, X[train_idx], y[train_idx], X[test_idx], config, seed + fold)
                    metrics = calculate_metrics(y[test_idx], prediction, class_labels)
                    row = {
                        "experiment_id": f"{mapping_name}__{representation}__{model_name}__seed{seed}__fold{fold}",
                        "mapping": mapping_name,
                        "feature_representation": representation,
                        "model": model_name,
                        "evaluation": "stratified_group_kfold",
                        "seed": seed,
                        "fold": fold,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "train_actor_count": len(train_actors),
                        "test_actor_count": len(test_actors),
                        "train_test_actor_overlap": 0,
                        "feature_dimensionality": X.shape[1],
                        "config_json": json.dumps(config["modeling"], sort_keys=True),
                        **{key: value for key, value in metrics.items() if key not in ("per_class", "confusion_matrix")},
                    }
                    fold_rows.append(row)
                    for label in class_labels:
                        entry = metrics["per_class"][str(label)]
                        class_rows.append({
                            "experiment_id": row["experiment_id"],
                            "mapping": mapping_name,
                            "feature_representation": representation,
                            "model": model_name,
                            "evaluation": "stratified_group_kfold",
                            "seed": seed,
                            "fold": fold,
                            "class_id": label,
                            "class_name": config["dataset"]["stress_labels"].get(str(label), str(label)),
                            "precision": entry["precision"],
                            "recall": entry["recall"],
                            "f1": entry["f1-score"],
                            "support": entry["support"],
                        })
                    seed_truth.extend(y[test_idx].tolist())
                    seed_predictions.extend(prediction.tolist())
                    seed_groups.extend(groups[test_idx].tolist())
                    seed_ids.extend(frame["sample_id"].iloc[test_idx].tolist())
                    if cost:
                        cost_rows.append({
                            "experiment_id": row["experiment_id"],
                            "mapping": mapping_name,
                            "feature_representation": representation,
                            "model": model_name,
                            "seed": seed,
                            "fold": fold,
                            "feature_dimensionality": X.shape[1],
                            **cost,
                        })
                seed_metrics = calculate_metrics(seed_truth, seed_predictions, class_labels)
                matrix_records[condition_key][str(seed)] = seed_metrics["confusion_matrix"]
                for sample_id, actor, actual, predicted in zip(seed_ids, seed_groups, seed_truth, seed_predictions):
                    prediction_rows.append({
                        "mapping": mapping_name,
                        "feature_representation": representation,
                        "model": model_name,
                        "evaluation": "stratified_group_kfold",
                        "seed": seed,
                        "actor_id": actor,
                        "sample_id": sample_id,
                        "y_true": actual,
                        "y_pred": predicted,
                    })
    return fold_rows, class_rows, prediction_rows, cost_rows, matrix_records


def run_grouped_holdout(frame, config, seeds):
    rows = []
    class_rows = []
    class_labels = sorted(set(int(value) for value in config["dataset"]["emotion_to_stress"].values()))
    y = frame["stress_label"].to_numpy(dtype=int)
    groups = frame["actor_id"].to_numpy()
    representations = list(FEATURE_NAMES)
    models = ["random_forest_balanced"]
    for representation in ("mfcc", "extended_acoustic"):
        models.extend(["logistic_regression", "svm", "random_forest", "random_forest_oversampled", "gradient_boosting"])
    for representation in representations:
        X = feature_matrix(frame, representation)
        for model_name in (models if representation in ("mfcc", "extended_acoustic") else ["random_forest_balanced"]):
            for seed in seeds:
                splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                train_idx, test_idx = next(splitter.split(X, y, groups))
                if set(groups[train_idx]) & set(groups[test_idx]):
                    raise AssertionError("Actor overlap detected in grouped holdout split")
                prediction, cost = fit_predict(model_name, X[train_idx], y[train_idx], X[test_idx], config, seed)
                metrics = calculate_metrics(y[test_idx], prediction, class_labels)
                experiment_id = f"original__{representation}__{model_name}__holdout_seed{seed}"
                rows.append({
                    "experiment_id": experiment_id,
                    "mapping": "original",
                    "feature_representation": representation,
                    "model": model_name,
                    "evaluation": "grouped_holdout",
                    "seed": seed,
                    "fold": 1,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "train_actor_count": len(set(groups[train_idx])),
                    "test_actor_count": len(set(groups[test_idx])),
                    "train_test_actor_overlap": 0,
                    "feature_dimensionality": X.shape[1],
                    "config_json": json.dumps(config["modeling"], sort_keys=True),
                    **{key: value for key, value in metrics.items() if key not in ("per_class", "confusion_matrix")},
                })
                for label in class_labels:
                    entry = metrics["per_class"][str(label)]
                    class_rows.append({
                        "experiment_id": experiment_id,
                        "mapping": "original",
                        "feature_representation": representation,
                        "model": model_name,
                        "evaluation": "grouped_holdout",
                        "seed": seed,
                        "fold": 1,
                        "class_id": label,
                        "class_name": config["dataset"]["stress_labels"].get(str(label), str(label)),
                        "precision": entry["precision"],
                        "recall": entry["recall"],
                        "f1": entry["f1-score"],
                        "support": entry["support"],
                    })
    return rows, class_rows


def write_outputs(output_dir, frame, config, feature_cost, skipped, args):
    seeds = args.seeds or config["modeling"].get("seeds", [1, 10, 21, 42, 100])
    feature_modes = list(FEATURE_NAMES)
    metric_rows, class_rows, predictions, cost_rows, matrices = run_grouped_cv(
        frame, config, feature_modes, seeds, "original", config["dataset"]["emotion_to_stress"]
    )

    alternate_mappings = config["dataset"].get("alternative_emotion_to_stress", {})
    for mapping_name, mapping in alternate_mappings.items():
        rows, classes, preds, costs, matrix_records = run_grouped_cv(
            frame, config, ["strided_mean", "mfcc", "extended_acoustic"], seeds, mapping_name, mapping,
            models=["random_forest_balanced"],
        )
        metric_rows.extend(rows)
        class_rows.extend(classes)
        predictions.extend(preds)
        cost_rows.extend(costs)
        matrices.update(matrix_records)

    holdout_rows, holdout_classes = run_grouped_holdout(frame, config, seeds)
    metric_rows.extend(holdout_rows)
    class_rows.extend(holdout_classes)

    metric_frame = pd.DataFrame(metric_rows)
    class_frame = pd.DataFrame(class_rows)
    prediction_frame = pd.DataFrame(predictions)
    cost_frame = pd.DataFrame(cost_rows)
    metric_frame.to_csv(output_dir / "fold_results.csv", index=False)
    class_frame.to_csv(output_dir / "per_class_results.csv", index=False)
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)

    group_columns = ["mapping", "feature_representation", "model", "evaluation"]
    aggregation = {
        metric: ["mean", "std", "min", "max"]
        for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted")
    }
    summary = metric_frame.groupby(group_columns, dropna=False).agg(aggregation)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output_dir / "metrics.csv", index=False)
    summary.reset_index().to_json(output_dir / "metrics.json", orient="records", indent=2)

    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)
    for (mapping, feature, model), seed_matrices in matrices.items():
        slug = "__".join((mapping, feature, model))
        (cm_dir / f"{slug}.json").write_text(json.dumps({"class_order": sorted(set(config["dataset"]["emotion_to_stress"].values())), "by_seed": seed_matrices}, indent=2), encoding="utf-8")

    cost_frame.to_csv(output_dir / "computational_cost.csv", index=False)
    pd.DataFrame(feature_cost).to_csv(output_dir / "feature_extraction_cost.csv", index=False)
    mappings = {"original": config["dataset"]["emotion_to_stress"], **alternate_mappings}
    distribution_rows = []
    for mapping_name, mapping in mappings.items():
        for emotion, class_id in mapping.items():
            distribution_rows.append({
                "mapping": mapping_name,
                "emotion": emotion,
                "class_id": int(class_id),
                "class_name": config["dataset"]["stress_labels"].get(str(class_id), f"Class {class_id}"),
                "sample_count": int((frame["emotion"] == emotion).sum()),
            })
    pd.DataFrame(distribution_rows).to_csv(output_dir / "class_distribution.csv", index=False)

    metadata = {
        "experiment_id": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16],
        "dataset_path": config["dataset"]["path"],
        "dataset_samples": len(frame),
        "unique_actors": int(frame["actor_id"].nunique()),
        "skipped_file_count": len(skipped),
        "feature_representations": {name: int(frame[f"{name}_dimensions"].iloc[0]) for name in FEATURE_NAMES},
        "seeds": list(seeds),
        "primary_evaluation": "StratifiedGroupKFold grouped by actor; five folds per seed; no actor overlap",
        "software": {"python": __import__("sys").version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "librosa": librosa.__version__},
        "config": config,
    }
    (output_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Speaker-independent RAVDESS emotion-derived stress experiments")
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--reuse-features", action="store_true", help="Reuse sample_features.csv in the output directory")
    args = parser.parse_args()
    config = read_config()
    args.results.mkdir(parents=True, exist_ok=True)
    feature_path = args.results / "sample_features.csv"
    if args.reuse_features and feature_path.exists():
        frame = pd.read_csv(feature_path)
        feature_cost = []
        skipped = pd.read_csv(args.results / "skipped_files.csv").to_dict("records") if (args.results / "skipped_files.csv").exists() else []
    else:
        frame, feature_cost, skipped = extract_dataset(config, args.results)
    write_outputs(args.results, frame, config, feature_cost, skipped, args)
    print(f"Completed grouped experiments: {len(frame)} samples, {frame['actor_id'].nunique()} actors, {len(skipped)} skipped files")
    print(f"Results: {args.results}")


if __name__ == "__main__":
    main()
