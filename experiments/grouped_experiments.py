import re
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

WORKSPACE = Path(__file__).resolve().parents[1]
DATASET = WORKSPACE / "ravdess-emotional-speech-audio"
RESULTS = WORKSPACE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def get_actor_from_filename(fname: str):
    # RAVDESS filenames include Actor_## as suffix in folder names; also filename parts contain actor id
    m = re.search(r"Actor_(\d{2})", str(fname))
    if m:
        return int(m.group(1))
    # fallback: try to parse filename parts
    base = Path(fname).name
    parts = base.split("-")
    if len(parts) >= 7:
        try:
            return int(parts[6])
        except Exception:
            return None
    return None


def _extract_grouped_xy(df: pd.DataFrame, label_col: str, filename_col: str):
    groups = df[filename_col].apply(get_actor_from_filename)
    if groups.isnull().any():
        raise ValueError('Could not parse some actor IDs from filenames')

    metadata_names = {
        label_col,
        filename_col,
        'actor',
        'actor_id',
        'sample_id',
        'emotion',
        'source_sample_rate',
        'sample_rate',
        'split',
        'fold',
        'seed',
    }
    metadata_columns = [
        column for column in df.columns
        if column in metadata_names or column.endswith('_dimensions')
    ]
    feature_df = df.drop(columns=metadata_columns, errors='ignore').select_dtypes(include=[np.number])
    if feature_df.empty:
        raise ValueError('No numeric feature columns found after excluding label and grouping metadata')

    return feature_df.to_numpy(), df[label_col].to_numpy(), groups.to_numpy()


def load_feature_table_segment_energy():
    # Reuse experiments' feature extraction: experiments/reviewer_experiments.py writes feature CSVs.
    # If those CSVs exist under results, load them; else signal error.
    csv = RESULTS / "feature_ablation_results.csv"
    # Instead we will reconstruct dataset by walking audio files and computing same segment-energy.
    # For speed, attempt to load precomputed per-file features if present.
    # The repo already has features implicit via experiments; we instead re-run minimal extraction using existing function.
    raise RuntimeError("This grouped runner expects precomputed feature CSV per-file; please run reviewer_experiments to generate per-file features first.")


def run_grouped_experiments(feature_csv_or_mode: str):
    # If input is an existing CSV, load it; otherwise treat it as a feature mode
    if Path(feature_csv_or_mode).exists():
        df = pd.read_csv(feature_csv_or_mode)
        # Expect a column 'filename' or 'file' that identifies the audio file and a label column
        label_col = None
        for c in ['label', 'stress_label', 'stress', 'y']:
            if c in df.columns:
                label_col = c
                break
        if label_col is None:
            raise ValueError('No label column found in feature CSV')

        if 'filename' not in df.columns and 'file' not in df.columns:
            raise ValueError('Expected feature CSV to include `filename` column for grouping by actor')

        fname_col = 'filename' if 'filename' in df.columns else 'file'

    else:
        # treat as mode string and generate features from raw RAVDESS files using reviewer_experiments functions
        # import reviewer_experiments by filepath to avoid package import issues
        import importlib.util
        rev_path = Path(__file__).resolve().parents[0] / 'reviewer_experiments.py'
        spec = importlib.util.spec_from_file_location('rev', str(rev_path))
        rev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rev)

        mode = feature_csv_or_mode
        files = rev.get_ravdess_files(rev.DATASET_PATH)
        feats = []
        labels = []
        filenames = []
        for file_path in files:
            try:
                audio, sr = rev.load_audio_file(file_path)
                if mode == 'standard_mfcc':
                    f = rev.build_standard_mfcc_features(audio, sr)
                elif mode == 'extended_acoustic':
                    f = rev.build_extended_acoustic_features(audio, sr)
                else:
                    f = rev.build_segment_energy_features(audio, sr)

                emotion = rev.emotion_from_ravdess_filename(file_path.name)
                label = rev.HEURISTIC_STRESS_MAPPING.get(emotion, 0)
                feats.append(np.asarray(f, dtype=float))
                labels.append(int(label))
                filenames.append(str(file_path))
            except Exception as exc:
                print(f"Skipping {file_path}: {exc}")

        # build DataFrame
        df_feats = pd.DataFrame(feats)
        df_feats['label'] = labels
        df_feats['filename'] = filenames
        df = df_feats
        label_col = 'label'
        fname_col = 'filename'

    X, y, groups = _extract_grouped_xy(df, label_col, fname_col)

    out = {}

    # grouped 80:20 split (actor-independent)
    gsplit = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gsplit.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5, class_weight='balanced', random_state=42)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }
    out['grouped_single_split'] = metrics
    pd.DataFrame([metrics]).to_csv(RESULTS / 'grouped_primary_split_results.csv', index=False)
    (RESULTS / 'grouped_primary_split_results.json').write_text(json.dumps(metrics, indent=2))

    # grouped 5-fold grouped CV (group kfold)
    gk = GroupKFold(n_splits=5)
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(gk.split(X, y, groups), start=1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        clf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5, class_weight='balanced', random_state=42)
        clf.fit(X_tr_s, y_tr)
        y_p = clf.predict(X_te_s)
        fold_metrics.append({
            'fold': fold,
            'accuracy': float(accuracy_score(y_te, y_p)),
            'precision': float(precision_score(y_te, y_p, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_te, y_p, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_te, y_p, average='weighted', zero_division=0)),
        })

    dfm = pd.DataFrame(fold_metrics)
    summary = {'mean_accuracy': float(dfm['accuracy'].mean()), 'std_accuracy': float(dfm['accuracy'].std())}
    dfm.to_csv(RESULTS / 'grouped_cross_validation_folds.csv', index=False)
    (RESULTS / 'grouped_cross_validation_summary.json').write_text(json.dumps(summary, indent=2))
    out['grouped_cv'] = {'folds': fold_metrics, 'summary': summary}

    return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True, help='Per-file feature CSV with filename and label columns')
    args = parser.parse_args()
    print('Running grouped experiments...')
    res = run_grouped_experiments(args.features)
    print('Done. Results saved to results/')
