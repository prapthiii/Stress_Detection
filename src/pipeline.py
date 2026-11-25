import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib


DEFAULT_RESULTS_DIR = Path("results")


def ensure_results_dir(path=DEFAULT_RESULTS_DIR):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _choose_label_column(df: pd.DataFrame):
    for candidate in ["label", "stress_label", "stress", "y"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("Could not find label column. Expected one of: label, stress_label, stress, y")


class Pipeline:
    def __init__(self, results_dir: str = "results", random_state: int = 42):
        self.results_dir = ensure_results_dir(results_dir)
        self.random_state = random_state

    def load_feature_table(self, csv_path: str):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Feature CSV not found at {csv_path}")
        df = pd.read_csv(csv_path)
        return df

    def _extract_Xy(self, df: pd.DataFrame):
        label_col = _choose_label_column(df)
        y = df[label_col].values
        X = df.drop(columns=[label_col])
        # drop any non-numeric columns if present
        X = X.select_dtypes(include=[np.number])
        return X.values, y

    def run_single_split(self, feature_csv: str, test_size: float = 0.2, save_model: bool = True):
        df = self.load_feature_table(feature_csv)
        X, y = self._extract_Xy(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = RandomForestClassifier(random_state=self.random_state)
        clf.fit(X_train_scaled, y_train)

        y_pred = clf.predict(X_test_scaled)
        metrics = self._compute_metrics(y_test, y_pred)

        results = {
            "split": "single",
            "test_size": test_size,
            "n_samples": int(len(y)),
            "metrics": metrics,
        }

        # save model and scaler for UI use
        if save_model:
            joblib.dump(clf, self.results_dir / "stress_model.pkl")
            joblib.dump(scaler, self.results_dir / "scaler.pkl")

        # write results
        (self.results_dir / "primary_split_results.json").write_text(json.dumps(results, indent=2))
        pd.DataFrame([metrics]).to_csv(self.results_dir / "primary_split_results.csv", index=False)
        return results

    def run_cross_validation(self, feature_csv: str, n_splits: int = 5):
        df = self.load_feature_table(feature_csv)
        X, y = self._extract_Xy(df)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        fold_metrics = []
        fold_idx = 0
        for train_idx, test_idx in skf.split(X, y):
            fold_idx += 1
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = RandomForestClassifier(random_state=self.random_state)
            clf.fit(X_train_scaled, y_train)

            y_pred = clf.predict(X_test_scaled)
            metrics = self._compute_metrics(y_test, y_pred)
            metrics["fold"] = fold_idx
            fold_metrics.append(metrics)

        # aggregate
        dfm = pd.DataFrame(fold_metrics)
        summary = {
            "n_splits": n_splits,
            "n_samples": int(len(y)),
            "mean_accuracy": float(dfm["accuracy"].mean()),
            "std_accuracy": float(dfm["accuracy"].std()),
            "mean_f1": float(dfm["f1"].mean()),
            "std_f1": float(dfm["f1"].std()),
        }

        (self.results_dir / "cross_validation_folds.csv").write_text(dfm.to_csv(index=False))
        (self.results_dir / "cross_validation_results.csv").write_text(json.dumps(summary, indent=2))
        return {"folds": fold_metrics, "summary": summary}

    def majority_class_baseline(self, feature_csv: str):
        df = self.load_feature_table(feature_csv)
        _, y = self._extract_Xy(df)
        # majority class accuracy on full dataset
        (values, counts) = np.unique(y, return_counts=True)
        maj_acc = float(counts.max() / counts.sum())
        # compute on a held-out stratified split to be comparable
        X = df.drop(columns=[_choose_label_column(df)])
        X = X.select_dtypes(include=[np.number]).values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        # majority class in train
        vals, cnts = np.unique(y_train, return_counts=True)
        maj_label = vals[np.argmax(cnts)]
        maj_pred = np.full_like(y_test, fill_value=maj_label)
        metrics = self._compute_metrics(y_test, maj_pred)
        out = {"dataset_majority": maj_acc, "test_majority_metrics": metrics}
        (self.results_dir / "majority_baseline_comparison.json").write_text(json.dumps(out, indent=2))
        pd.DataFrame([metrics]).to_csv(self.results_dir / "majority_baseline_comparison.csv", index=False)
        return out

    def _compute_metrics(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()
        return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "confusion_matrix": cm}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run standardized experiments on a feature CSV")
    parser.add_argument("--features", type=str, required=True, help="Path to feature CSV (numeric features + label column)")
    parser.add_argument("--action", type=str, choices=["single", "cv", "majority", "all"], default="all")
    parser.add_argument("--results", type=str, default="results")
    args = parser.parse_args()

    p = Pipeline(results_dir=args.results)
    if args.action in ("single", "all"):
        print("Running single split...")
        p.run_single_split(args.features)
    if args.action in ("cv", "all"):
        print("Running cross-validation...")
        p.run_cross_validation(args.features)
    if args.action in ("majority", "all"):
        print("Computing majority baseline...")
        p.majority_class_baseline(args.features)
