from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import numpy as np
import pandas as pd


class Evaluator:
    """Evaluator that computes multiple metrics and exports confusion matrices.

    - Supports macro and weighted averages
    - Returns per-class precision/recall/f1
    - Can save confusion matrix as CSV
    """

    def evaluate(self, y_true, y_pred, labels=None):
        acc = accuracy_score(y_true, y_pred)
        prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # per-class report dict via sklearn
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

        return {
            "accuracy": float(acc),
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(prec_weighted),
            "recall_weighted": float(rec_weighted),
            "f1_weighted": float(f1_weighted),
            "confusion_matrix": cm.tolist(),
            "per_class": report,
        }

    def save_confusion_matrix(self, y_true, y_pred, path, labels=None):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        labels = labels if labels is not None else np.unique(np.concatenate([np.array(y_true), np.array(y_pred)]))
        df = pd.DataFrame(cm, index=labels, columns=labels)
        df.to_csv(path)
        return df


if __name__ == "__main__":
    ev = Evaluator()
    y_true = [0, 1, 2, 0, 1]
    y_pred = [0, 1, 1, 0, 2]
    metrics = ev.evaluate(y_true, y_pred)
    print("Metrics:", metrics)