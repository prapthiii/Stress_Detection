from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _save(fig, output_dir, name):
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def generate(results_dir: Path, figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(results_dir / "metrics.csv")
    folds = pd.read_csv(results_dir / "fold_results.csv")
    classes = pd.read_csv(results_dir / "per_class_results.csv")
    distribution = pd.read_csv(results_dir / "class_distribution.csv")
    costs = pd.read_csv(results_dir / "computational_cost.csv")

    primary = metrics[(metrics.mapping == "original") & (metrics.evaluation == "stratified_group_kfold")]
    selected_rf = primary[primary.model == "random_forest_balanced"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    dist = distribution[distribution.mapping == "original"].groupby("class_name", as_index=False).sample_count.sum()
    ax.bar(dist.class_name, dist.sample_count, color="#2a6f97")
    ax.set_ylabel("Recordings")
    ax.set_title("Class Distribution Under Original Heuristic Mapping")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, figures_dir, "class_distribution")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    selected_rf = selected_rf.sort_values("feature_representation")
    x = np.arange(len(selected_rf))
    ax.bar(x - 0.18, selected_rf.accuracy_mean, width=0.36, label="Accuracy", color="#287271")
    ax.bar(x + 0.18, selected_rf.f1_macro_mean, width=0.36, label="Macro-F1", color="#e09f3e")
    ax.set_xticks(x, selected_rf.feature_representation, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean grouped-CV score")
    ax.set_title("Acoustic Representation Comparison: Balanced Random Forest")
    ax.legend(frameon=False)
    _save(fig, figures_dir, "feature_representation_comparison")

    mfcc_models = primary[primary.feature_representation == "mfcc"].sort_values("model")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(mfcc_models))
    ax.bar(x - 0.18, mfcc_models.accuracy_mean, width=0.36, label="Accuracy", color="#287271")
    ax.bar(x + 0.18, mfcc_models.f1_macro_mean, width=0.36, label="Macro-F1", color="#e09f3e")
    ax.set_xticks(x, mfcc_models.model, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean grouped-CV score")
    ax.set_title("Model Comparison on Standard MFCC Features")
    ax.legend(frameon=False)
    _save(fig, figures_dir, "model_comparison")

    primary_classes = classes[
        (classes.mapping == "original")
        & (classes.evaluation == "stratified_group_kfold")
        & (classes.model == "random_forest_balanced")
    ].groupby(["feature_representation", "class_name"], as_index=False).f1.mean()
    pivot = primary_classes.pivot(index="class_name", columns="feature_representation", values="f1")
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean per-class F1")
    ax.set_title("Per-Class F1 Across Acoustic Representations")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Representation", frameon=False, ncol=2)
    _save(fig, figures_dir, "per_class_f1_comparison")

    fold_cv = folds[(folds.mapping == "original") & (folds.evaluation == "stratified_group_kfold") & (folds.model == "random_forest_balanced")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for metric, label, color in (("accuracy", "Accuracy", "#287271"), ("f1_macro", "Macro-F1", "#e09f3e")):
        curve = fold_cv.groupby(["feature_representation", "fold"], as_index=False)[metric].mean()
        for representation in ["strided_mean", "mfcc", "extended_acoustic"]:
            subset = curve[curve.feature_representation == representation]
            ax.plot(subset.fold, subset[metric], marker="o", label=f"{representation}: {label}", color=color, alpha=0.85 if representation == "mfcc" else 0.6, linestyle="-" if representation == "mfcc" else "--")
    ax.set_xticks(sorted(fold_cv.fold.unique()))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Grouped fold")
    ax.set_ylabel("Mean score across seeds")
    ax.set_title("Grouped-Fold Accuracy and Macro-F1")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    _save(fig, figures_dir, "grouped_fold_scores")

    sensitivity = metrics[(metrics.mapping != "original") & (metrics.evaluation == "stratified_group_kfold") & (metrics.model == "random_forest_balanced")]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = [f"{row.mapping}\n{row.feature_representation}" for row in sensitivity.itertuples()]
    ax.bar(np.arange(len(sensitivity)), sensitivity.f1_macro_mean, color="#8a5a44")
    ax.set_xticks(np.arange(len(sensitivity)), labels, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean grouped-CV macro-F1")
    ax.set_title("Sensitivity to Alternative Heuristic Emotion Mappings")
    _save(fig, figures_dir, "label_mapping_sensitivity")

    model_cost = costs.groupby("model", as_index=False).agg(training_seconds=("training_seconds", "mean"), inference_seconds_per_sample=("inference_seconds_per_sample", "mean"), model_size_bytes=("model_size_bytes", "mean"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, metric, title, color in zip(
        axes,
        ["training_seconds", "inference_seconds_per_sample", "model_size_bytes"],
        ["Fit Time (s)", "Inference Time / Sample (s)", "Serialized Model Size (bytes)"],
        ["#287271", "#e09f3e", "#8a5a44"],
    ):
        ax.bar(model_cost.model, model_cost[metric], color=color)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=70, labelsize=7)
    fig.suptitle("Grouped Evaluation Computational Cost")
    _save(fig, figures_dir, "computational_cost")

    cm_path = results_dir / "confusion_matrices" / "original__mfcc__random_forest_balanced.json"
    with cm_path.open(encoding="utf-8") as file:
        confusion = json.load(file)
    matrices = [np.asarray(matrix) for matrix in confusion["by_seed"].values()]
    cm = np.mean(matrices, axis=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Mean count across seed-level OOF matrices")
    labels_order = ["No Stress", "Mild Stress", "Moderate Stress", "High Stress"]
    ax.set_xticks(range(len(labels_order)), labels_order, rotation=25, ha="right")
    ax.set_yticks(range(len(labels_order)), labels_order)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Emotion-derived label")
    ax.set_title("Grouped-CV Confusion Matrix: Balanced RF + MFCC")
    for row in range(cm.shape[0]):
        for column in range(cm.shape[1]):
            ax.text(column, row, f"{cm[row, column]:.1f}", ha="center", va="center", color="black")
    _save(fig, figures_dir, "primary_confusion_matrix")

    feature_cost = pd.read_csv(results_dir / "feature_extraction_cost.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(feature_cost.feature_representation, feature_cost.extraction_seconds_per_sample, color="#2a6f97")
    ax.set_ylabel("Seconds per recording")
    ax.set_title("Feature Extraction Cost")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, figures_dir, "feature_extraction_cost")


def main():
    parser = argparse.ArgumentParser(description="Generate result-backed research figures")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    generate(args.results, args.figures)
    print(f"Figures written to {args.figures}")


if __name__ == "__main__":
    main()
