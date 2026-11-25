from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]


COMPARISONS = [
    ("original", "strided_mean", "random_forest_balanced", "original", "mfcc", "random_forest_balanced", "MFCC - strided mean"),
    ("original", "mfcc", "random_forest_balanced", "original", "extended_acoustic", "random_forest_balanced", "Extended - MFCC"),
    ("original", "mfcc", "random_forest_balanced", "original", "mfcc", "logistic_regression", "Logistic Regression - balanced RF"),
    ("original", "mfcc", "random_forest_balanced", "original", "mfcc", "svm", "SVM - balanced RF"),
    ("original", "mfcc", "random_forest_balanced", "original", "mfcc", "gradient_boosting", "Gradient Boosting - balanced RF"),
]


def actor_cluster_bootstrap(predictions, first_key, second_key, n_boot=1000, seed=42):
    """Paired percentile interval by resampling actors, preserving all rows per actor."""
    first = predictions[
        (predictions["mapping"] == first_key[0])
        & (predictions["feature_representation"] == first_key[1])
        & (predictions["model"] == first_key[2])
    ]
    second = predictions[
        (predictions["mapping"] == second_key[0])
        & (predictions["feature_representation"] == second_key[1])
        & (predictions["model"] == second_key[2])
    ]
    seeds = sorted(set(first["seed"]) & set(second["seed"]))
    if not seeds:
        raise ValueError("No matched seeds are available for the paired comparison")

    paired = {}
    actors = None
    for current_seed in seeds:
        left = first[first["seed"] == current_seed].sort_values(["actor_id", "sample_id"])
        right = second[second["seed"] == current_seed].sort_values(["actor_id", "sample_id"])
        if not np.array_equal(left[["actor_id", "sample_id", "y_true"]].to_numpy(), right[["actor_id", "sample_id", "y_true"]].to_numpy()):
            raise ValueError("Compared conditions do not share the same held-out samples")
        current_actors = sorted(left["actor_id"].unique())
        actors = current_actors if actors is None else sorted(set(actors) & set(current_actors))
        paired[current_seed] = {
            "first": {actor: left[left["actor_id"] == actor] for actor in current_actors},
            "second": {actor: right[right["actor_id"] == actor] for actor in current_actors},
        }

    if not actors:
        raise ValueError("No shared actor clusters are available")

    def mean_difference(sampled_actors):
        seed_differences = []
        for current_seed in seeds:
            left_rows = pd.concat([paired[current_seed]["first"][actor] for actor in sampled_actors], ignore_index=True)
            right_rows = pd.concat([paired[current_seed]["second"][actor] for actor in sampled_actors], ignore_index=True)
            labels = left_rows["y_true"].to_numpy()
            labels_order = sorted(set(labels) | set(right_rows["y_true"]))
            left_score = f1_score(labels, left_rows["y_pred"], labels=labels_order, average="macro", zero_division=0)
            right_score = f1_score(labels, right_rows["y_pred"], labels=labels_order, average="macro", zero_division=0)
            seed_differences.append(right_score - left_score)
        return float(np.mean(seed_differences))

    estimate = mean_difference(actors)
    rng = np.random.default_rng(seed)
    bootstrap_values = np.asarray(
        [mean_difference(rng.choice(actors, size=len(actors), replace=True).tolist()) for _ in range(n_boot)],
        dtype=float,
    )
    return {
        "estimate_macro_f1_difference": estimate,
        "ci_95_lower": float(np.percentile(bootstrap_values, 2.5)),
        "ci_95_upper": float(np.percentile(bootstrap_values, 97.5)),
        "bootstrap_replicates": int(n_boot),
        "bootstrap_unit": "actor cluster",
        "inference": "paired; same held-out samples per seed",
    }


def run(results_dir: Path, n_boot=1000, seed=42):
    predictions_path = results_dir / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Grouped out-of-fold predictions not found: {predictions_path}")
    predictions = pd.read_csv(predictions_path)
    rows = []
    for first_map, first_feature, first_model, second_map, second_feature, second_model, label in COMPARISONS:
        result = actor_cluster_bootstrap(
            predictions,
            (first_map, first_feature, first_model),
            (second_map, second_feature, second_model),
            n_boot=n_boot,
            seed=seed,
        )
        rows.append({"comparison": label, **result})
    output = pd.DataFrame(rows)
    output.to_csv(results_dir / "statistical_tests.csv", index=False)
    return output


def main():
    parser = argparse.ArgumentParser(description="Paired actor-cluster bootstrap comparisons for grouped predictions")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(run(args.results, args.replicates, args.seed).to_string(index=False))


if __name__ == "__main__":
    main()
