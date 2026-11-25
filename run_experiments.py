import argparse
from pathlib import Path
import pandas as pd
from src.pipeline import Pipeline


def main(feature_files, names=None, results_dir="results"):
    p = Pipeline(results_dir=results_dir)
    summary_rows = []
    for i, feat in enumerate(feature_files):
        name = names[i] if names and i < len(names) else Path(feat).stem
        print(f"Running experiments for {name} ({feat})")
        single = p.run_single_split(feat, save_model=False)
        cv = p.run_cross_validation(feat)
        maj = p.majority_class_baseline(feat)

        row = {
            "representation": name,
            "single_accuracy": single["metrics"]["accuracy"],
            "cv_mean_accuracy": cv["summary"]["mean_accuracy"],
            "cv_std_accuracy": cv["summary"]["std_accuracy"],
            "single_f1": single["metrics"]["f1"],
            "cv_mean_f1": cv["summary"]["mean_f1"],
            "cv_std_f1": cv["summary"]["std_f1"],
            "majority_dataset": maj["dataset_majority"],
        }
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(results_dir) / "feature_ablation_results.csv", index=False)
    df.to_json(Path(results_dir) / "feature_ablation_results.json", orient="records", indent=2)
    print("Wrote feature ablation summary to results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all experiments over feature CSVs")
    parser.add_argument("--features", nargs="+", required=True, help="One or more feature CSV files to evaluate")
    parser.add_argument("--names", nargs="*", help="Optional names for the representations (in same order as --features)")
    parser.add_argument("--results", default="results", help="Results directory")
    args = parser.parse_args()
    main(args.features, names=args.names, results_dir=args.results)
