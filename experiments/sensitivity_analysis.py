import json
from pathlib import Path
import numpy as np
import pandas as pd

import importlib.util
rev_path = Path(__file__).resolve().parents[0] / 'reviewer_experiments.py'
spec = importlib.util.spec_from_file_location('rev', str(rev_path))
rev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rev)

RESULTS = Path(__file__).resolve().parents[1] / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)


ALTERNATE_MAPPINGS = {
    'alt_sad_to_mild': dict(rev.HEURISTIC_STRESS_MAPPING, **{'sad': 1}),
    'alt_calm_to_mild': dict(rev.HEURISTIC_STRESS_MAPPING, **{'calm': 1}),
    'alt_sad_calm_to_mild': dict(rev.HEURISTIC_STRESS_MAPPING, **{'sad': 1, 'calm': 1}),
}

FEATURE_MODES = ['segment_energy', 'standard_mfcc']


def run():
    rows = []
    for name, mapping in ALTERNATE_MAPPINGS.items():
        print(f'Running mapping: {name}')
        # monkeypatch mapping
        orig_map = rev.HEURISTIC_STRESS_MAPPING.copy()
        rev.HEURISTIC_STRESS_MAPPING = mapping

        for mode in FEATURE_MODES:
            print(f'  feature mode: {mode}')
            X, y = rev.extract_dataset(mode)
            metrics, y_test, y_pred = rev.evaluate_classifier(X, y, config={
                "n_estimators": 300,
                "max_depth": 20,
                "min_samples_split": 5,
                "class_weight": "balanced",
                "random_state": rev.RANDOM_STATE,
            })
            row = {'mapping': name, 'feature_mode': mode}
            row.update({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating, np.integer))})
            rows.append(row)

        # restore
        rev.HEURISTIC_STRESS_MAPPING = orig_map

    df = pd.DataFrame(rows)
    out_csv = RESULTS / 'sensitivity_analysis_results.csv'
    df.to_csv(out_csv, index=False)
    (RESULTS / 'sensitivity_analysis_results.json').write_text(df.to_json(orient='records', indent=2))
    print('Wrote', out_csv)


if __name__ == '__main__':
    run()
