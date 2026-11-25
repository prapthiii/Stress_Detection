from pathlib import Path
import json
import numpy as np
import pandas as pd
import importlib.util

REV_PATH = Path(__file__).resolve().parents[0] / 'reviewer_experiments.py'
spec = importlib.util.spec_from_file_location('rev', str(REV_PATH))
rev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rev)

RESULTS = Path(__file__).resolve().parents[1] / 'results'
RESULTS.mkdir(exist_ok=True)


def compute_single_split_testset(mode='segment_energy'):
    X, y = rev.extract_dataset(mode)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rev.RANDOM_STATE)
    return X_train, X_test, y_train, y_test


def fit_classifier_on_train(X_train, y_train, config=None):
    if config is None:
        config = {
            "n_estimators": 300,
            "max_depth": 20,
            "min_samples_split": 5,
            "class_weight": "balanced",
            "random_state": rev.RANDOM_STATE,
        }
    clf = rev.build_classifier(config)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    clf.fit(X_train_s, y_train)
    return clf, scaler


def preds_on_test(clf, scaler, X_test):
    X_test_s = scaler.transform(X_test)
    return clf.predict(X_test_s)


def bootstrap_paired_diff(obs_a, obs_b, n_boot=2000, seed=42):
    # obs_a and obs_b are arrays of predictions for the same test set; compute accuracy diff bootstrap
    rng = np.random.default_rng(seed)
    n = len(obs_a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        a = obs_a[idx]
        b = obs_b[idx]
        # need true labels too; we'll pass differences of indicator equalities outside
        diffs[i] = None  # placeholder
    return diffs


def bootstrap_accuracy_diff(y_true, pred_a, pred_b, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        a_acc = np.mean(y_true[idx] == pred_a[idx])
        b_acc = np.mean(y_true[idx] == pred_b[idx])
        diffs[i] = a_acc - b_acc
    return diffs


def two_sided_pvalue(diffs, observed):
    # proportion of bootstrap diffs as or more extreme than observed
    more_extreme = np.sum(np.abs(diffs) >= abs(observed))
    p = more_extreme / len(diffs)
    return p


def run_statistics():
    out = {}

    # MFCC vs segment-energy: use same train/test split
    Xs_tr, Xs_te, ys_tr, ys_te = compute_single_split_testset('segment_energy')
    Xm_tr, Xm_te, ym_tr, ym_te = compute_single_split_testset('standard_mfcc')
    # Ensure labels align
    assert len(ys_te) == len(ym_te)
    y_test = ys_te

    # Fit on their respective training sets
    clf_s, sc_s = fit_classifier_on_train(Xs_tr, ys_tr)
    clf_m, sc_m = fit_classifier_on_train(Xm_tr, ym_tr)

    pred_s = preds_on_test(clf_s, sc_s, Xs_te)
    pred_m = preds_on_test(clf_m, sc_m, Xm_te)

    acc_s = float(np.mean(pred_s == y_test))
    acc_m = float(np.mean(pred_m == y_test))
    obs_diff = acc_m - acc_s

    diffs = bootstrap_accuracy_diff(y_test, pred_m, pred_s, n_boot=2000, seed=rev.RANDOM_STATE)
    lower = float(np.percentile(diffs, 2.5))
    upper = float(np.percentile(diffs, 97.5))
    p = two_sided_pvalue(diffs, obs_diff)

    out['mfcc_vs_segment'] = {
        'mfcc_accuracy': acc_m,
        'segment_accuracy': acc_s,
        'observed_diff': obs_diff,
        '95ci_lower': lower,
        '95ci_upper': upper,
        'p_value': p,
    }

    # RF ablation: compare two RF configs on same segment-energy split
    X_tr, X_te, y_tr, y_te = compute_single_split_testset('segment_energy')
    cfg_a = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "class_weight": "balanced",
        "random_state": rev.RANDOM_STATE,
    }
    cfg_b = {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": rev.RANDOM_STATE,
    }
    clf_a, sc_a = fit_classifier_on_train(X_tr, y_tr, config=cfg_a)
    clf_b, sc_b = fit_classifier_on_train(X_tr, y_tr, config=cfg_b)
    pred_a = preds_on_test(clf_a, sc_a, X_te)
    pred_b = preds_on_test(clf_b, sc_b, X_te)
    acc_a = float(np.mean(pred_a == y_te))
    acc_b = float(np.mean(pred_b == y_te))
    obs_diff_rf = acc_b - acc_a
    diffs_rf = bootstrap_accuracy_diff(y_te, pred_b, pred_a, n_boot=2000, seed=rev.RANDOM_STATE)
    lower_rf = float(np.percentile(diffs_rf, 2.5))
    upper_rf = float(np.percentile(diffs_rf, 97.5))
    p_rf = two_sided_pvalue(diffs_rf, obs_diff_rf)
    out['rf_ablation'] = {
        'acc_model_a': acc_a,
        'acc_model_b': acc_b,
        'observed_diff': obs_diff_rf,
        '95ci_lower': lower_rf,
        '95ci_upper': upper_rf,
        'p_value': p_rf,
    }

    # Macro-F1 and confusion matrices for primary results (single-split segment and mfcc)
    from sklearn.metrics import f1_score, confusion_matrix
    macro_f1_s = float(f1_score(y_test, pred_s, average='macro', zero_division=0))
    cm_s = confusion_matrix(y_test, pred_s).tolist()
    macro_f1_m = float(f1_score(y_test, pred_m, average='macro', zero_division=0))
    cm_m = confusion_matrix(y_test, pred_m).tolist()
    out['primary_reports'] = {
        'segment': {'macro_f1': macro_f1_s, 'confusion_matrix': cm_s, 'accuracy': acc_s},
        'mfcc': {'macro_f1': macro_f1_m, 'confusion_matrix': cm_m, 'accuracy': acc_m},
    }

    # write results
    (RESULTS / 'statistics_summary.json').write_text(json.dumps(out, indent=2))
    print('Wrote statistics_summary.json')

    return out


if __name__ == '__main__':
    res = run_statistics()
    # Update paper with a brief summary insertion
    paper = Path(__file__).resolve().parents[1] / 'paper' / 'Stress_Management_Paper.md'
    if paper.exists():
        text = paper.read_text(encoding='utf-8')
        insert_lines = []
        insert_lines.append('\n**Statistical Testing:** Bootstrap 95% CI and two-sided p-values were computed by resampling the held-out test set (n=288) 2,000 times. For MFCC vs segment-energy the observed accuracy difference (MFCC - segment) = ' + f"{res['mfcc_vs_segment']['observed_diff']:.4f}" + f" (95% CI [{res['mfcc_vs_segment']['95ci_lower']:.4f}, {res['mfcc_vs_segment']['95ci_upper']:.4f}], p={res['mfcc_vs_segment']['p_value']:.3f})." )
        insert_lines.append('\nFor the Random Forest ablation (Model_B - Model_A) the observed accuracy difference = ' + f"{res['rf_ablation']['observed_diff']:.4f}" + f" (95% CI [{res['rf_ablation']['95ci_lower']:.4f}, {res['rf_ablation']['95ci_upper']:.4f}], p={res['rf_ablation']['p_value']:.3f})." )
        # append near Results section
        new_text = text.replace('## 4. Experimental Results (Corrected)', '## 4. Experimental Results (Corrected)\n' + '\n'.join(insert_lines))
        paper.write_text(new_text, encoding='utf-8')
        print('Paper updated with statistics summary')
