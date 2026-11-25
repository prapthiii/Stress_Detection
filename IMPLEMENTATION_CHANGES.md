# Implementation Changes

## Files modified
- app.py
- research_config.json
- tests/test_pss10.py

## Files created
- experiments/reviewer_experiments.py
- feature_ablation_results.csv
- feature_ablation_results.json
- cross_validation_results.csv
- cross_validation_folds.csv
- baseline_comparison.csv
- random_forest_ablation.csv
- experiment_summary.md
- text_only_evaluation_status.txt
- modality_evaluation_status.txt
- pss10_validation_report.txt

## What was changed
- Added a central experiment configuration record for the reviewer deliverables.
- Added an experiment driver that documents and evaluates the current segment-energy baseline, the standard MFCC ablation, the 5-fold CV protocol, RF ablation, and majority baseline.
- Added a status-only evaluation package for text-only and multimodal validation because the repository does not contain paired labelled text samples.
- Added a PSS-10 validation note and direct Python scoring verification evidence.
- Added UI wording that calls the current speech pipeline a heuristic stress estimate and warns that the system is research-only and not clinical.

## Experiments added
- Standard MFCC ablation: feature_ablation_results.csv
- Stratified cross-validation: cross_validation_results.csv and cross_validation_folds.csv
- Majority baseline comparison: baseline_comparison.csv
- Random Forest ablation: random_forest_ablation.csv
- Text-only status report: text_only_evaluation_status.txt
- Multimodal/fusion status report: modality_evaluation_status.txt
- PSS-10 validation report: pss10_validation_report.txt

## Corrections and numeric updates
- Corrected data-splitting: switched to stratified 80:20 splits to avoid accidental class-skew in the test set. This reduced the original unstratified reported accuracy (57%) to the corrected segment-energy baseline (37.5%).
- Added 5-fold stratified CV reporting: mean accuracy = 44.4% ± 1.05% for the segment-energy baseline.
- Documented that the corrected segment-energy baseline is at/near the majority-class baseline (37.5% vs. 39.9%), and surfaced this in the Abstract and summary deliverables as an honesty signal.

## Commands needed to run experiments
- python experiments/reviewer_experiments.py
- python -m streamlit run app.py --server.address localhost --server.port 8501

## Output files generated
- feature_ablation_results.csv
- feature_ablation_results.json
- cross_validation_results.csv
- cross_validation_folds.csv
- baseline_comparison.csv
- random_forest_ablation.csv
- experiment_summary.md
- text_only_evaluation_status.txt
- modality_evaluation_status.txt
- pss10_validation_report.txt

## Limitations
- There is no labelled text corpus inside the current RAVDESS dataset package for text-only classification.
- There is no legitimate paired audio/text dataset with shared stress labels in the repository.
- The existing OCR capability is runtime/library dependent and can fail safely when OCR engines are unavailable.
