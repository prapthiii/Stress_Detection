# Experiment Summary

## Dataset
The repository currently contains a RAVDESS speech subset under the workspace folder `ravdess-emotional-speech-audio`. The same dataset is used for the experiment scripts. The heuristic emotion-to-stress mapping is defined in `research_config.json`.

## Label Mapping
Heuristic emotion-to-stress mapping: Neutral -> No Stress; Calm -> No Stress; Happy -> Mild Stress; Surprised -> Mild Stress; Sad -> Moderate Stress; Angry -> High Stress; Fearful -> High Stress; Disgust -> High Stress.

## Feature Extraction
The baseline implementation in `app.py` calculates a 13-dimensional segment-energy coefficient vector through coefficient_i = mean(audio[i::13]), then appends energy and zero-crossing rate, yielding a 15-dimensional feature vector. This is documented as a segment-energy coefficient representation and must not be described as a standard MFCC implementation.

## Evaluation Status
The current repository supports text-only sentiment analysis and OCR, but the available RAVDESS dataset does not provide corresponding text labels. Text-only and multimodal evaluation therefore remain status-only reports rather than quantitative claims.

## Current Outputs
The following CSV artefacts are produced by the reproducible experiments when they are run: feature_ablation_results.csv, cross_validation_results.csv, cross_validation_folds.csv, baseline_comparison.csv, random_forest_ablation.csv.

## Reproducibility
The experiment configuration is saved in `research_config.json` with random_state = 42, 80:20 train/test methodology, 5-fold stratified CV and the Random Forest configuration used in the nominal baseline.