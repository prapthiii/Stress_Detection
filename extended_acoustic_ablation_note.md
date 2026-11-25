# Extended acoustic feature ablation note

## Results
Extended acoustic features accuracy: 0.673611; precision: 0.676147; recall: 0.673611; F1: 0.632470.
Standard MFCC accuracy from the existing ablation: 0.652778.
Relative to those standard-MFCC results the extended acoustic representation improved on held-out evaluation accuracy.

## Per-class results
Class 0 (No Stress): precision=0.685714, recall=0.827586, F1=0.750000.
Class 1 (Mild Stress): precision=0.707692, recall=0.597403, F1=0.647887.
Class 2 (Moderate Stress): precision=0.666667, recall=0.052632, F1=0.097561.
Class 3 (High Stress): precision=0.653333, recall=0.852174, F1=0.739623.

## Interpretation
This note follows the measured accuracy comparison only. It does not claim a clinical or externally validated improvement beyond the experiment output shown above.