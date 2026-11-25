from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]

ab = pd.read_csv(root / 'feature_ablation_results.csv')
cv = pd.read_csv(root / 'cross_validation_results.csv')
base = pd.read_csv(root / 'baseline_comparison.csv')
rf = pd.read_csv(root / 'random_forest_ablation.csv')

segment = ab.loc[ab['Feature Representation'] == 'Segment-energy coefficients'].iloc[0]
mfcc = ab.loc[ab['Feature Representation'] == 'Standard MFCC'].iloc[0]

summary = []
summary.append('# Experiment Summary\n')
summary.append('## Dataset\n')
summary.append('The repository currently contains a RAVDESS speech subset under the workspace folder `ravdess-emotional-speech-audio`. The same dataset is used for the experiment scripts. The heuristic emotion-to-stress mapping is defined in `research_config.json`.\n')
summary.append('## Label Mapping\n')
summary.append('Heuristic emotion-to-stress mapping: Neutral -> No Stress; Calm -> No Stress; Happy -> Mild Stress; Surprised -> Mild Stress; Sad -> Moderate Stress; Angry -> High Stress; Fearful -> High Stress; Disgust -> High Stress.\n')
summary.append('## Feature Extraction\n')
summary.append('The baseline implementation in `app.py` calculates a 13-dimensional segment-energy coefficient vector through coefficient_i = mean(audio[i::13]), then appends energy and zero-crossing rate, yielding a 15-dimensional voice feature vector. This is documented as a segment-energy coefficient representation and is not standard MFCC extraction. The standard MFCC ablation uses librosa.feature.mfcc with sr=22050, n_mfcc=13, n_fft=2048, hop_length=512, Hann window and mean aggregation over time.\n')
summary.append('## Majority Baseline\n')
summary.append(f"Majority class: {base.iloc[0]['Majority_Class_Name']} ({int(base.iloc[0]['Majority_Class'])}); number of samples in the held-out test set: {int(base.iloc[0]['Number_of_Samples'])}; percentage: {float(base.iloc[0]['Percentage_of_Test_Samples']):.2f}%. The RF test accuracy was {float(base.iloc[0]['Random_Forest_Test_Accuracy']):.4f}, compared with the majority baseline of {float(base.iloc[0]['Percentage_of_Test_Samples']):.2f}%.\n")
summary.append('## Original 80:20 result\n')
summary.append(f"The reproducible segment-energy baseline result extracted from the active project pipeline gives accuracy {float(segment['Accuracy']):.6f} with weighted precision/recall/F1 of {float(segment['Precision']):.6f}/{float(segment['Recall']):.6f}/{float(segment['F1']):.6f}.\n")
summary.append('## 5-fold cross-validation\n')
acc = cv.loc[cv['Metric'] == 'Accuracy'].iloc[0]
wf1 = cv.loc[cv['Metric'] == 'Weighted_F1'].iloc[0]
nof1 = cv.loc[cv['Metric'] == 'Class_No Stress_F1'].iloc[0]
msf1 = cv.loc[cv['Metric'] == 'Class_Mild Stress_F1'].iloc[0]
modf1 = cv.loc[cv['Metric'] == 'Class_Moderate Stress_F1'].iloc[0]
hif1 = cv.loc[cv['Metric'] == 'Class_High Stress_F1'].iloc[0]
summary.append(f"Accuracy mean ± std: {float(acc['Mean']):.6f} ± {float(acc['Std']):.6f}; weighted F1 mean ± std: {float(wf1['Mean']):.6f} ± {float(wf1['Std']):.6f}; No Stress F1 mean ± std: {float(nof1['Mean']):.6f} ± {float(nof1['Std']):.6f}; Mild Stress F1 mean ± std: {float(msf1['Mean']):.6f} ± {float(msf1['Std']):.6f}; Moderate Stress F1 mean ± std: {float(modf1['Mean']):.6f} ± {float(modf1['Std']):.6f}; High Stress F1 mean ± std: {float(hif1['Mean']):.6f} ± {float(hif1['Std']):.6f}.\n")
summary.append('## Standard MFCC ablation\n')
summary.append(f"Standard MFCC representation: accuracy {float(mfcc['Accuracy']):.6f}; precision {float(mfcc['Precision']):.6f}; recall {float(mfcc['Recall']):.6f}; F1 {float(mfcc['F1']):.6f}. Segment-energy coefficients baseline: accuracy {float(segment['Accuracy']):.6f}; F1 {float(segment['F1']):.6f}.\n")
summary.append('## Segment-energy feature results\n')
summary.append(f"The baseline feature representation in the current runtime is the segment-energy coefficient vector with 13 coefficients plus energy and ZCR; the 80:20 test result from the CSV is Accuracy {float(segment['Accuracy']):.6f}, Precision {float(segment['Precision']):.6f}, Recall {float(segment['Recall']):.6f}, F1 {float(segment['F1']):.6f}.\n")
summary.append('## Random Forest ablation\n')
model_a = rf.loc[rf['Configuration'] == 'Model_A_simple_RF'].iloc[0]
model_b = rf.loc[rf['Configuration'] == 'Model_B_current_RF'].iloc[0]
summary.append(f"Model A small RF: accuracy {float(model_a['Accuracy']):.6f}; F1 {float(model_a['F1']):.6f}; Model B current RF: accuracy {float(model_b['Accuracy']):.6f}; F1 {float(model_b['F1']):.6f}.\n")
summary.append('## Text-only evaluation status/results\n')
summary.append('Text-only classification could not be quantitatively evaluated because the available RAVDESS dataset does not contain corresponding text samples with stress labels. The application continues to support text analysis via NLTK VADER and TF-IDF-style feature extraction.\n')
summary.append('## Multimodal/fusion evaluation status/results\n')
summary.append('Multimodal fusion is implemented as a system capability in the Streamlit application, but it is not quantitatively validated because the repository does not contain paired labelled audio/text samples with common stress labels.\n')
summary.append('## PSS-10 validation\n')
summary.append('The PSS-10 scoring contract is implemented in `src/questionnaire.py` with 10 questions, reverse-scored items 3, 4, 6, 7, and a 0-40 score range. The validation note is stored in `pss10_validation_report.txt`.\n')
summary.append('## Limitations\n')
summary.append('The corpus is speech-only and no textual stress-labelled RAVDESS alignment is present. No multimodal label ground truth is available. The classifier is a heuristic/feature-prototype pipeline and its outputs are not clinical predictions.\n')
summary.append('## Reproducibility information\n')
summary.append('The reproducible settings are recorded in `research_config.json`, including random_state=42, an 80:20 train/test split, 5-fold stratified CV, segment-energy baseline features, standard MFCC ablation features, and the Random Forest model configuration.\n')

(root / 'experiment_summary.md').write_text(''.join(summary), encoding='utf-8')
print('experiment_summary.md regenerated from actual output CSV files')
