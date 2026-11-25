import json
from pathlib import Path
import math

base = Path(__file__).resolve().parents[1] / 'results'
# load grouped primary
gp = json.load(open(base / 'grouped_primary_split_results.json'))
cm = gp['confusion_matrix']
cm = [[int(x) for x in row] for row in cm]
# compute per-class precision/recall/f1
num_classes = len(cm)
precisions = []
recalls = []
f1s = []
for i in range(num_classes):
    tp = cm[i][i]
    fp = sum(cm[r][i] for r in range(num_classes) if r!=i)
    fn = sum(cm[i][c] for c in range(num_classes) if c!=i)
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
    if prec+rec>0:
        f1 = 2*prec*rec/(prec+rec)
    else:
        f1 = 0.0
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)

macro_f1 = sum(f1s)/num_classes
weighted_f1 = sum(f1s[i]*sum(cm[i]) for i in range(num_classes))/sum(sum(row) for row in cm)
accuracy = sum(cm[i][i] for i in range(num_classes))/sum(sum(row) for row in cm)

out = {
    'per_class_precision':precisions,
    'per_class_recall':recalls,
    'per_class_f1':f1s,
    'macro_f1':macro_f1,
    'weighted_f1':weighted_f1,
    'accuracy':accuracy,
    'support_per_class':[sum(row) for row in cm],
}
print(json.dumps(out, indent=2))
json.dump(out, open(base / 'derived_grouped_primary_metrics.json','w'), indent=2)
print('wrote', base / 'derived_grouped_primary_metrics.json')
