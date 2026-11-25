from pathlib import Path
import re, json
import numpy as np

WORK = Path(__file__).resolve().parents[1]
DATA = WORK / 'ravdess-emotional-speech-audio'

actors = []
if DATA.exists():
    for p in sorted(DATA.iterdir()):
        if p.is_dir():
            actors.append(p.name)

print('actor_dirs_count=', len(actors))
actor_nums = []
for a in actors:
    m = re.search(r"(\d{2})", a)
    if m:
        actor_nums.append(int(m.group(1)))

print('actor_nums_sorted_sample=', sorted(actor_nums)[:5], '... total=', len(actor_nums))

resf = WORK / 'results' / 'grouped_primary_split_results.json'
if resf.exists():
    data = json.loads(resf.read_text())
    cm = data.get('confusion_matrix')
    if cm is None:
        print('no confusion matrix in json')
    else:
        cm = np.array(cm)
        tp = cm.diagonal()
        support = cm.sum(axis=1)
        pred_counts = cm.sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.where(pred_counts>0, tp / pred_counts, 0.0)
            recall = np.where(support>0, tp / support, 0.0)
            f1 = np.where((precision+recall)>0, 2*precision*recall/(precision+recall), 0.0)
        print('per_class_f1=', [float(x) for x in f1.tolist()])
        print('macro_f1=', float(np.mean(f1)))
else:
    print('grouped_primary_split_results.json missing')
