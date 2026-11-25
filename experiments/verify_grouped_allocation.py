from pathlib import Path
import re
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

WORK = Path(__file__).resolve().parents[1]
DATA = WORK / 'ravdess-emotional-speech-audio'

files = []
for actor_dir in sorted(DATA.iterdir()):
    if not actor_dir.is_dir():
        continue
    for f in sorted(actor_dir.iterdir()):
        if f.suffix.lower() == '.wav':
            files.append(str(f))

def actor_from_path(p):
    m = re.search(r'Actor_(\d{2})', p)
    if m:
        return int(m.group(1))
    parts = Path(p).name.split('-')
    if len(parts) >= 7:
        try:
            return int(parts[6])
        except:
            return None
    return None

groups = [actor_from_path(p) for p in files]
uniq_actors = sorted(set(groups))
print('unique_actors_count=', len(uniq_actors))

# grouped 80:20
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(np.zeros(len(files)), np.zeros(len(files)), groups))
tr_actors = sorted(set([groups[i] for i in tr_idx]))
te_actors = sorted(set([groups[i] for i in te_idx]))
print('train_actors_count=', len(tr_actors), 'test_actors_count=', len(te_actors))
print('intersection_count=', len(set(tr_actors).intersection(set(te_actors))))

# grouped 5-fold using GroupKFold: ensure no actor appears in both train and test per fold
gk = GroupKFold(n_splits=5)
fold_ok = True
for fold, (train_idx, test_idx) in enumerate(gk.split(np.zeros(len(files)), np.zeros(len(files)), groups), start=1):
    tr_a = set([groups[i] for i in train_idx])
    te_a = set([groups[i] for i in test_idx])
    inter = tr_a.intersection(te_a)
    print(f'fold {fold}: train_actors={len(tr_a)}, test_actors={len(te_a)}, intersection={len(inter)}')
    if len(inter) != 0:
        fold_ok = False

print('grouped_cv_speaker_independent=', fold_ok)
