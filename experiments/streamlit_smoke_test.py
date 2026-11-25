from pathlib import Path
import joblib
import importlib.util
import numpy as np

REV_PATH = Path(__file__).resolve().parents[0] / 'reviewer_experiments.py'
spec = importlib.util.spec_from_file_location('rev', str(REV_PATH))
rev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rev)

WORKSPACE = Path(__file__).resolve().parents[1]
MODEL_PKL = WORKSPACE / 'results' / 'stress_model.pkl'
SCALER_PKL = WORKSPACE / 'results' / 'scaler.pkl'

def run_smoke():
    if not MODEL_PKL.exists() or not SCALER_PKL.exists():
        print('Model or scaler pickle not found in results/. Run main.py first to produce them.')
        return
    clf = joblib.load(MODEL_PKL)
    scaler = joblib.load(SCALER_PKL)

    files = rev.get_ravdess_files(rev.DATASET_PATH)
    if not files:
        print('No audio files found')
        return
    sample = files[0]
    audio, sr = rev.load_audio_file(sample)
    feat = rev.build_segment_energy_features(audio, sr)
    feat = np.asarray(feat).reshape(1, -1)
    feat_s = scaler.transform(feat)
    pred = clf.predict(feat_s)
    print(f'Smoke test prediction for {sample.name}: {pred[0]}')
    Path(WORKSPACE / 'results' / 'streamlit_smoke_test.txt').write_text(f'{sample.name},{int(pred[0])}\n')

if __name__ == '__main__':
    run_smoke()
