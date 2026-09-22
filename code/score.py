"""Score audio with the trained verifier: predicted arousal & valence (1-9).

Used by: the generation audit, the ISO-agent loop, and best-of-N.
Lower predicted arousal = calmer. This is a proxy validated on DEAM
listener annotations — not a clinical anxiety measure.
"""
import os, sys, glob, json, argparse
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features, extract_egemaps

_CACHE = {}


def load_verifier(vdir, variant="deploy"):
    key = (os.path.abspath(vdir), variant)
    if key not in _CACHE:
        _CACHE[key] = {
            t: joblib.load(os.path.join(vdir, f"verifier_{t}_{variant}.joblib"))
            for t in ["arousal_mean", "valence_mean"]}
    return _CACHE[key]


def score_file(path, vdir, use_egemaps=True, variant="deploy"):
    v = load_verifier(vdir, variant)
    feats = extract_features(path)
    if use_egemaps:
        feats.update(extract_egemaps(path))
    out = {}
    for tgt, pack in v.items():
        x = np.array([[feats.get(c, np.nan) for c in pack["cols"]]])
        out[tgt.replace("_mean", "_pred")] = float(pack["model"].predict(x)[0])
    return out


def score_array(y, sr, vdir, variant="deploy"):
    """Score a raw waveform by round-tripping through a temp wav
    (openSMILE needs a file)."""
    import soundfile as sf, tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, y, sr)
        try:
            return score_file(f.name, vdir, variant=variant)
        finally:
            os.unlink(f.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--vdir", default="/home/claude/amg/results/verifier")
    args = ap.parse_args()
    for pat in args.paths:
        for p in sorted(glob.glob(pat)):
            print(p, json.dumps(score_file(p, args.vdir)))
