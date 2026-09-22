"""Verify (or refute) the octave-error hypothesis for the weak tempo-
arousal correlation, instead of asserting it.

For a random subsample of DEAM songs:
- compute librosa global tempo AND tempogram-based dominant tempo;
- fold both into a perceptual range [60, 180) by octave shifts;
- report correlations of raw vs folded tempo with arousal, and the
  fraction of songs whose folded tempo differs from raw (octave errors).

Honest wording rule: if folded tempo still shows little independent
association after accounting for onset density, the paper says exactly
that — not "librosa failed".

Usage: python3 tempo_check.py --features results/deam_features.csv \
    --wav_dir data/deam/wav22 --n 150 --out results/tempo_check.json
"""
import argparse, json, os
import numpy as np
import pandas as pd
import librosa


def fold_tempo(t, lo=60.0, hi=180.0):
    if t <= 0:
        return t
    while t < lo:
        t *= 2
    while t >= hi:
        t /= 2
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    rng = np.random.default_rng(args.seed)
    sub = df.sample(min(args.n, len(df)), random_state=args.seed)

    rows = []
    for _, r in sub.iterrows():
        wav = os.path.join(args.wav_dir, f"{int(r.song_id)}.wav")
        if not os.path.exists(wav):
            continue
        y, sr = librosa.load(wav, sr=22050, mono=True)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        # tempogram-based dominant tempo
        tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        tempo_axis = librosa.tempo_frequencies(tg.shape[0], sr=sr)
        strength = tg.mean(axis=1)
        valid = np.isfinite(tempo_axis) & (tempo_axis > 30) & (tempo_axis < 300)
        t_tg = float(tempo_axis[valid][np.argmax(strength[valid])])
        rows.append({"song_id": int(r.song_id),
                     "arousal": float(r.arousal_mean),
                     "onset_density": float(r.onset_density),
                     "t_raw": float(r.tempo_bpm),
                     "t_raw_folded": fold_tempo(float(r.tempo_bpm)),
                     "t_tempogram": t_tg,
                     "t_tempogram_folded": fold_tempo(t_tg)})
    d = pd.DataFrame(rows)

    def pcorr(a, b):
        return float(np.corrcoef(d[a], d[b])[0, 1])

    # partial correlation of folded tempo with arousal given onset density
    def partial(a, b, c):
        ra = d[a] - np.polyval(np.polyfit(d[c], d[a], 1), d[c])
        rb = d[b] - np.polyval(np.polyfit(d[c], d[b], 1), d[c])
        return float(np.corrcoef(ra, rb)[0, 1])

    out = {"n": len(d),
           "octave_disagreement_raw_vs_folded":
               float((abs(d.t_raw - d.t_raw_folded) > 1).mean()),
           "r_arousal": {
               "t_raw": pcorr("t_raw", "arousal"),
               "t_raw_folded": pcorr("t_raw_folded", "arousal"),
               "t_tempogram": pcorr("t_tempogram", "arousal"),
               "t_tempogram_folded": pcorr("t_tempogram_folded", "arousal"),
               "onset_density": pcorr("onset_density", "arousal")},
           "partial_r_arousal_given_onset_density": {
               "t_tempogram_folded": partial("t_tempogram_folded",
                                             "arousal", "onset_density")}}
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
