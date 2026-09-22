"""
Anxiolytic-relevant acoustic feature extraction.

Every feature here is motivated by published evidence linking acoustic
properties to arousal / anxiety response. References (verified to exist):

- Bernardi, Porta & Sleight (2006), Heart, 92(4):445-450.
  "Cardiovascular, cerebrovascular, and respiratory changes induced by
  different types of music..." -> tempo drives autonomic arousal.
- Gomez & Danuser (2007), Emotion 7(2):377-387. "Relationships between
  musical structure and psychophysiological measures of emotion"
  -> tempo, accentuation, rhythmic articulation predict arousal.
- Rocha, Almeida & Perone (2022), IJERPH 19(2):994. "An Exploratory Study
  on the Acoustic Musical Properties to Decrease Self-Perceived Anxiety"
  -> BPM, loudness, mode, spectral characteristics of anxiolytic music.
- Sethares (1993), JASA 94(3):1218. Sensory dissonance curve used for the
  roughness estimate implemented below (Plomp-Levelt 1965 curve fit).
- Eyben et al. (2016), IEEE Trans. Affective Computing 7(2):190-202.
  GeMAPS/eGeMAPS standard acoustic parameter set (via openSMILE).

Output: one flat dict of named features per audio file.
"""
from __future__ import annotations
import numpy as np
import librosa

TARGET_SR = 22050

# ---------------------------------------------------------------- roughness
def _pl_dissonance(f1, f2, a1, a2):
    """Plomp-Levelt pairwise dissonance (Sethares 1993 parametrisation)."""
    b1, b2, dstar, s1, s2 = 3.5, 5.75, 0.24, 0.0207, 18.96
    fmin = np.minimum(f1, f2)
    s = dstar / (s1 * fmin + s2)
    fdif = np.abs(f2 - f1)
    return a1 * a2 * (np.exp(-b1 * s * fdif) - np.exp(-b2 * s * fdif))


def roughness_sethares(y: np.ndarray, sr: int, n_peaks: int = 12) -> float:
    """Mean sensory roughness over frames, from spectral peaks.

    Standard implementation of the Plomp-Levelt/Sethares estimate:
    pick the strongest spectral peaks per frame, sum pairwise dissonance.
    """
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=1024))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    frame_energy = S.sum(axis=0)
    energy_gate = 0.1 * np.median(frame_energy[frame_energy > 0]) \
        if np.any(frame_energy > 0) else 0.0
    vals = []
    for t in range(0, S.shape[1], 4):  # every ~186 ms
        mag = S[:, t]
        if mag.max() <= 1e-8 or frame_energy[t] <= energy_gate:
            continue  # roughness is only defined for audible content
        # local maxima
        idx = np.where((mag[1:-1] > mag[:-2]) & (mag[1:-1] > mag[2:]))[0] + 1
        if len(idx) < 2:
            continue
        idx = idx[np.argsort(mag[idx])[-n_peaks:]]
        f, a = freqs[idx], mag[idx] / mag[idx].max()
        tot = 0.0
        for i in range(len(f)):
            tot += float(np.sum(_pl_dissonance(f[i], f[i + 1:], a[i], a[i + 1:])))
        vals.append(tot)
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------- mode
_KRUMHANSL_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KRUMHANSL_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def mode_majorness(y: np.ndarray, sr: int) -> float:
    """Correlation-based major-vs-minor score in [-1, 1] (Krumhansl-Schmuckler).

    > 0 leans major, < 0 leans minor. Continuous, not a hard label.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    if chroma.max() <= 0:
        return 0.0
    best_maj = max(np.corrcoef(np.roll(_KRUMHANSL_MAJ, k), chroma)[0, 1]
                   for k in range(12))
    best_min = max(np.corrcoef(np.roll(_KRUMHANSL_MIN, k), chroma)[0, 1]
                   for k in range(12))
    return float(best_maj - best_min)


# ---------------------------------------------------------------- main
def extract_features(path_or_y, sr: int | None = None) -> dict:
    if isinstance(path_or_y, (str,)):
        y, sr = librosa.load(path_or_y, sr=TARGET_SR, mono=True)
    else:
        y = path_or_y
        assert sr is not None
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
    y, _ = librosa.effects.trim(y, top_db=40)
    if len(y) < sr:  # under 1 s of signal
        raise ValueError("audio too short after trim")

    out = {}

    # --- tempo & rhythm (Bernardi 2006; Gomez & Danuser 2007)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    out["tempo_bpm"] = float(np.atleast_1d(tempo)[0])
    env_n = onset_env / (onset_env.max() + 1e-8)
    onsets = librosa.onset.onset_detect(
        onset_envelope=env_n, sr=sr, delta=0.07, wait=2)
    out["onset_density"] = float(len(onsets) / (len(y) / sr))
    out["pulse_clarity"] = float(onset_env.std() / (onset_env.mean() + 1e-8))

    # --- energy / dynamics (loudness dimension of arousal)
    rms = librosa.feature.rms(y=y)[0]
    out["rms_mean"] = float(rms.mean())
    out["rms_std"] = float(rms.std())
    out["rms_db_range"] = float(
        np.percentile(librosa.amplitude_to_db(rms + 1e-10), 95)
        - np.percentile(librosa.amplitude_to_db(rms + 1e-10), 5))
    out["attack_slope_mean"] = float(np.mean(np.maximum(np.diff(rms), 0)))

    # --- spectral shape / brightness
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    out["spectral_centroid_mean"] = float(cent.mean())
    out["spectral_centroid_std"] = float(cent.std())
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    out["spectral_rolloff_mean"] = float(rolloff.mean())
    flat = librosa.feature.spectral_flatness(y=y)[0]
    out["spectral_flatness_mean"] = float(flat.mean())

    # --- spectral flux (event/transient density in spectrum)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    Sn = S / (S.sum(axis=0, keepdims=True) + 1e-10)
    flux = np.sqrt(np.sum(np.diff(Sn, axis=1) ** 2, axis=0))
    out["spectral_flux_mean"] = float(flux.mean())
    out["spectral_flux_std"] = float(flux.std())

    # --- roughness (Plomp-Levelt/Sethares)
    out["roughness"] = roughness_sethares(y, sr)

    # --- tonality / mode (Rocha 2022: mode relates to perceived anxiety)
    out["mode_majorness"] = mode_majorness(y, sr)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    out["zcr_mean"] = float(zcr.mean())

    # --- register
    try:
        f0 = librosa.yin(y, fmin=55, fmax=1760, sr=sr)
        f0 = f0[np.isfinite(f0)]
        out["f0_median"] = float(np.median(f0)) if len(f0) else 0.0
    except Exception:
        out["f0_median"] = 0.0

    return out


FEATURE_NAMES = None  # filled on first call by callers if needed


def extract_egemaps(path: str) -> dict:
    """eGeMAPSv02 functionals via openSMILE (88 features), prefixed 'eg_'."""
    import opensmile
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    df = smile.process_file(path)
    return {f"eg_{c}": float(df[c].iloc[0]) for c in df.columns}
