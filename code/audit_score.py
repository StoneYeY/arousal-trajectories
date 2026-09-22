"""Score audit clips + DEAM human reference groups; produce the audit table.

Reference groups (human-composed, from DEAM annotations, never seen by
the verifier's held-out logic is irrelevant here — these use *ground
truth* labels, not verifier output):
  human_calm  : DEAM songs in the lowest arousal quartile AND valence >= median
  human_tense : DEAM songs in the highest arousal quartile

Audit quantities per group/category (all with 95% bootstrap CIs):
  - verifier predicted arousal / valence of generated clips
  - same for human reference groups (predicted, for apples-to-apples,
    AND ground-truth labels reported alongside)
  - key acoustic features (tempo, flux, onset density, roughness, RMS)

Usage:
  python3 audit_score.py --audit_dir amg/results/audit_audio \
      --deam_features amg/results/deam_features.csv \
      --vdir amg/results/verifier --out amg/results/audit_table.json
"""
import argparse, json, os, sys, tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import score_file, score_array
from features import extract_features


def score_loudness_normalized(path, vdir, target_lufs=-23.0):
    """Re-score after normalising integrated loudness to target LUFS.

    Sensitivity analysis for the mastering-loudness confound: if the
    calm-prompt arousal gap shrinks under normalisation, part of the gap
    is production loudness, not compositional structure — reported as a
    diagnostic either way.
    """
    import librosa, pyloudnorm
    y, sr = librosa.load(path, sr=22050, mono=True)
    meter = pyloudnorm.Meter(sr)
    lufs = meter.integrated_loudness(y)
    if np.isfinite(lufs):
        y = pyloudnorm.normalize.loudness(y, lufs, target_lufs)
        peak = np.max(np.abs(y))
        if peak > 0.99:          # avoid clipping after gain
            y = y * (0.99 / peak)
    return score_array(y, sr, vdir)

KEYS = ["tempo_bpm", "onset_density", "spectral_flux_mean", "roughness",
        "rms_mean", "spectral_centroid_mean", "mode_majorness"]


def boot_ci(x, n=2000, seed=0, clusters=None):
    """Bootstrap CI of the mean. If `clusters` given (e.g. prompt ids),
    resample CLUSTERS with replacement — clips from the same prompt are
    not independent, so naive bootstrap would understate uncertainty."""
    x = np.asarray(x, float)
    m = np.isfinite(x)
    x = x[m]
    if len(x) == 0:
        return None
    rng = np.random.default_rng(seed)
    if clusters is not None:
        c = np.asarray(clusters)[m]
        uniq = np.unique(c)
        groups = [x[c == u] for u in uniq]
        means = []
        for _ in range(n):
            idx = rng.integers(0, len(groups), len(groups))
            means.append(np.concatenate([groups[i] for i in idx]).mean())
        return {"mean": float(x.mean()),
                "lo": float(np.percentile(means, 2.5)),
                "hi": float(np.percentile(means, 97.5)),
                "n": int(len(x)), "n_clusters": int(len(uniq))}
    means = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return {"mean": float(x.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n": int(len(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit_dir", required=True)
    ap.add_argument("--deam_features", required=True)
    ap.add_argument("--vdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target_lufs", type=float, default=-23.0)
    args = ap.parse_args()

    rows = []
    man = os.path.join(args.audit_dir, "manifest.jsonl")
    for line in open(man):
        rec = json.loads(line)
        # audio may have been transcoded (e.g. .flac at 22.05k mono);
        # resolve by clip_id in audit_dir rather than trusting server paths
        path = None
        for ext in (".flac", ".wav"):
            cand = os.path.join(args.audit_dir, rec["clip_id"] + ext)
            if os.path.exists(cand):
                path = cand; break
        if path is None:
            continue
        try:
            sc = score_file(path, args.vdir)
            fe = extract_features(path)
            # loudness-normalized sensitivity pass: re-score at a common
            # LUFS so production-level loudness can't drive the result
            sc_ln = score_loudness_normalized(path, args.vdir,
                                              target_lufs=args.target_lufs)
        except Exception as e:
            print("skip", rec["clip_id"], e); continue
        rows.append({**rec, **sc,
                     **{k + "_ln": v for k, v in sc_ln.items()},
                     **{k: fe[k] for k in KEYS}})
    gen = pd.DataFrame(rows)
    print(f"scored {len(gen)} generated clips")

    deam = pd.read_csv(args.deam_features)
    q1, q3 = deam["arousal_mean"].quantile([0.25, 0.75])
    vmed = deam["valence_mean"].median()
    human_calm = deam[(deam["arousal_mean"] <= q1) & (deam["valence_mean"] >= vmed)]
    human_tense = deam[deam["arousal_mean"] >= q3]

    out = {"groups": {}}

    def add_group(name, df, pred_a=None, pred_v=None, true_a=None,
                  true_v=None, clusters=None):
        g = {"n": int(len(df))}
        if pred_a is not None:
            g["arousal_pred"] = boot_ci(pred_a, clusters=clusters)
            g["valence_pred"] = boot_ci(pred_v, clusters=clusters)
        if true_a is not None:
            g["arousal_true"] = boot_ci(true_a)
            g["valence_true"] = boot_ci(true_v)
        for k in KEYS:
            if k in df.columns:
                g[k] = boot_ci(df[k], clusters=clusters)
        out["groups"][name] = g

    for (grp, cat), df in gen.groupby(["group", "category"]):
        add_group(f"gen/{grp}/{cat}", df,
                  df["arousal_pred"], df["valence_pred"],
                  clusters=df["prompt"])
        out["groups"][f"gen/{grp}/{cat}"]["arousal_pred_LUFSnorm"] = \
            boot_ci(df["arousal_pred_ln"], clusters=df["prompt"])
    anx = gen[gen.group == "anxiolytic"]
    add_group("gen/anxiolytic/ALL", anx, anx["arousal_pred"],
              anx["valence_pred"], clusters=anx["prompt"])
    out["groups"]["gen/anxiolytic/ALL"]["arousal_pred_LUFSnorm"] = \
        boot_ci(anx["arousal_pred_ln"], clusters=anx["prompt"])

    # mixed-effects model: arousal_pred ~ category, prompt random intercept
    try:
        import statsmodels.formula.api as smf
        g = gen.copy()
        g["category"] = pd.Categorical(g["category"])
        m = smf.mixedlm("arousal_pred ~ C(category)", g,
                        groups=g["prompt"]).fit(reml=True)
        out["mixed_effects"] = {
            "formula": "arousal_pred ~ C(category) + (1|prompt)",
            "params": {k: float(v) for k, v in m.params.items()},
            "pvalues": {k: float(v) for k, v in m.pvalues.items()},
            "prompt_var": float(m.cov_re.iloc[0, 0]),
            "resid_var": float(m.scale)}
    except Exception as e:
        out["mixed_effects"] = {"error": str(e)[:300]}

    # human refs: verifier predictions come from the deployed model applied
    # to their features - do it via stored features (fair: same features)
    import joblib
    for name, df in [("human_calm", human_calm), ("human_tense", human_tense)]:
        preds = {}
        for tgt in ["arousal_mean", "valence_mean"]:
            pack = joblib.load(os.path.join(
                args.vdir, f"verifier_{tgt}_deploy.joblib"))
            X = df[[c for c in pack["cols"]]].values
            preds[tgt] = pack["model"].predict(X)
        add_group(f"ref/{name}", df, preds["arousal_mean"],
                  preds["valence_mean"], df["arousal_mean"], df["valence_mean"])

    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)
    for k, v in out["groups"].items():
        a = v.get("arousal_pred")
        if a:
            print(f"{k:28s} arousal_pred={a['mean']:.2f} [{a['lo']:.2f},{a['hi']:.2f}] n={v['n']}")


if __name__ == "__main__":
    main()
