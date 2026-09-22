"""Stage 2 of the audit: statistics from cached features (audit_extract).

Inputs:
  --features   results/sa3_features.jsonl (+ .shard*.jsonl merged)
  --deam       results/deam_features.csv   (human reference groups)
  --vdir       results/verifier            (deploy + independent models)
  --clap       results/clap_alignment.jsonl (optional)

Outputs results/audit_table.json with:
  - per category x {raw, LUFS-normalised} x {deploy, independent}:
    predicted arousal/valence, prompt-clustered bootstrap 95% CI
  - human_calm / human_tense reference groups (predictions + true labels)
  - mixed-effects model arousal ~ C(category) + (1|prompt)
  - CLAP cross-analysis: semantic alignment vs acoustic arousal
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KEYS = ["tempo_bpm", "onset_density", "spectral_flux_mean", "roughness",
        "rms_mean", "spectral_centroid_mean", "mode_majorness"]


def boot_ci(x, clusters=None, n=2000, seed=0):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    x = x[m]
    if len(x) == 0:
        return None
    rng = np.random.default_rng(seed)
    if clusters is not None:
        c = np.asarray(clusters)[m]
        groups = [x[c == u] for u in np.unique(c)]
        means = [np.concatenate([groups[i] for i in
                 rng.integers(0, len(groups), len(groups))]).mean()
                 for _ in range(n)]
        return {"mean": float(x.mean()), "lo": float(np.percentile(means, 2.5)),
                "hi": float(np.percentile(means, 97.5)), "n": int(len(x)),
                "n_clusters": int(len(groups))}
    means = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return {"mean": float(x.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n": int(len(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--deam", required=True)
    ap.add_argument("--vdir", required=True)
    ap.add_argument("--clap", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for shard in sorted(glob.glob(args.features.replace(".jsonl", "*.jsonl"))):
        for line in open(shard):
            try:
                r = json.loads(line)
                if "_error" not in r:
                    rows.append(r)
            except json.JSONDecodeError:
                pass
    seen = set()
    rows = [r for r in rows if not (r["clip_id"] in seen or seen.add(r["clip_id"]))]
    print(f"clips with features: {len(rows)}")

    packs = {}
    for variant in ["deploy", "independent"]:
        packs[variant] = {t: joblib.load(os.path.join(
            args.vdir, f"verifier_{t}_{variant}.joblib"))
            for t in ["arousal_mean", "valence_mean"]}

    def predict(feat, variant, tgt):
        p = packs[variant][tgt]
        X = np.array([[feat.get(c, np.nan) for c in p["cols"]]])
        return float(p["model"].predict(X)[0])

    recs = []
    for r in rows:
        rec = {"clip_id": r["clip_id"], "group": r["group"],
               "category": r["category"], "prompt": r["prompt"],
               "lufs_in": r.get("lufs_in")}
        for fkey, suffix in [("feat_raw", ""), ("feat_ln", "_ln")]:
            rec["arousal" + suffix] = predict(r[fkey], "deploy", "arousal_mean")
            rec["valence" + suffix] = predict(r[fkey], "deploy", "valence_mean")
            rec["arousal_indep" + suffix] = predict(r[fkey], "independent",
                                                    "arousal_mean")
        for k in KEYS:
            rec[k] = r["feat_raw"].get(k)
        recs.append(rec)
    gen = pd.DataFrame(recs)

    if args.clap and os.path.exists(args.clap):
        clap = pd.DataFrame([json.loads(l) for l in open(args.clap)])
        gen = gen.merge(clap[["clip_id", "clap_similarity"]],
                        on="clip_id", how="left")

    deam = pd.read_csv(args.deam)
    q1, q3 = deam["arousal_mean"].quantile([0.25, 0.75])
    vmed = deam["valence_mean"].median()
    refs = {"human_calm": deam[(deam.arousal_mean <= q1) &
                               (deam.valence_mean >= vmed)],
            "human_tense": deam[deam.arousal_mean >= q3]}

    out = {"groups": {}, "n_clips": int(len(gen))}
    for (grp, cat), df in gen.groupby(["group", "category"]):
        g = {"n": int(len(df))}
        for col, label in [("arousal", "arousal_raw"),
                           ("arousal_ln", "arousal_LUFSnorm"),
                           ("arousal_indep", "arousal_independent_raw"),
                           ("arousal_indep_ln", "arousal_independent_LUFSnorm"),
                           ("valence", "valence_raw"),
                           ("clap_similarity", "clap")]:
            if col in df.columns:
                g[label] = boot_ci(df[col], clusters=df["prompt"])
        g["lufs_in_mean"] = float(df["lufs_in"].mean())
        for k in KEYS:
            g[k] = boot_ci(df[k], clusters=df["prompt"])
        out["groups"][f"gen/{grp}/{cat}"] = g
    anx = gen[gen.group == "anxiolytic"]
    out["groups"]["gen/anxiolytic/ALL"] = {
        "n": int(len(anx)),
        "arousal_raw": boot_ci(anx["arousal"], clusters=anx["prompt"]),
        "arousal_LUFSnorm": boot_ci(anx["arousal_ln"], clusters=anx["prompt"]),
        "arousal_independent_raw": boot_ci(anx["arousal_indep"],
                                           clusters=anx["prompt"]),
        "valence_raw": boot_ci(anx["valence"], clusters=anx["prompt"])}

    for name, df in refs.items():
        g = {"n": int(len(df)),
             "arousal_true": boot_ci(df["arousal_mean"]),
             "valence_true": boot_ci(df["valence_mean"])}
        for variant, label in [("deploy", "arousal_pred"),
                               ("independent", "arousal_pred_independent")]:
            p = packs[variant]["arousal_mean"]
            X = df[[c for c in p["cols"]]].values
            g[label] = boot_ci(p["model"].predict(X))
        for k in KEYS:
            g[k] = boot_ci(df[k])
        out["groups"][f"ref/{name}"] = g

    # mixed-effects: arousal ~ category with prompt random intercept
    try:
        import statsmodels.formula.api as smf
        m = smf.mixedlm("arousal ~ C(category)", gen,
                        groups=gen["prompt"]).fit(reml=True)
        out["mixed_effects"] = {
            "formula": "arousal ~ C(category) + (1|prompt)",
            "params": {k: float(v) for k, v in m.params.items()},
            "pvalues": {k: float(v) for k, v in m.pvalues.items()},
            "prompt_var": float(m.cov_re.iloc[0, 0]),
            "resid_var": float(m.scale)}
    except Exception as e:
        out["mixed_effects"] = {"error": str(e)[:300]}

    # CLAP cross-analysis on anxiolytic clips
    if "clap_similarity" in gen.columns:
        a = anx.dropna(subset=["clap_similarity"])
        hc = out["groups"]["ref/human_calm"]["arousal_pred"]["mean"]
        med_clap = float(a["clap_similarity"].median())
        out["clap_cross"] = {
            "anxiolytic_clap_mean": float(a["clap_similarity"].mean()),
            "r_clap_vs_arousal_anxiolytic":
                float(a["clap_similarity"].corr(a["arousal"])),
            "median_clap_split": med_clap,
            "high_clap_frac_above_human_calm_arousal":
                float((a[a.clap_similarity >= med_clap]["arousal"] > hc).mean()),
            "note": "high-CLAP & above-human-calm-arousal = followed the "
                    "prompt semantically yet acoustically not calm"}

    gen.to_csv(args.out.replace(".json", "_clips.csv"), index=False)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)
    for k in sorted(out["groups"]):
        g = out["groups"][k]
        a = g.get("arousal_raw") or g.get("arousal_pred")
        ln = g.get("arousal_LUFSnorm")
        s = f"{k:28s} n={g['n']:4d}"
        if a: s += f" arousal={a['mean']:.2f} [{a['lo']:.2f},{a['hi']:.2f}]"
        if ln: s += f" | LUFSnorm={ln['mean']:.2f}"
        print(s)


if __name__ == "__main__":
    main()
