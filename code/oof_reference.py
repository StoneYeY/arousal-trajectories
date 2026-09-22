"""Out-of-fold reference analysis (review item 2).

The deployed proxy is trained on all 1,795 DEAM excerpts, so the human
reference groups it scores are in-sample while generations are
out-of-sample. Here every DEAM excerpt gets an out-of-fold (5-fold)
prediction, and the 840 SA3 generations are scored by the SAME five fold
models (averaged), so both sides of every gap come from models that never
saw the reference excerpts. Same recipe as the deploy model (HGB,
max_iter=300, seed 42, 105 features). Also repeated for the 20-prompt
decoding-sweep subset and MusicGen-small. Output: results/oof_reference.json
"""
import glob, json, os
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
pack = joblib.load(os.path.join(R, "verifier", "verifier_arousal_mean_deploy.joblib"))
COLS = pack["cols"]
deam = pd.read_csv(os.path.join(R, "deam_features.csv")).dropna(subset=["arousal_mean"]).reset_index(drop=True)
X = deam[COLS].values; y = deam["arousal_mean"].values


def load_feats(pattern):
    rows, seen = [], set()
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_error" in r or r["clip_id"] in seen:
                continue
            seen.add(r["clip_id"]); rows.append(r)
    ids = [r["clip_id"] for r in rows]
    G = np.array([[r["feat_raw"].get(c, np.nan) for c in COLS] for r in rows])
    meta = pd.DataFrame([{"clip_id": r["clip_id"], "group": r.get("group"), "category": r.get("category"),
                          "prompt": r["prompt"]} for r in rows])
    return meta, G


sets = {"sa3": load_feats(os.path.join(R, "sa3_features*.jsonl")),
        "musicgen": load_feats(os.path.join(R, "step3", "audit_musicgen_features*.jsonl"))}
for cfg in ["sweep_s8_cfg1.0", "sweep_s25_cfg1.0", "sweep_s25_cfg3.0", "sweep_s50_cfg3.0", "sweep_s50_cfg6.0"]:
    sets[cfg] = load_feats(os.path.join(R, "step3", cfg + "_features*.jsonl"))
# LUFS-normalised features of the SA3 audit (second pass) scored by the same fold models
def load_ln(pattern):
    rows, seen = [], set()
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_error" in r or r["clip_id"] in seen or not r.get("feat_ln"):
                continue
            seen.add(r["clip_id"]); rows.append(r)
    meta = pd.DataFrame([{"clip_id": r["clip_id"], "group": r.get("group"), "category": r.get("category"), "prompt": r["prompt"]} for r in rows])
    return meta, np.array([[r["feat_ln"].get(c, np.nan) for c in COLS] for r in rows])
sets["sa3_lufs"] = load_ln(os.path.join(R, "sa3_features*.jsonl"))

oof = np.full(len(deam), np.nan)
gen_pred = {k: np.zeros((5, len(m))) for k, (m, _) in sets.items()}
kf = KFold(5, shuffle=True, random_state=42)
for i, (tr, te) in enumerate(kf.split(X)):
    m = HistGradientBoostingRegressor(random_state=42, max_iter=300).fit(X[tr], y[tr])
    oof[te] = m.predict(X[te])
    for k, (_, G) in sets.items():
        gen_pred[k][i] = m.predict(G)
deam["oof"] = oof
deam["insample"] = pack["model"].predict(X)
# independent estimator (ridge, 17 hand features): in-sample vs out-of-fold, for the table caption
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
ipack = joblib.load(os.path.join(R, "verifier", "verifier_arousal_mean_independent.joblib"))
XI = deam[ipack["cols"]].values; oof_i = np.full(len(deam), np.nan)
for tr, te in KFold(5, shuffle=True, random_state=42).split(XI):
    oof_i[te] = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]).fit(XI[tr], y[tr]).predict(XI[te])
deam["oof_indep"] = oof_i; deam["insample_indep"] = ipack["model"].predict(XI)
q1, q3 = deam.arousal_mean.quantile([0.25, 0.75]); vmed = deam.valence_mean.median()
refs = {"human_calm": deam[(deam.arousal_mean <= q1) & (deam.valence_mean >= vmed)],
        "human_tense": deam[deam.arousal_mean >= q3]}


def boot(x, clusters=None, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    if clusters is None:
        ms = [rng.choice(x, len(x)).mean() for _ in range(n)]
    else:
        c = np.asarray(clusters); u = np.unique(c); groups = [x[c == g] for g in u]
        ms = [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(n)]
    return {"mean": float(x.mean()), "lo": float(np.percentile(ms, 2.5)), "hi": float(np.percentile(ms, 97.5)), "n": int(len(x))}


out = {"oof_r_all_deam": float(np.corrcoef(oof, y)[0, 1]),
       "insample_r_all_deam": float(np.corrcoef(deam.insample, y)[0, 1]),
       "refs": {}}
for k, d in refs.items():
    out["refs"][k] = {"n": int(len(d)), "true": float(d.arousal_mean.mean()),
                      "insample": boot(d.insample), "oof": boot(d.oof),
                      "indep_insample": float(d.insample_indep.mean()), "indep_oof": float(d.oof_indep.mean())}
# four reference definitions under OOF (mirrors reference_sensitivity.json)
d1, d9 = deam.arousal_mean.quantile([0.10, 0.90])
defs = {"A_quartiles_only": (deam[deam.arousal_mean <= q1], deam[deam.arousal_mean >= q3]),
        "B_quartile_valence": (refs["human_calm"], refs["human_tense"]),
        "C_deciles": (deam[deam.arousal_mean <= d1], deam[deam.arousal_mean >= d9]),
        "D_quartile_valence_both": (refs["human_calm"], deam[(deam.arousal_mean >= q3) & (deam.valence_mean <= vmed)])}
out["ref_definitions_oof"] = {k: {"n_calm": int(len(c)), "n_tense": int(len(t)), "calm_oof": float(c.oof.mean()), "tense_oof": float(t.oof.mean())} for k, (c, t) in defs.items()}
out["deam_oof_percentiles"] = {"frac_below_3.39": float((deam.oof <= 3.39).mean()), "frac_above_5.18": float((deam.oof >= 5.18).mean()),
                               "oof_quantiles": {q: float(deam.oof.quantile(q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]}}

for k, (meta, _) in sets.items():
    P = gen_pred[k].mean(0)
    meta = meta.assign(pred_oofmodels=P)
    res = {}
    for grp, sub in [("anxiolytic", meta[meta.group == "anxiolytic"]),
                     ("high_arousal", meta[meta.category == "high_arousal"]),
                     ("neutral", meta[meta.category == "neutral"])]:
        if len(sub):
            res[grp] = boot(sub.pred_oofmodels, clusters=sub.prompt)
    # gap bootstrap: resample prompts (gen) and songs (ref) independently
    rng = np.random.default_rng(1)
    def gap_ci(sub, ref, n=2000):
        c = sub.prompt.values; u = np.unique(c); groups = [sub.pred_oofmodels.values[c == g] for g in u]
        r = ref.oof.values; ds = []
        for _ in range(n):
            g = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]).mean()
            ds.append(g - rng.choice(r, len(r)).mean())
        return {"mean": float(sub.pred_oofmodels.mean() - r.mean()), "lo": float(np.percentile(ds, 2.5)), "hi": float(np.percentile(ds, 97.5))}
    res["calm_gap_oof"] = gap_ci(meta[meta.group == "anxiolytic"], refs["human_calm"])
    res["high_gap_oof"] = gap_ci(meta[meta.category == "high_arousal"], refs["human_tense"])
    if k == "sa3":
        res["gaps_by_definition"] = {}
        for dk, (c, t) in defs.items():
            res["gaps_by_definition"][dk] = {"calm_gap": float(meta[meta.group == "anxiolytic"].pred_oofmodels.mean() - c.oof.mean()),
                                             "high_gap": float(meta[meta.category == "high_arousal"].pred_oofmodels.mean() - t.oof.mean())}
        res["frac_high_clips_above_tense_oof"] = float((meta[meta.category == "high_arousal"].pred_oofmodels >= refs["human_tense"].oof.mean()).mean())
        res["per_category"] = {c_: boot(sub.pred_oofmodels, clusters=sub.prompt) for c_, sub in meta.groupby("category")}
        meta.to_csv(os.path.join(R, "audit_clips_foldmodels.csv"), index=False)
    if k in ("musicgen", "sa3_lufs"):
        res["per_category"] = {c_: boot(sub.pred_oofmodels, clusters=sub.prompt) for c_, sub in meta.groupby("category")}
    if k.startswith("sweep"):
        res["frac_high_clips_above_tense_oof"] = float((meta[meta.category == "high_arousal"].pred_oofmodels >= refs["human_tense"].oof.mean()).mean())
    out[k] = res

# also: sweep subset (20 prompts, 4 seeds, main recipe) scored by the fold models
json.dump(out, open(os.path.join(R, "oof_reference.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
