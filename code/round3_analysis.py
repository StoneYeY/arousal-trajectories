"""Round-3 review analyses (all from existing data). Output results/round3.json.

A. descriptor-level audit: onset density, spectral flux, attack slope,
   loudness (RMS + eGeMAPS loudness) for anxiolytic / high-arousal
   generations vs the human calm / tense references; fraction of
   generated clips outside the DEAM 1st-99th percentile range on >=1 of
   the four descriptors (extrapolation check).
B. calm-end perceptual anchor: listener means of generated items the
   proxy scores <= 3.5 vs DEAM items labelled <= 4.6 (32-item calibration).
C. item-level tracking error vs ratings: pooled and within scheduled
   planners (the listening trajectories carry their own E_track).
D. excess tracking error relative to the reachable optimum: schedule
   clipped at the synthesiser's reachable proxy maximum.
E. proxy stability under short windows: DEAM subset, 5-s and 12-s
   windows vs the 45-s excerpt (deployed proxy).
F. loudness of the listening stimuli by condition (were they matched?).
G. per-item rater counts.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results"); LS = os.path.join(ROOT, "listening_study")
out = {}

# ---------------------------------------------------------------- A
deam = pd.read_csv(os.path.join(R, "deam_features.csv"))
oof = json.load(open(os.path.join(R, "oof_reference.json")))
q1, q3 = deam.arousal_mean.quantile([0.25, 0.75]); vmed = deam.valence_mean.median()
calm = deam[(deam.arousal_mean <= q1) & (deam.valence_mean >= vmed)]
tense = deam[deam.arousal_mean >= q3]
rows = []
for f in sorted(glob.glob(os.path.join(R, "sa3_features*.jsonl"))):
    for line in open(f):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "_error" in r:
            continue
        d = dict(r["feat_raw"]); d.update(clip_id=r["clip_id"], group=r["group"], category=r["category"],
                                         prompt=r["prompt"], lufs_in=r["lufs_in"])
        rows.append(d)
gen = pd.DataFrame(rows)
DESC = {"onset_density": "onset density (/s)", "spectral_flux_mean": "spectral flux",
        "attack_slope_mean": "attack slope", "eg_loudness_sma3_amean": "eGeMAPS loudness",
        "rms_mean": "RMS"}
A = {"descriptors": {}, "n_gen": int(len(gen))}
for k, name in DESC.items():
    A["descriptors"][k] = {"label": name,
        "anxiolytic": float(gen[gen.group == "anxiolytic"][k].median()),
        "high_arousal": float(gen[gen.category == "high_arousal"][k].median()),
        "human_calm": float(calm[k].median()), "human_tense": float(tense[k].median()),
        "deam_all": float(deam[k].median()),
        "deam_p1": float(deam[k].quantile(0.01)), "deam_p99": float(deam[k].quantile(0.99)),
        "anx_frac_below_deam_p1": float((gen[gen.group == "anxiolytic"][k] < deam[k].quantile(0.01)).mean()),
        "anx_frac_outside_p1_p99": float(((gen[gen.group == "anxiolytic"][k] < deam[k].quantile(0.01)) |
                                          (gen[gen.group == "anxiolytic"][k] > deam[k].quantile(0.99))).mean()),
        "mwu_p_anx_vs_calm": float(stats.mannwhitneyu(gen[gen.group == "anxiolytic"][k], calm[k]).pvalue)}
four = ["onset_density", "spectral_flux_mean", "attack_slope_mean", "eg_loudness_sma3_amean"]
def outside(df):
    m = np.zeros(len(df), bool)
    for k in four:
        lo, hi = deam[k].quantile(0.01), deam[k].quantile(0.99)
        m |= (df[k].values < lo) | (df[k].values > hi)
    return m
A["frac_outside_any_of_four"] = {"all_gen": float(outside(gen).mean()),
                                 "anxiolytic": float(outside(gen[gen.group == "anxiolytic"]).mean()),
                                 "high_arousal": float(outside(gen[gen.category == "high_arousal"]).mean())}
# all 17 hand features
hand = [c for c in deam.columns if c not in ("song_id", "valence_mean", "valence_std", "arousal_mean", "arousal_std") and not c.startswith("eg_")]
def outside_any(df, cols):
    m = np.zeros(len(df), bool)
    for k in cols:
        lo, hi = deam[k].quantile(0.01), deam[k].quantile(0.99)
        m |= (df[k].values < lo) | (df[k].values > hi)
    return m
A["frac_outside_any_of_17"] = {"all_gen": float(outside_any(gen, hand).mean()),
                               "anxiolytic": float(outside_any(gen[gen.group == "anxiolytic"], hand).mean())}
# how does the calm reference itself compare to DEAM p1 (sanity)
A["calm_ref_frac_outside_four"] = float(outside(calm).mean())
out["A_descriptors"] = A

# ---------------------------------------------------------------- B
LSUM = json.load(open(os.path.join(R, "listening_summary.json")))
D = pd.DataFrame(LSUM["D_proxy_validity"]["per_item"])
g_calm = D[(D.domain == "generated") & (D.proxy <= 3.5)]
d_calm = D[(D.domain == "real") & (D["true"] <= 4.6)]
c_calm = D[(D.domain == "midi") & (D.proxy <= 3.5)]
out["B_calm_anchor"] = {
    "generated_proxy<=3.5": {"n_items": int(len(g_calm)), "proxy_range": [float(g_calm.proxy.min()), float(g_calm.proxy.max())],
                             "human_mean": float(g_calm.human.mean()), "human_range": [float(g_calm.human.min()), float(g_calm.human.max())],
                             "raters_per_item": [int(g_calm["count"].min()), int(g_calm["count"].max())]},
    "deam_label<=4.6": {"n_items": int(len(d_calm)), "label_range": [float(d_calm["true"].min()), float(d_calm["true"].max())],
                        "proxy_oos_range": [float(d_calm.proxy.min()), float(d_calm.proxy.max())],
                        "human_mean": float(d_calm.human.mean()), "human_range": [float(d_calm.human.min()), float(d_calm.human.max())]},
    "midi_proxy<=3.5": {"n_items": int(len(c_calm)), "human_mean": float(c_calm.human.mean())},
    "midi_ends": {"c01": {"proxy": 3.17, "human": float(D[D.item == "c01"].human.iloc[0])},
                  "c12": {"proxy": 5.77, "human": float(D[D.item == "c12"].human.iloc[0])}},
    "raters_per_item_all_calib": [int(D["count"].min()), int(D["count"].max())]}
# rater-level: for each rater who rated both a calm generated and a calm DEAM item, sign of difference
c = pd.read_csv(os.path.join(LS, "data", "calib_ratings_template-24.csv"))
c["rating"] = pd.to_numeric(c.arousal_1to9, errors="coerce")
gi, di = set(g_calm.item), set(d_calm.item)
pairs = []
for p, grp in c.groupby("participant"):
    a = grp[grp.item.isin(gi)].rating; b = grp[grp.item.isin(di)].rating
    if len(a) and len(b):
        pairs.append(a.mean() - b.mean())
pairs = np.array(pairs)
out["B_calm_anchor"]["rater_level"] = {"n_raters": int(len(pairs)), "mean_diff_gen_minus_deam": float(pairs.mean()),
                                        "n_negative": int((pairs < 0).sum()),
                                        "wilcoxon_p": float(stats.wilcoxon(pairs).pvalue) if len(pairs) > 5 else None}

# ---------------------------------------------------------------- C
E = pd.DataFrame(LSUM["E_trajectory_level"]["per_item"])
sched = E[E.condition != "jump"]
C = {"pooled": {}, "scheduled_only": {}}
for q in ["overall_traj", "endpoint_calm", "smoothness", "start_match"]:
    for lab, df in [("pooled", E), ("scheduled_only", sched)]:
        rho, p = stats.spearmanr(df.track_rmse, df[q])
        C[lab][f"{q}~track_rmse"] = {"spearman": float(rho), "p": float(p), "n": int(len(df))}
rho, p = stats.spearmanr(sched.start_err, sched.start_match); C["scheduled_only"]["start_match~start_err"] = {"spearman": float(rho), "p": float(p), "n": int(len(sched))}
out["C_item_level"] = C

# ---------------------------------------------------------------- D
eps = []
for f in ["iso_episodes.jsonl", "iso_episodes_b.jsonl"]:
    eps += [json.loads(l) for l in open(os.path.join(R, f))]
first = [e["arousal_trajectory"][0] for e in eps if e["planner"] in ("rule", "rule+BoN")]
plateau = float(np.percentile(first, 95))
Dd = {"plateau_proxy_p95_first_segment": plateau, "plateau_max": float(max(first))}
res = {}
def cosine_schedule(s0, s_star, K):
    # same ISO schedule for every planner (the jump episodes store a flat schedule)
    return s_star + (s0 - s_star) * 0.5 * (1 + np.cos(np.pi * np.arange(K) / (K - 1)))
for e in eps:
    K = len(e["schedule"]); sch = cosine_schedule(e["s0"], e["s_star"], K)
    A_ = np.array(e["arousal_trajectory"]); Ai = np.array(e["arousal_trajectory_independent"])
    clipped = np.minimum(sch, plateau)
    res.setdefault(e["planner"], []).append({
        "track": float(np.sqrt(np.mean((A_ - sch) ** 2))),
        "track_clipped": float(np.sqrt(np.mean((A_ - clipped) ** 2))),
        "track_indep": float(np.sqrt(np.mean((Ai - sch) ** 2))),
        "track_clipped_indep": float(np.sqrt(np.mean((Ai - clipped) ** 2))),
        "lower_bound_seg1": float(max(0.0, e["s0"] - plateau)),
        "track_excl_seg1": float(np.sqrt(np.mean((A_[1:] - sch[1:]) ** 2)))})
Dd["planners"] = {pl: {k: float(np.mean([x[k] for x in v])) for k in v[0]} for pl, v in res.items()}
out["D_reachable"] = Dd

# ---------------------------------------------------------------- E (window stability)
from score import score_array
import librosa, joblib
rng = np.random.default_rng(3)
songs = rng.choice(deam.song_id.values, 60, replace=False)
win = {5: [], 12: []}; full = []
for sid in songs:
    path = os.path.join(ROOT, "data", "deam", "wav22", f"{sid}.wav")
    if not os.path.exists(path):
        continue
    y, sr = librosa.load(path, sr=22050, mono=True)
    fullp = score_array(y, sr, os.path.join(R, "verifier"))["arousal_pred"]
    label = float(deam[deam.song_id == sid].arousal_mean.iloc[0])
    full.append((sid, fullp, label))
    for w in (5, 12):
        n = int(w * sr); starts = np.arange(0, len(y) - n + 1, n)[:8]
        preds = [score_array(y[s:s + n], sr, os.path.join(R, "verifier"))["arousal_pred"] for s in starts]
        win[w].append((sid, float(np.mean(preds)), float(np.std(preds)), [float(p) for p in preds]))
fullp = np.array([f[1] for f in full]); labels = np.array([f[2] for f in full])
Ew = {"n_songs": int(len(full)), "r_full_vs_label": float(np.corrcoef(fullp, labels)[0, 1])}
for w in (5, 12):
    m = np.array([x[1] for x in win[w]]); sd = np.array([x[2] for x in win[w]])
    allw = np.concatenate([x[3] for x in win[w]]); alllab = np.concatenate([[f[2]] * len(x[3]) for f, x in zip(full, win[w])])
    allfull = np.concatenate([[f[1]] * len(x[3]) for f, x in zip(full, win[w])])
    Ew[f"win{w}s"] = {"r_windowmean_vs_full": float(np.corrcoef(m, fullp)[0, 1]),
                      "r_windowmean_vs_label": float(np.corrcoef(m, labels)[0, 1]),
                      "r_singlewindow_vs_full": float(np.corrcoef(allw, allfull)[0, 1]),
                      "r_singlewindow_vs_label": float(np.corrcoef(allw, alllab)[0, 1]),
                      "mean_within_song_sd": float(sd.mean()),
                      "mean_bias_window_minus_full": float((m - fullp).mean())}
out["E_window_stability"] = Ew

# ---------------------------------------------------------------- F (stimulus loudness)
import pyloudnorm
meter = pyloudnorm.Meter(22050)
key = pd.read_csv(os.path.join(LS, "key.csv"))
lufs = {}
for _, k in key[key.kind == "trajectory"].iterrows():
    y, sr = librosa.load(os.path.join(LS, "stimuli_flac", k["item"] + ".flac"), sr=22050, mono=True)
    lufs.setdefault(k["condition"], []).append(float(meter.integrated_loudness(y)))
out["F_stimulus_lufs"] = {c: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))} for c, v in lufs.items()}
# per-segment loudness of one jump vs rule clip (first vs last segment)
def seg_lufs(item):
    y, sr = librosa.load(os.path.join(LS, "stimuli_flac", item + ".flac"), sr=22050, mono=True)
    n = len(y) // 5
    return [float(meter.integrated_loudness(y[i * n:(i + 1) * n])) for i in range(5)]
out["F_stimulus_lufs"]["segments_example"] = {it: seg_lufs(it) for it in ["t01", "t02", "t03"]}

# ---------------------------------------------------------------- G
r = pd.read_csv(os.path.join(LS, "data", "ratings_template-24.csv"))
n_traj = r.groupby("item").participant.nunique()
out["G_raters_per_item"] = {"trajectory": [int(n_traj.min()), int(n_traj.max())], "median": float(n_traj.median()),
                           "calibration": [int(D["count"].min()), int(D["count"].max())]}

json.dump(out, open(os.path.join(R, "round3.json"), "w"), indent=1)
print(json.dumps(out, indent=1)[:12000])
