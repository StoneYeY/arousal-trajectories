"""Build the self-contained audio companion page (docs/index.html).

Everything embedded as data-URI MP3 (48 kbps mono, 22.05 kHz); no
external requests. All numbers shown come from results/*.json.
"""
import base64, json, os, subprocess, tempfile
import numpy as np, pandas as pd, librosa, soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results"); LS = os.path.join(ROOT, "listening_study")
OUT = os.path.join(ROOT, "docs"); os.makedirs(OUT, exist_ok=True)
SR = 22050


def mp3_b64(y, sr=SR, kbps=48):
    with tempfile.TemporaryDirectory() as d:
        w = os.path.join(d, "a.wav"); m = os.path.join(d, "a.mp3")
        sf.write(w, y.astype(np.float32), sr)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", w, "-ac", "1", "-ar", str(sr),
                        "-codec:a", "libmp3lame", "-b:a", f"{kbps}k", m], check=True)
        return "data:audio/mpeg;base64," + base64.b64encode(open(m, "rb").read()).decode()


def load(path, start=0.0, dur=None):
    y, _ = librosa.load(path, sr=SR, mono=True, offset=start, duration=dur)
    return y


def fade(y, t=0.15):
    n = int(t * SR); y = y.copy()
    y[:n] *= np.linspace(0, 1, n); y[-n:] *= np.linspace(1, 0, n)
    return y


def norm(y, peak=0.9):
    m = np.max(np.abs(y)) or 1.0
    return y * (peak / m)


# ------------------------------------------------------------ data
audit = pd.read_csv(os.path.join(R, "audit_table_clips.csv"))
# out-of-fold footing (paper Sec. 4.1): reference bands and clip scores from the fold models
OOF = json.load(open(os.path.join(R, "oof_reference.json")))
fold = pd.read_csv(os.path.join(R, "audit_clips_foldmodels.csv")).set_index("clip_id").pred_oofmodels
REF = {"calm": OOF["refs"]["human_calm"]["oof"]["mean"], "tense": OOF["refs"]["human_tense"]["oof"]["mean"]}
LS_SUM = json.load(open(os.path.join(R, "listening_summary.json")))
E = {r["item"]: r for r in LS_SUM["E_trajectory_level"]["per_item"]}
D = {r["item"]: r for r in LS_SUM["D_proxy_validity"]["per_item"]}
B = LS_SUM["B_mixed"]
pool = pd.read_csv(os.path.join(R, "step3", "traj_pool_clips.csv"))
S3 = json.load(open(os.path.join(R, "step3", "step3_summary.json")))

# 1. audit examples: clip nearest each category mean + the most intense clip
AUDIT_IDS = [("sleep", "anxiolytic.sleep.p6.g9"), ("meditation", "anxiolytic.meditation.p2.g0"),
             ("anxiety relief", "anxiolytic.anxiety_relief.p2.g5"), ("neutral", "controls.neutral.p9.g7"),
             ("high arousal (typical)", "controls.high_arousal.p5.g10"),
             ("high arousal (most intense of 120)", "controls.high_arousal.p0.g7")]
audit_items = []
for label, cid in AUDIT_IDS:
    row = audit[audit.clip_id == cid].iloc[0]
    y = norm(fade(load(os.path.join(ROOT, "data", "sa3_audit", "sa322", cid + ".flac"), 0, 20)))
    audit_items.append({"label": label, "prompt": row.prompt, "proxy": round(float(fold[cid]), 2),
                        "indep": round(float(row.arousal_indep), 2), "lufs": round(float(row.lufs_in), 1),
                        "cat_mean": round(float(OOF["sa3"]["per_category"][row.category]["mean"]), 2),
                        "src": mp3_b64(y)})

# 2. blinded listening sets
key = pd.read_csv(os.path.join(LS, "key.csv")).set_index("item")
sets = []
for s0, items in [(7.38, ["t01", "t02", "t03"]), (8.12, ["t16", "t17", "t18"])]:
    rows = []
    for it in items:
        y = norm(load(os.path.join(LS, "stimuli_flac", it + ".flac")))
        e = E[it]
        rows.append({"item": it, "planner": e["condition"], "proxy_traj": [round(x, 2) for x in e["proxy_traj"]],
                     "schedule": [round(x, 2) for x in e["schedule"]],
                     "ratings": {k: round(e[k], 2) for k in ["endpoint_calm", "smoothness", "start_match", "overall_traj"]},
                     "track_rmse": round(e["track_rmse"], 2), "src": mp3_b64(y)})
    sets.append({"s0": s0, "rows": rows})
paper_means = {q: {c: round(v["mean"], 2) for c, v in B[q]["raw_means"].items()} for q in B}

# 3. SA3 transfer (episode 1, s0 = 8.19): prompted / schedule+BoN / jump, 6-s slices
ep = pool[pool.episode == 1].sort_values(["segment", "candidate"])
K = 5; s0 = float(ep.s0.iloc[0])
sch = 3.0 + (s0 - 3.0) * 0.5 * (1 + np.cos(np.pi * np.arange(K) / (K - 1)))
pools = [ep[ep.segment == k] for k in range(K)]
def pick(kind):
    idx = []
    for k in range(K):
        P = pools[k]
        if kind == "prompted":
            idx.append((k, 0))
        elif kind == "schedule+BoN":
            idx.append((k, int(np.argmin(np.abs(P.arousal.values - sch[k])))))
        elif kind == "jump":
            idx.append((K - 1, k % 4))
    return idx
transfer = {"s0": round(s0, 2), "schedule": [round(x, 2) for x in sch], "rows": []}
for kind in ["prompted", "schedule+BoN", "jump"]:
    segs, A, Ai, prompts = [], [], [], []
    for (k, c) in pick(kind):
        P = pools[k].iloc[c]
        y = load(os.path.join(ROOT, "data", "step3", "flac_traj_pool", P.clip_id + ".flac"), 0, 6)
        segs.append(fade(y, 0.2)); A.append(round(float(P.arousal), 2)); Ai.append(round(float(P.arousal_indep), 2))
        prompts.append(P.prompt.split(",")[0])
    y = norm(np.concatenate(segs))
    A_ = np.array(A)
    transfer["rows"].append({"planner": kind, "proxy_traj": A, "indep_traj": Ai, "prompts": prompts,
                             "track_rmse": round(float(np.sqrt(np.mean((A_ - sch) ** 2))), 2), "src": mp3_b64(y)})
T = S3["C_traj"]["planners"]
transfer["paper"] = {p: {"deploy": round(T[p]["tracking_rmse_deploy"]["mean"], 2), "indep": round(T[p]["tracking_rmse_indep"]["mean"], 2)}
                     for p in ["prompted", "schedule+BoN", "jump"]}

# 4. calibration items
calib = []
for it, dom, path, dur in [("c01", "synthesiser", os.path.join(LS, "stimuli_flac", "c01.flac"), None),
                           ("c05", "synthesiser", os.path.join(LS, "stimuli_flac", "c05.flac"), None),
                           ("c09", "synthesiser", os.path.join(LS, "stimuli_flac", "c09.flac"), None),
                           ("c12", "synthesiser", os.path.join(LS, "stimuli_flac", "c12.flac"), None),
                           ("g02", "stable-audio-3", os.path.join(LS, "calib_domains", "g02.flac"), 15),
                           ("g06", "stable-audio-3", os.path.join(LS, "calib_domains", "g06.flac"), 15),
                           ("g10", "stable-audio-3", os.path.join(LS, "calib_domains", "g10.flac"), 15),
                           ("g12", "stable-audio-3", os.path.join(LS, "calib_domains", "g12.flac"), 15)]:
    y = norm(fade(load(path, 0, dur)))
    d = D[it]
    calib.append({"item": it, "domain": dom, "proxy": round(float(d["proxy"]), 2),
                  "human": round(float(d["human"]), 2), "n": int(d["count"]),
                  "intended": (round(float(d["intended"]), 1) if d.get("intended") == d.get("intended") and d.get("intended") is not None else None),
                  "src": mp3_b64(y)})
DOM = LS_SUM["D_proxy_validity"]["domains"]
rho = {"synthesiser": round(DOM["midi"]["rho_proxy_human"]["spearman"], 2),
       "stable-audio-3": round(DOM["generated"]["rho_proxy_human"]["spearman"], 2),
       "real": round(DOM["real"]["rho_proxy_human"]["spearman"], 2)}

DATA = {"ref": {k: round(v, 2) for k, v in REF.items()}, "audit": audit_items, "sets": sets,
        "paper_means": paper_means, "transfer": transfer, "calib": calib, "rho": rho,
        "audit_summary": {"anx": round(OOF["sa3"]["anxiolytic"]["mean"], 2),
                          "high": round(OOF["sa3"]["high_arousal"]["mean"], 2)}}

html = open(os.path.join(ROOT, "scripts", "demo_template.html"), encoding="utf-8").read()
html = html.replace("/*__DATA__*/", "const DATA = " + json.dumps(DATA) + ";")
open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(html)
print("wrote docs/index.html", round(os.path.getsize(os.path.join(OUT, "index.html")) / 1e6, 2), "MB")
