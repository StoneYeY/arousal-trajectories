"""Build the public data release (release/) for the paper's repository.

Everything is derived from files under results/, listening_study/ and
scripts/; nothing is typed by hand. The out-of-fold reference scores are
recomputed with the same recipe as scripts/oof_reference.py (KFold(5,
shuffle, seed 42), HGB max_iter=300, seed 42) and checked against
results/oof_reference.json before writing.

Output layout (release/):
  README.md
  data/prompts.csv                      70 audit prompts (+ the 20-prompt sweep subset flag)
  data/audit_scores.csv                 840 SA3 clips: proxy (deploy, fold-mean, LUFS-normalised), auxiliary, CLAP, LUFS, seeds
  data/audit_sweep_scores.csv           400 decoding-sweep clips
  data/audit_musicgen_scores.csv        280 MusicGen-small clips
  data/deam_reference_oof.csv           1,795 DEAM excerpts: fold, label, out-of-fold and in-sample proxy, reference-group flags
  data/planning_synth.json              100 synthesiser episodes (5 planners x 20) + best-of-N curve
  data/planning_sa3.json                200-candidate SA3 pool + per-planner selections and summary
  data/listening_ratings.csv            857 valid trajectory ratings (anonymised P01-P24)
  data/listening_calibration_ratings.csv 384 single-clip ratings
  data/listening_participants.csv       age, years of musical training, headphones (yes/no)
  data/listening_items.csv              24 trajectory items: planner, s0, schedule, proxy trajectory, metrics, n raters, mean ratings
  data/calibration_items.csv            32 single-clip items: domain, proxy, DEAM song id / label, intended level, n raters, listener mean
  stimuli/                              listening stimuli (t01-t24, c01-c12, g01-g12); DEAM excerpts are not redistributed (song ids given)
  models/                               proxy and auxiliary-estimator weights + training report
  code/                                 scripts used for every number in the paper
"""
import glob, json, os, shutil
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results"); LS = os.path.join(ROOT, "listening_study"); S = os.path.join(ROOT, "scripts")
OUT = os.path.join(ROOT, "release")
for d in ["data", "stimuli", "models", "code"]:
    os.makedirs(os.path.join(OUT, d), exist_ok=True)

# ------------------------------------------------------------------ prompts
P = json.load(open(os.path.join(S, "prompts_audit.json")))
SW = json.load(open(os.path.join(S, "prompts_sweep.json")))
sweep_prompts = set()
for grp in ("anxiolytic", "controls"):
    for cat, lst in SW[grp].items():
        sweep_prompts.update(lst)
rows = []
for grp in ("anxiolytic", "controls"):
    for cat, lst in P[grp].items():
        for i, text in enumerate(lst):
            rows.append({"prompt_id": f"{grp}.{cat}.p{i}", "group": grp, "category": cat, "prompt": text,
                         "in_decoding_sweep": text in sweep_prompts})
prompts = pd.DataFrame(rows); prompts.to_csv(os.path.join(OUT, "data", "prompts.csv"), index=False)
assert len(prompts) == 70, len(prompts)

# ------------------------------------------------------------------ fold models (same recipe as oof_reference.py)
pack = joblib.load(os.path.join(R, "verifier", "verifier_arousal_mean_deploy.joblib")); COLS = pack["cols"]
ipack = joblib.load(os.path.join(R, "verifier", "verifier_arousal_mean_independent.joblib")); ICOLS = ipack["cols"]
deam = pd.read_csv(os.path.join(R, "deam_features.csv")).dropna(subset=["arousal_mean"]).reset_index(drop=True)
X = deam[COLS].values; y = deam["arousal_mean"].values; XI = deam[ICOLS].values


def load_feats(pattern, key="feat_raw"):
    rows, seen = [], set()
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_error" in r or r["clip_id"] in seen or not r.get(key):
                continue
            seen.add(r["clip_id"]); rows.append(r)
    ids = [r["clip_id"] for r in rows]
    G = np.array([[r[key].get(c, np.nan) for c in COLS] for r in rows])
    return ids, G


sets = {"sa3": load_feats(os.path.join(R, "sa3_features*.jsonl")),
        "sa3_lufs": load_feats(os.path.join(R, "sa3_features*.jsonl"), key="feat_ln"),
        "musicgen": load_feats(os.path.join(R, "step3", "audit_musicgen_features*.jsonl"))}
for cfg in ["sweep_s8_cfg1.0", "sweep_s25_cfg1.0", "sweep_s25_cfg3.0", "sweep_s50_cfg3.0", "sweep_s50_cfg6.0"]:
    sets[cfg] = load_feats(os.path.join(R, "step3", cfg + "_features*.jsonl"))
oof = np.full(len(deam), np.nan); fold = np.full(len(deam), -1); oof_i = np.full(len(deam), np.nan)
gen_pred = {k: np.zeros((5, len(ids))) for k, (ids, _) in sets.items()}
for i, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=42).split(X)):
    m = HistGradientBoostingRegressor(random_state=42, max_iter=300).fit(X[tr], y[tr])
    oof[te] = m.predict(X[te]); fold[te] = i
    oof_i[te] = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]).fit(XI[tr], y[tr]).predict(XI[te])
    for k, (_, G) in sets.items():
        gen_pred[k][i] = m.predict(G)
foldmean = {k: dict(zip(ids, gen_pred[k].mean(0))) for k, (ids, _) in sets.items()}

q1, q3 = deam.arousal_mean.quantile([0.25, 0.75]); vmed = deam.valence_mean.median()
d1, d9 = deam.arousal_mean.quantile([0.10, 0.90])
ref = pd.DataFrame({"song_id": deam.song_id.astype(int), "fold": fold,
                    "arousal_label": deam.arousal_mean, "valence_label": deam.valence_mean,
                    "proxy_oof": oof, "proxy_insample": pack["model"].predict(X),
                    "aux_oof": oof_i, "aux_insample": ipack["model"].predict(XI),
                    "human_calm_ref": (deam.arousal_mean <= q1) & (deam.valence_mean >= vmed),
                    "human_tense_ref": deam.arousal_mean >= q3,
                    "arousal_lowest_quartile": deam.arousal_mean <= q1,
                    "arousal_lowest_decile": deam.arousal_mean <= d1,
                    "arousal_highest_decile": deam.arousal_mean >= d9,
                    "tense_low_valence_ref": (deam.arousal_mean >= q3) & (deam.valence_mean <= vmed)})
ref.to_csv(os.path.join(OUT, "data", "deam_reference_oof.csv"), index=False)
# consistency check against the published numbers
O = json.load(open(os.path.join(R, "oof_reference.json")))
assert abs(ref[ref.human_calm_ref].proxy_oof.mean() - O["refs"]["human_calm"]["oof"]["mean"]) < 1e-6
assert abs(ref[ref.human_tense_ref].proxy_oof.mean() - O["refs"]["human_tense"]["oof"]["mean"]) < 1e-6
assert int(ref.human_calm_ref.sum()) == O["refs"]["human_calm"]["n"] and int(ref.human_tense_ref.sum()) == O["refs"]["human_tense"]["n"]
fm = pd.read_csv(os.path.join(R, "audit_clips_foldmodels.csv")).set_index("clip_id").pred_oofmodels
assert max(abs(fm[c] - foldmean["sa3"][c]) for c in fm.index) < 1e-6
print("fold-model recomputation matches results/oof_reference.json and audit_clips_foldmodels.csv")

# ------------------------------------------------------------------ audit scores
man = {json.loads(l)["clip_id"]: json.loads(l) for l in open(os.path.join(ROOT, "data", "sa3_audit", "sa322", "manifest.jsonl"))}
a = pd.read_csv(os.path.join(R, "audit_table_clips.csv"))
a["prompt_id"] = a.clip_id.str.rsplit(".", n=1).str[0]
audit = pd.DataFrame({
    "clip_id": a.clip_id, "prompt_id": a.prompt_id, "group": a.group, "category": a.category, "prompt": a.prompt,
    "seed": [man[c]["seed"] for c in a.clip_id], "clip_seed": [man[c]["clip_seed"] for c in a.clip_id],
    "num_inference_steps": [man[c]["num_inference_steps"] for c in a.clip_id], "cfg_scale": [man[c]["cfg_scale"] for c in a.clip_id],
    "duration_s": [man[c]["duration_s"] for c in a.clip_id],
    "lufs_in": a.lufs_in,
    "proxy_arousal_deploy": a.arousal, "proxy_valence_deploy": a.valence,
    "proxy_arousal_foldmean": [foldmean["sa3"][c] for c in a.clip_id],
    "proxy_arousal_lufs23_deploy": a.arousal_ln, "proxy_arousal_lufs23_foldmean": [foldmean["sa3_lufs"].get(c, np.nan) for c in a.clip_id],
    "aux_arousal": a.arousal_indep, "aux_arousal_lufs23": a.arousal_indep_ln,
    "clap_similarity": a.clap_similarity,
    "tempo_bpm": a.tempo_bpm, "onset_density": a.onset_density, "spectral_flux_mean": a.spectral_flux_mean,
    "roughness": a.roughness, "rms_mean": a.rms_mean, "spectral_centroid_mean": a.spectral_centroid_mean, "mode_majorness": a.mode_majorness})
audit.to_csv(os.path.join(OUT, "data", "audit_scores.csv"), index=False)
assert len(audit) == 840

sw = pd.read_csv(os.path.join(R, "step3", "sweep_clips.csv"))
sw["proxy_arousal_foldmean"] = [foldmean[c][cid] for c, cid in zip(sw.config, sw.clip_id)]
sw = sw.rename(columns={"arousal": "proxy_arousal_deploy", "valence": "proxy_valence_deploy", "arousal_indep": "aux_arousal"})
sw.to_csv(os.path.join(OUT, "data", "audit_sweep_scores.csv"), index=False)
mg = pd.read_csv(os.path.join(R, "step3", "musicgen_clips.csv"))
mg["proxy_arousal_foldmean"] = [foldmean["musicgen"][cid] for cid in mg.clip_id]
mg = mg.rename(columns={"arousal": "proxy_arousal_deploy", "valence": "proxy_valence_deploy", "arousal_indep": "aux_arousal"})
mg.to_csv(os.path.join(OUT, "data", "audit_musicgen_scores.csv"), index=False)
assert len(sw) == 400 and len(mg) == 280

# ------------------------------------------------------------------ planning (synthesiser)
eps = []
for f in ["iso_episodes.jsonl", "iso_episodes_b.jsonl"]:
    eps += [json.loads(l) for l in open(os.path.join(R, f))]
for e in eps:  # a common cosine schedule for every planner (jump episodes store a flat one)
    K = e["K"]; e["iso_schedule"] = [float(v) for v in e["s_star"] + (e["s0"] - e["s_star"]) * 0.5 * (1 + np.cos(np.pi * np.arange(K) / (K - 1)))]
    e["proxy_trajectory"] = e.pop("arousal_trajectory"); e["aux_trajectory"] = e.pop("arousal_trajectory_independent")
    e["costs_aux"] = e.pop("costs_independent")
synth = {"description": "20 episodes x 5 planners on the parametric MIDI synthesiser; K=5 segments of 12 s; s*=3; N=4 candidates for rule+BoN/endpoint-BoN. "
                        "proxy_trajectory: deployed proxy per segment; aux_trajectory: auxiliary (ridge, 17 descriptors) estimator; "
                        "iso_schedule: cosine schedule from s0 to s* used for every planner's tracking error (the stored 'schedule' of jump episodes is the flat calm target).",
         "episodes": eps,
         "best_of_n_curve": json.load(open(os.path.join(R, "bon_dual_curve.json"))),
         "tracking_recomputed": json.load(open(os.path.join(R, "iso_tracking_recomputed.json")))}
json.dump(synth, open(os.path.join(OUT, "data", "planning_synth.json"), "w"), indent=1)

# ------------------------------------------------------------------ planning (SA3 pool)
pool = pd.read_csv(os.path.join(R, "step3", "traj_pool_clips.csv"))
S3 = json.load(open(os.path.join(R, "step3", "step3_summary.json")))
pool = pool.rename(columns={"arousal": "proxy_arousal_deploy", "valence": "proxy_valence_deploy", "arousal_indep": "aux_arousal"})
sa3 = {"description": "10 episodes x K=5 segments x N=4 stable-audio-3-medium candidates (12-s generations) whose prompt verbalises the scheduled level r_k. "
                      "Selection is offline: schedule+BoN picks argmin |proxy - r_k| per segment; the summary reproduces Table 2 of the paper.",
       "candidates": pool.to_dict("records"),
       "summary": S3["C_traj"]}
json.dump(sa3, open(os.path.join(OUT, "data", "planning_sa3.json"), "w"), indent=1)

# ------------------------------------------------------------------ listening study
key = pd.read_csv(os.path.join(LS, "key.csv")).set_index("item")
kd = pd.read_csv(os.path.join(LS, "calib_domains", "key_domains.csv")).set_index("item")
LSUM = json.load(open(os.path.join(R, "listening_summary.json")))
QMAP = {"1": "endpoint_calm", "2": "smoothness", "3": "start_match", "4": "overall_traj",
        "endpoint": "endpoint_calm", "smooth": "smoothness", "start_match": "start_match", "overall": "overall_traj"}
r = pd.read_csv(os.path.join(LS, "data", "ratings_template-24.csv"))
r["question"] = r.question.astype(str).map(lambda q: QMAP.get(q, q))
r["rating"] = pd.to_numeric(r.rating_1to7, errors="coerce")
r["valid"] = r.rating.between(1, 7)
r["planner"] = r["item"].map(key.condition); r["s0"] = r["item"].map(key.s0)
r[["participant", "item", "planner", "s0", "question", "rating_1to7", "valid"]].to_csv(os.path.join(OUT, "data", "listening_ratings.csv"), index=False)
assert int(r.valid.sum()) == LSUM["A_data"]["n_valid"]
c = pd.read_csv(os.path.join(LS, "data", "calib_ratings_template-24.csv"))
c["domain"] = c["item"].map(lambda it: "synthesiser" if it.startswith("c") else ("stable-audio-3" if it.startswith("g") else "DEAM"))
c[["participant", "item", "domain", "arousal_1to9"]].to_csv(os.path.join(OUT, "data", "listening_calibration_ratings.csv"), index=False)
pp = pd.read_csv(os.path.join(LS, "data", "participants_template-24.csv"), keep_default_na=False)
pp_out = pd.DataFrame({"participant": pp.participant, "age": pp.age_band, "music_training_years": pp.music_background_years,
                       "headphones": pp.headphones_model.astype(str).str.strip().str.upper().map(lambda s: "no" if s in ("N/A", "NA", "NONE", "", "NAN") else "yes")})
pp_out.to_csv(os.path.join(OUT, "data", "listening_participants.csv"), index=False)
assert int((pp_out.headphones == "yes").sum()) == LSUM["A_data"]["headphones_reported"]

E = pd.DataFrame(LSUM["E_trajectory_level"]["per_item"])
n_r = r[r.valid].groupby("item").participant.nunique()
items = pd.DataFrame({"item": E.item, "planner": E.condition, "s0": E.s0,
                      "schedule": [json.dumps([round(v, 3) for v in s]) for s in E.schedule],
                      "proxy_trajectory": [json.dumps([round(v, 3) for v in s]) for s in E.proxy_traj],
                      "track_rmse": E.track_rmse, "start_err": E.start_err, "max_step": E.max_step, "endpoint_err": E.endpoint_err,
                      "n_raters": [int(n_r[i]) for i in E.item],
                      "mean_endpoint_calm": E.endpoint_calm, "mean_smoothness": E.smoothness, "mean_start_match": E.start_match, "mean_overall_traj": E.overall_traj})
items.to_csv(os.path.join(OUT, "data", "listening_items.csv"), index=False)
D = pd.DataFrame(LSUM["D_proxy_validity"]["per_item"])
ids = json.load(open(os.path.join(LS, "calib_domains", "d_item_song_ids.json")))
DOM = {"midi": "synthesiser", "generated": "stable-audio-3", "real": "DEAM"}
calib = pd.DataFrame({"item": D.item, "domain": D.domain.map(DOM), "proxy_arousal": D.proxy,
                      "proxy_note": D.domain.map({"midi": "deployed proxy on the 12-s segment", "generated": "deployed proxy on the full 45-s generation (listeners heard the first 20 s)",
                                                  "real": "proxy refit without the 8 songs, scored on the 20-s excerpt heard"}),
                      "deam_song_id": [ids[i]["song_id"] if i in ids else "" for i in D.item],
                      "deam_arousal_label": D["true"], "intended_level": D.intended,
                      "n_raters": D["count"].astype(int), "listener_mean_1to9": D.human, "listener_sd": D["std"]})
calib.to_csv(os.path.join(OUT, "data", "calibration_items.csv"), index=False)

# ------------------------------------------------------------------ stimuli, models, code
for f in sorted(glob.glob(os.path.join(LS, "stimuli_flac", "*.flac"))):
    shutil.copy(f, os.path.join(OUT, "stimuli", os.path.basename(f)))
for f in sorted(glob.glob(os.path.join(LS, "calib_domains", "g*.flac"))):
    shutil.copy(f, os.path.join(OUT, "stimuli", os.path.basename(f)))
for f in ["verifier_arousal_mean_deploy.joblib", "verifier_valence_mean_deploy.joblib",
          "verifier_arousal_mean_independent.joblib", "verifier_valence_mean_independent.joblib", "report.json"]:
    shutil.copy(os.path.join(R, "verifier", f), os.path.join(OUT, "models", f))
for f in ["features.py", "score.py", "prep_deam.py", "train_verifier.py", "audit_extract.py", "audit_score.py", "audit_stats.py",
          "oof_reference.py", "midi_synth.py", "iso_agent.py", "bon_curve.py", "gen_traj_pool.py", "step3_analysis.py",
          "clap_alignment.py", "make_stimuli.py", "analyze_listening.py", "round3_analysis.py", "make_figs.py", "build_demo.py",
          "make_release.py", "audit_generate_sa3.py", "audit_generate.py", "prompts_audit.json", "prompts_sweep.json", "tempo_check.py"]:
    if os.path.exists(os.path.join(S, f)):
        shutil.copy(os.path.join(S, f), os.path.join(OUT, "code", f))
for f in ["protocol.md", "key.csv", "randomization.csv", "calib_randomization.csv", "item_start_levels.csv"]:
    shutil.copy(os.path.join(LS, f), os.path.join(OUT, "data", "listening_" + f))
shutil.copy(os.path.join(LS, "calib_domains", "key_domains.csv"), os.path.join(OUT, "data", "listening_key_domains.csv"))
shutil.copy(os.path.join(R, "round3.json"), os.path.join(OUT, "data", "round3_analyses.json"))
shutil.copy(os.path.join(R, "listening_summary.json"), os.path.join(OUT, "data", "listening_summary.json"))
shutil.copy(os.path.join(R, "oof_reference.json"), os.path.join(OUT, "data", "oof_reference_summary.json"))

README = f"""# Data and code release: Auditing and Steering Arousal Trajectories in Text-to-Music Generation

Audio companion page: https://stoneyey.github.io/arousal-trajectories/

Every number in the paper is computed by the scripts in `code/` from the files in `data/`
(plus the DEAM features, which `code/prep_deam.py` and `code/features.py` regenerate from the
public DEAM corpus). Scores are on the DEAM 1–9 arousal scale.

## data/
| file | rows | what it is |
|---|---|---|
| `prompts.csv` | 70 | audit prompts (5 anxiolytic categories x 10, neutral x 10, high-arousal x 10); `in_decoding_sweep` marks the 20 reused in Table 1 |
| `audit_scores.csv` | 840 | stable-audio-3-medium generations (45 s, 12 seeds/prompt): sampler settings and seeds; `proxy_arousal_deploy` (proxy refit on all DEAM), `proxy_arousal_foldmean` (mean of the five fold models = the paper's out-of-fold footing), `*_lufs23_*` (after -23 LUFS normalisation), `aux_arousal` (auxiliary ridge estimator), CLAP similarity, integrated loudness, six descriptors |
| `audit_sweep_scores.csv` | 400 | decoding sweep (5 configurations x 20 prompts x 4 seeds), same columns plus `config`, `peak_in` |
| `audit_musicgen_scores.csv` | 280 | MusicGen-small replication (70 prompts x 4 seeds, 30 s) |
| `deam_reference_oof.csv` | {len(ref)} | every DEAM excerpt: fold (KFold(5, shuffle, seed 42)), labels, out-of-fold and in-sample proxy/auxiliary scores, and the reference-group flags (`human_calm_ref` = lowest-arousal quartile & above-median valence, n={int(ref.human_calm_ref.sum())}; `human_tense_ref` = highest quartile, n={int(ref.human_tense_ref.sum())}; the other flags give the three alternative definitions) |
| `planning_synth.json` | 100 episodes | synthesiser planning: per-episode schedule, proxy and auxiliary trajectories, costs; best-of-N dual curve |
| `planning_sa3.json` | 200 candidates | stable-audio-3-medium candidate pool (10 episodes x 5 segments x 4 candidates) and the per-planner selection summary (Table 2) |
| `listening_ratings.csv` | {len(r)} | trajectory ratings, 24 listeners (P01–P24), 9 items each, 4 questions (7-point); `valid` flags the 857 rows used (3 out-of-range, 4 blank) |
| `listening_calibration_ratings.csv` | {len(c)} | single-clip perceived-arousal ratings (1–9), 16 items per listener |
| `listening_participants.csv` | 24 | age, years of musical training, headphones yes/no (no other personal data was collected) |
| `listening_items.csv` | 24 | trajectory stimuli: planner, s0, cosine schedule, proxy trajectory, tracking metrics, number of raters, mean ratings |
| `calibration_items.csv` | 32 | single-clip items: domain, proxy score (and how it was computed), DEAM song id and label for the 8 real-music items, intended level for the 12 synthesiser presets, number of raters, listener mean |
| `listening_protocol.md`, `listening_key*.csv`, `listening_randomization.csv`, `listening_calib_randomization.csv`, `listening_item_start_levels.csv` | | the protocol and the per-listener presentation orders |
| `listening_summary.json`, `round3_analyses.json`, `oof_reference_summary.json` | | the analysis outputs the paper quotes (Sections 3–4) |

Listening scales, as read to participants (translated from the Chinese protocol): endpoint calmness 1 = not calm at all … 7 = very calm;
transition smoothness 1 = abrupt … 7 = very smooth; start match 1 = does not match the stated start at all … 7 = matches very well
(too high or too low both count as mismatch); overall trajectory 1 = not realised at all … 7 = realised very well;
single clips: 1 = extremely calm … 9 = extremely aroused. Before each trajectory the stated starting level s0 was read out.

## stimuli/
`t01–t24`: the 24 synthesiser trajectories (five 5-s segments joined by hard cuts, not loudness-matched; jump items average
about -43 LUFS and scheduled items about -25 LUFS). `c01–c12`: the 12 synthesiser calibration segments (12 s).
`g01–g12`: the 12 stable-audio-3 calibration generations (45 s; listeners heard the first 20 s). The 8 DEAM excerpts
(`d01–d08`) are not redistributed; they are DEAM songs {', '.join(str(ids[k]['song_id']) for k in sorted(ids))} (first 20 s of the
45-s excerpt), see `calibration_items.csv`.

## models/
`verifier_*_deploy.joblib`: the deployed proxy (HistGradientBoosting on 17 descriptors + 88 eGeMAPS functionals, refit on all
1,795 DEAM excerpts; `cols` lists the feature order). `verifier_*_independent.joblib`: the auxiliary ridge estimator on the 17
descriptors. `report.json`: the training protocol and held-out metrics. Score new audio with `code/score.py`.

## code/
Feature extraction (`features.py`), proxy training (`train_verifier.py`), audit scoring and statistics (`audit_extract.py`,
`audit_score.py`, `audit_stats.py`, `oof_reference.py`), planning (`midi_synth.py`, `iso_agent.py`, `bon_curve.py`,
`gen_traj_pool.py`, `step3_analysis.py`), listening-study stimuli and analysis (`make_stimuli.py`, `analyze_listening.py`),
the round-3 analyses (`round3_analysis.py`), figures (`make_figs.py`), the companion page (`build_demo.py`) and this release
(`make_release.py`). Generation scripts (`audit_generate_sa3.py`, `audit_generate.py`) were run on a separate GPU host with
stable-audio-tools 0.0.20 / audiocraft; sampler settings and seeds are in the score tables.

## Ethics
The listening test involved 24 adult volunteers; no compensation was paid, no personal data beyond age, years of musical
training and headphone use were recorded, and ratings are released under pseudonyms P01–P24.
"""
open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(README)
print("release written to", OUT)
for d in ["data", "stimuli", "models", "code"]:
    n = len(os.listdir(os.path.join(OUT, d))); sz = sum(os.path.getsize(os.path.join(OUT, d, f)) for f in os.listdir(os.path.join(OUT, d))) / 1e6
    print(f"  {d}: {n} files, {sz:.1f} MB")
