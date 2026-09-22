"""Step-3 analyses (all offline, from cached features in results/step3/).

  A. decoding-config sweep   : is the high-arousal gap a decoding artefact?
  B. musicgen-small audit    : does the calm~human / high<human asymmetry
                               replicate on a second (autoregressive) T2M model?
  C. SA3 trajectory pool     : offline selection with the audited model as
                               the action space (transfer of Eq.(sel)), plus
                               global-search / oracle baselines, scored by the
                               deploy verifier AND the independent estimator.
  D. CLAP matched vs shuffled: context baseline for the 0.29 alignment score.

Outputs results/step3/step3_summary.json (+ per-clip csvs).
"""
import argparse, glob, itertools, json, os, sys
import numpy as np, pandas as pd, joblib
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_stats import boot_ci  # prompt-clustered bootstrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R3 = os.path.join(ROOT, "results", "step3")
VDIR = os.path.join(ROOT, "results", "verifier")
SETS = ["sweep_s8_cfg1.0", "sweep_s25_cfg1.0", "sweep_s25_cfg3.0",
        "sweep_s50_cfg3.0", "sweep_s50_cfg6.0", "audit_musicgen", "traj_pool"]


def load_set(name):
    rows, seen = [], set()
    for shard in sorted(glob.glob(os.path.join(R3, f"{name}_features*.jsonl"))):
        for line in open(shard):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_error" in r or r["clip_id"] in seen:
                continue
            seen.add(r["clip_id"]); rows.append(r)
    return rows


_PACKS = {}
def predict(feat, variant, tgt="arousal_mean"):
    if variant not in _PACKS:
        _PACKS[variant] = {t: joblib.load(os.path.join(
            VDIR, f"verifier_{t}_{variant}.joblib"))
            for t in ["arousal_mean", "valence_mean"]}
    p = _PACKS[variant][tgt]
    X = np.array([[feat.get(c, np.nan) for c in p["cols"]]])
    return float(p["model"].predict(X)[0])


def frame(rows):
    recs = []
    for r in rows:
        f = r["feat_raw"]
        rec = {"clip_id": r["clip_id"], "group": r.get("group"),
               "category": r.get("category"), "prompt": r["prompt"],
               "gen_index": r.get("gen_index"), "model": r.get("model"),
               "lufs_in": r.get("lufs_in"), "peak_in": r.get("peak_in"),
               "arousal": predict(f, "deploy"),
               "valence": predict(f, "deploy", "valence_mean"),
               "arousal_indep": predict(f, "independent"),
               "tempo_bpm": f.get("tempo_bpm"), "onset_density": f.get("onset_density"),
               "rms_mean": f.get("rms_mean"), "spectral_flux_mean": f.get("spectral_flux_mean"),
               "roughness": f.get("roughness")}
        rec.update({k: v for k, v in (r.get("meta") or {}).items()
                    if k in ("episode", "segment", "candidate", "s0", "r_k",
                             "clip_seed", "seed", "num_inference_steps",
                             "cfg_scale", "guidance_scale")})
        recs.append(rec)
    return pd.DataFrame(recs)


def refs():
    t = json.load(open(os.path.join(ROOT, "results", "audit_table.json")))["groups"]
    return {k.split("/")[1]: {"true": t[k]["arousal_true"]["mean"],
                              "deploy": t[k]["arousal_pred"]["mean"],
                              "indep": t[k]["arousal_pred_independent"]["mean"]}
            for k in t if k.startswith("ref/")}


def group_stats(df, label):
    out = {}
    for grp, sub in [("anxiolytic", df[df.group == "anxiolytic"]),
                     ("high_arousal", df[df.category == "high_arousal"]),
                     ("neutral", df[df.category == "neutral"])]:
        if len(sub) == 0:
            continue
        out[grp] = {"n": int(len(sub)),
                    "deploy": boot_ci(sub.arousal, clusters=sub.prompt),
                    "indep": boot_ci(sub.arousal_indep, clusters=sub.prompt),
                    "valence": boot_ci(sub.valence, clusters=sub.prompt),
                    "lufs_in": float(sub.lufs_in.mean()) if sub.lufs_in.notna().any() else None,
                    "frac_peak_ge_0.99": float((sub.peak_in >= 0.99).mean()) if sub.peak_in.notna().any() else None,
                    "tempo_bpm": float(sub.tempo_bpm.mean()),
                    "onset_density": float(sub.onset_density.mean()),
                    "rms_mean": float(sub.rms_mean.mean())}
    return out


# ------------------------------------------------------------------ A. sweep
def analyse_sweep(summary):
    R = refs()
    sweep_prompts = json.load(open(os.path.join(ROOT, "scripts", "prompts_sweep.json")))
    sp = set()
    for k, v in sweep_prompts.items():
        if k == "design_note":
            continue
        if isinstance(v, dict):
            for lst in v.values():
                sp.update(lst)
        elif isinstance(v, list):
            sp.update(v)
    # original audit on the same 20 prompts, first 4 seeds (same config as s8/cfg1)
    clips = pd.read_csv(os.path.join(ROOT, "results", "audit_table_clips.csv"))
    clips["gen_index"] = clips.clip_id.str.extract(r"\.g(\d+)$").astype(int)
    orig = clips[clips.prompt.isin(sp) & (clips.gen_index < 4)].copy()
    orig["arousal_indep"] = orig["arousal_indep"]
    res = {"refs": R, "n_sweep_prompts": len(sp),
           "configs": {"orig_audit_s8_cfg1.0_subset": group_stats(orig.assign(
               peak_in=np.nan, lufs_in=orig.lufs_in), "orig")}}
    per_clip = []
    for name in SETS[:5]:
        df = frame(load_set(name))
        if len(df) == 0:
            continue
        df["config"] = name
        per_clip.append(df)
        res["configs"][name] = group_stats(df, name)
    if per_clip:
        allc = pd.concat(per_clip)
        allc.to_csv(os.path.join(R3, "sweep_clips.csv"), index=False)
        # replication check: same clip ids in orig audit vs sweep s8/cfg1
        rep = allc[allc.config == "sweep_s8_cfg1.0"].merge(
            orig[["clip_id", "arousal"]], on="clip_id", suffixes=("", "_orig"))
        if len(rep):
            res["replication_s8_cfg1_vs_original"] = {
                "n": int(len(rep)),
                "r": float(np.corrcoef(rep.arousal, rep.arousal_orig)[0, 1]),
                "mean_abs_diff": float(np.abs(rep.arousal - rep.arousal_orig).mean())}
        # gap table
        gaps = {}
        for name, g in res["configs"].items():
            if "high_arousal" in g and "anxiolytic" in g:
                gaps[name] = {
                    "calm_gap_deploy": g["anxiolytic"]["deploy"]["mean"] - R["human_calm"]["deploy"],
                    "high_gap_deploy": g["high_arousal"]["deploy"]["mean"] - R["human_tense"]["deploy"],
                    "high_gap_indep": g["high_arousal"]["indep"]["mean"] - R["human_tense"]["indep"],
                    "high_minus_calm_deploy": g["high_arousal"]["deploy"]["mean"] - g["anxiolytic"]["deploy"]["mean"],
                    "high_ci_deploy": [g["high_arousal"]["deploy"]["lo"], g["high_arousal"]["deploy"]["hi"]],
                    "high_ci_indep": [g["high_arousal"]["indep"]["lo"], g["high_arousal"]["indep"]["hi"]],
                    "high_frac_clipped": g["high_arousal"]["frac_peak_ge_0.99"],
                    "high_lufs": g["high_arousal"]["lufs_in"]}
        res["gaps"] = gaps
        # max over configs of high-arousal mean (best case for the model)
        hs = {n: g["high_arousal"]["deploy"]["mean"] for n, g in res["configs"].items()
              if "high_arousal" in g}
        res["high_arousal_best_config"] = max(hs, key=hs.get)
        res["high_arousal_best_mean"] = hs[res["high_arousal_best_config"]]
        # per-clip max: does ANY clip in any config reach human_tense?
        hi = allc[allc.category == "high_arousal"]
        res["frac_high_clips_above_human_tense_deploy"] = float(
            (hi.arousal >= R["human_tense"]["deploy"]).mean())
        res["high_clip_arousal_p90_deploy"] = float(hi.arousal.quantile(0.9))
    summary["A_sweep"] = res


# --------------------------------------------------------------- B. musicgen
def analyse_musicgen(summary):
    R = refs()
    df = frame(load_set("audit_musicgen"))
    if len(df) == 0:
        return
    df.to_csv(os.path.join(R3, "musicgen_clips.csv"), index=False)
    res = {"n": int(len(df)), "model": df.model.iloc[0], "refs": R}
    res["groups"] = group_stats(df, "musicgen")
    per_cat = {}
    for cat, sub in df[df.group == "anxiolytic"].groupby("category"):
        per_cat[cat] = {"deploy": boot_ci(sub.arousal, clusters=sub.prompt),
                        "indep": boot_ci(sub.arousal_indep, clusters=sub.prompt)}
    res["anxiolytic_by_category"] = per_cat
    g = res["groups"]
    res["gaps"] = {
        "calm_gap_deploy": g["anxiolytic"]["deploy"]["mean"] - R["human_calm"]["deploy"],
        "calm_gap_indep": g["anxiolytic"]["indep"]["mean"] - R["human_calm"]["indep"],
        "high_gap_deploy": g["high_arousal"]["deploy"]["mean"] - R["human_tense"]["deploy"],
        "high_gap_indep": g["high_arousal"]["indep"]["mean"] - R["human_tense"]["indep"]}
    # prompt-level comparison to SA3 (same 70 prompts)
    clips = pd.read_csv(os.path.join(ROOT, "results", "audit_table_clips.csv"))
    sa3p = clips.groupby("prompt").arousal.mean()
    mgp = df.groupby("prompt").arousal.mean()
    j = pd.concat([sa3p.rename("sa3"), mgp.rename("mg")], axis=1).dropna()
    res["prompt_level_sa3_vs_musicgen"] = {
        "n_prompts": int(len(j)),
        "spearman": float(stats.spearmanr(j.sa3, j.mg).correlation),
        "pearson": float(np.corrcoef(j.sa3, j.mg)[0, 1])}
    # mixed-effects style variance split (prompt vs residual) via ANOVA
    sub = df[df.group == "anxiolytic"]
    grand = sub.arousal.mean()
    pm = sub.groupby("prompt").arousal.transform("mean")
    res["anxiolytic_var_between_prompt"] = float(((pm - grand) ** 2).mean())
    res["anxiolytic_var_within_prompt"] = float(((sub.arousal - pm) ** 2).mean())
    # Mann-Whitney high vs human_tense clips (distribution level)
    deam = pd.read_csv(os.path.join(ROOT, "results", "deam_features.csv"))
    q3 = deam.arousal_mean.quantile(0.75)
    tense_true = deam[deam.arousal_mean >= q3].arousal_mean
    hi = df[df.category == "high_arousal"].arousal
    res["high_vs_human_tense_MWU_p"] = float(stats.mannwhitneyu(hi, tense_true).pvalue)
    summary["B_musicgen"] = res


# -------------------------------------------------------------- C. traj pool
def traj_metrics(A, sch, s_star):
    A = np.asarray(A, float); sch = np.asarray(sch, float)
    return {"tracking_rmse": float(np.sqrt(np.mean((A - sch) ** 2))),
            "start_err": float(abs(A[0] - sch[0])),
            "smooth_max_step": float(np.max(np.abs(np.diff(A)))),
            "endpoint_err": float(abs(A[-1] - s_star))}


def analyse_traj(summary, s_star=3.0, lam=0.5):
    df = frame(load_set("traj_pool"))
    if len(df) == 0:
        return
    df.to_csv(os.path.join(R3, "traj_pool_clips.csv"), index=False)
    eps = sorted(df.episode.unique())
    K = int(df.segment.max()) + 1
    N = int(df.candidate.max()) + 1
    res = {"n_clips": int(len(df)), "n_episodes": len(eps), "K": K, "N": N,
           "lambda_global": lam}
    # pool descriptive: does prompted level track r_k at all?
    res["pool_corr_rk_vs_deploy"] = float(np.corrcoef(df.r_k, df.arousal)[0, 1])
    res["pool_corr_rk_vs_indep"] = float(np.corrcoef(df.r_k, df.arousal_indep)[0, 1])
    res["pool_by_prompt"] = {p: {"n": int(len(s)), "r_k_mean": float(s.r_k.mean()),
                                 "deploy": float(s.arousal.mean()),
                                 "deploy_sd": float(s.arousal.std()),
                                 "indep": float(s.arousal_indep.mean())}
                             for p, s in df.groupby("prompt")}

    rng = np.random.default_rng(0)
    per_ep = {}   # planner -> list of (metrics_deploy, metrics_indep)
    def add(pl, Ad, Ai, sch):
        per_ep.setdefault(pl, []).append((traj_metrics(Ad, sch, s_star),
                                          traj_metrics(Ai, sch, s_star)))
    for e in eps:
        E = df[df.episode == e]
        sch = [float(E[E.segment == k].r_k.iloc[0]) for k in range(K)]
        pools_d = [E[E.segment == k].sort_values("candidate").arousal.values for k in range(K)]
        pools_i = [E[E.segment == k].sort_values("candidate").arousal_indep.values for k in range(K)]
        # prompted (no selection, candidate 0)
        add("prompted", [p[0] for p in pools_d], [p[0] for p in pools_i], sch)
        # schedule + BoN (Eq. sel, deploy selects)
        idx = [int(np.argmin(np.abs(pools_d[k] - sch[k]))) for k in range(K)]
        add("schedule+BoN", [pools_d[k][idx[k]] for k in range(K)],
            [pools_i[k][idx[k]] for k in range(K)], sch)
        # endpoint-BoN (select toward s* every segment)
        idx = [int(np.argmin(np.abs(pools_d[k] - s_star))) for k in range(K)]
        add("endpoint-BoN", [pools_d[k][idx[k]] for k in range(K)],
            [pools_i[k][idx[k]] for k in range(K)], sch)
        # jump: target level from the start = the r_K pool for every segment
        jd, ji = pools_d[K - 1], pools_i[K - 1]
        add("jump", [jd[k % N] for k in range(K)], [ji[k % N] for k in range(K)], sch)
        j = int(np.argmin(np.abs(jd - s_star)))
        add("jump+BoN", [jd[j]] * K, [ji[j]] * K, sch)
        # random candidate per segment (expectation over 200 draws)
        md, mi = [], []
        for _ in range(200):
            idx = rng.integers(0, N, K)
            md.append(traj_metrics([pools_d[k][idx[k]] for k in range(K)], sch, s_star))
            mi.append(traj_metrics([pools_i[k][idx[k]] for k in range(K)], sch, s_star))
        per_ep.setdefault("random", []).append((
            {k: float(np.mean([m[k] for m in md])) for k in md[0]},
            {k: float(np.mean([m[k] for m in mi])) for k in mi[0]}))
        # global search: exhaustive N^K, minimise tracking + lam*max-step (deploy)
        best, bestJ = None, np.inf
        for combo in itertools.product(range(N), repeat=K):
            Ad = np.array([pools_d[k][combo[k]] for k in range(K)])
            J = np.sqrt(np.mean((Ad - sch) ** 2)) + lam * np.max(np.abs(np.diff(Ad)))
            if J < bestJ:
                bestJ, best = J, combo
        add("global-search", [pools_d[k][best[k]] for k in range(K)],
            [pools_i[k][best[k]] for k in range(K)], sch)
        # oracle: independent estimator selects (upper bound for transfer)
        idx = [int(np.argmin(np.abs(pools_i[k] - sch[k]))) for k in range(K)]
        add("oracle-indep-select", [pools_d[k][idx[k]] for k in range(K)],
            [pools_i[k][idx[k]] for k in range(K)], sch)
        # a pool 'best case': choose per segment the candidate closest under indep
        # is oracle above; also record min achievable tracking under deploy
    table = {}
    for pl, lst in per_ep.items():
        table[pl] = {}
        for i, tag in enumerate(["deploy", "indep"]):
            for m in lst[0][i]:
                v = np.array([x[i][m] for x in lst])
                table[pl][f"{m}_{tag}"] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                                           "n": int(len(v))}
    res["planners"] = table
    # paired comparisons (10 episodes -> sign counts + Wilcoxon exact)
    def paired(a, b, m, i):
        da = np.array([x[i][m] for x in per_ep[a]]); db = np.array([x[i][m] for x in per_ep[b]])
        d = da - db
        try:
            p = float(stats.wilcoxon(da, db).pvalue)
        except ValueError:
            p = None
        return {"mean_diff": float(d.mean()), "n_improve": int((d < 0).sum()),
                "n": int(len(d)), "wilcoxon_p": p}
    res["paired"] = {
        "BoN-prompted_tracking_deploy": paired("schedule+BoN", "prompted", "tracking_rmse", 0),
        "BoN-prompted_tracking_indep": paired("schedule+BoN", "prompted", "tracking_rmse", 1),
        "BoN-endpoint_start_deploy": paired("schedule+BoN", "endpoint-BoN", "start_err", 0),
        "BoN-endpoint_start_indep": paired("schedule+BoN", "endpoint-BoN", "start_err", 1),
        "global-BoN_tracking_deploy": paired("global-search", "schedule+BoN", "tracking_rmse", 0),
        "global-BoN_tracking_indep": paired("global-search", "schedule+BoN", "tracking_rmse", 1),
        "global-BoN_smooth_deploy": paired("global-search", "schedule+BoN", "smooth_max_step", 0),
        "global-BoN_smooth_indep": paired("global-search", "schedule+BoN", "smooth_max_step", 1),
        "prompted-jump_tracking_deploy": paired("prompted", "jump", "tracking_rmse", 0),
        "prompted-jump_tracking_indep": paired("prompted", "jump", "tracking_rmse", 1)}
    # selection agreement between deploy and indep within pools
    agree = []
    for (e, k), P in df.groupby(["episode", "segment"]):
        P = P.sort_values("candidate")
        r = P.r_k.iloc[0]
        agree.append(int(np.argmin(np.abs(P.arousal.values - r))) ==
                     int(np.argmin(np.abs(P.arousal_indep.values - r))))
    res["selection_agreement_deploy_vs_indep"] = float(np.mean(agree))
    res["within_pool_corr_deploy_indep"] = float(np.mean([
        np.corrcoef(P.arousal, P.arousal_indep)[0, 1]
        for _, P in df.groupby(["episode", "segment"]) if P.arousal.std() > 0 and P.arousal_indep.std() > 0]))
    summary["C_traj"] = res


# ------------------------------------------------------------------ D. CLAP
def analyse_clap(summary):
    m = pd.DataFrame([json.loads(l) for l in open(os.path.join(ROOT, "results", "clap_alignment.jsonl"))])
    s = pd.DataFrame([json.loads(l) for l in open(os.path.join(ROOT, "data", "step3", "clap_shuffled.jsonl"))])
    j = m.merge(s, on="clip_id", suffixes=("_m", "_s"))
    j = j[j.prompt_m != j.prompt_s]   # drop the rare self-matches
    clips = pd.read_csv(os.path.join(ROOT, "results", "audit_table_clips.csv"))
    j = j.merge(clips[["clip_id", "group", "category"]], on="clip_id", how="left")
    d = j.clap_similarity_m - j.clap_similarity_s
    res = {"n_pairs": int(len(j)),
           "matched": boot_ci(j.clap_similarity_m, clusters=j.prompt_m),
           "shuffled": boot_ci(j.clap_similarity_s, clusters=j.prompt_m),
           "diff": boot_ci(d, clusters=j.prompt_m),
           "frac_matched_gt_shuffled": float((d > 0).mean()),
           "wilcoxon_p": float(stats.wilcoxon(j.clap_similarity_m, j.clap_similarity_s).pvalue),
           "cohen_d_paired": float(d.mean() / d.std(ddof=1))}
    # same-group shuffles vs cross-group shuffles (is the residual just 'calm music' semantics?)
    p2g = clips.drop_duplicates("prompt").set_index("prompt").group
    j["shuf_group"] = j.prompt_s.map(p2g)
    same = j[j.shuf_group == j.group]; cross = j[j.shuf_group != j.group]
    res["shuffled_same_group"] = boot_ci(same.clap_similarity_s, clusters=same.prompt_m) if len(same) else None
    res["shuffled_cross_group"] = boot_ci(cross.clap_similarity_s, clusters=cross.prompt_m) if len(cross) else None
    res["by_group"] = {g: {"matched": float(sub.clap_similarity_m.mean()),
                           "shuffled": float(sub.clap_similarity_s.mean()), "n": int(len(sub))}
                       for g, sub in j.groupby("group")}
    summary["D_clap"] = res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="ABCD")
    args = ap.parse_args()
    summary = {}
    if "A" in args.parts: analyse_sweep(summary)
    if "B" in args.parts: analyse_musicgen(summary)
    if "C" in args.parts: analyse_traj(summary)
    if "D" in args.parts: analyse_clap(summary)
    out = os.path.join(R3, "step3_summary.json")
    prev = json.load(open(out)) if os.path.exists(out) else {}
    prev.update(summary)
    json.dump(prev, open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1)[:6000])
