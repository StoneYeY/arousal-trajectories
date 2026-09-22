"""Listening-study analysis (perceptual validation; not a clinical study).

Inputs  (listening_study/data/*.csv, as returned by the experimenter):
  ratings_template-24.csv        participant,item,question(1-4),rating_1to7
  calib_ratings_template-24.csv  participant,item,arousal_1to9
  participants_template-24.csv   participant,age_band,music_background_years,headphones_model
Keys: listening_study/key.csv (t/c items), calib_domains/key_domains.csv (g/d).

Outputs results/listening_summary.json (+ per-item csv). Every number in
Sec. 4.3 of the paper comes from here.

Analyses
  A  cleaning + descriptives (out-of-range ratings are excluded, documented)
  B  per-question crossed random-effects model
       rating ~ condition + (1|participant) + (1|item)      [REML, statsmodels]
     3 pairwise contrasts per question, Holm-corrected within question
  C  participant-level paired effects (mean per condition; Wilcoxon, Cohen dz)
  D  proxy validity by domain: Spearman rho(proxy, perceived) for
       MIDI (c01-c12), generated SA3 (g01-g12), real DEAM (d01-d08, proxy
       re-fitted WITHOUT those 8 songs so the score is out-of-sample)
     + rho(DEAM crowd label, our raters) for the real-music items
  E  trajectory-level: proxy metrics of each stimulus (5-s segment scores)
     vs human item means
  F  human ratings at N=1 (rule) vs N=4 (rule+BoN): the human side of the
     best-of-N over-optimisation finding
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LS = os.path.join(ROOT, "listening_study")
RES = os.path.join(ROOT, "results")
VDIR = os.path.join(RES, "verifier")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

QNAME = {1: "endpoint_calm", 2: "smoothness", 3: "start_match", 4: "overall_traj"}
CONDS = ["jump", "rule", "rule+BoN"]
CONTRASTS = [("rule", "jump"), ("rule+BoN", "jump"), ("rule+BoN", "rule")]


def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1.0, running)
    return adj.tolist()


def load():
    r = pd.read_csv(os.path.join(LS, "data", "ratings_template-24.csv"), keep_default_na=False)
    c = pd.read_csv(os.path.join(LS, "data", "calib_ratings_template-24.csv"), keep_default_na=False)
    p = pd.read_csv(os.path.join(LS, "data", "participants_template-24.csv"), keep_default_na=False)
    key = pd.read_csv(os.path.join(LS, "key.csv"))
    kd = pd.read_csv(os.path.join(LS, "calib_domains", "key_domains.csv"))
    return r, c, p, key, kd


# ------------------------------------------------------------------ A
def clean(r, c):
    raw = r.copy()
    r["rating"] = pd.to_numeric(r.rating_1to7, errors="coerce")
    n_blank = int(r.rating.isna().sum())
    oor = r[(r.rating.notna()) & ((r.rating < 1) | (r.rating > 7))]
    r.loc[oor.index, "rating"] = np.nan
    r["question"] = r.question.map(QNAME)
    c["rating"] = pd.to_numeric(c.arousal_1to9, errors="coerce")
    c_oor = int(((c.rating < 1) | (c.rating > 9)).sum())
    info = {"n_rows_returned": int(len(raw)), "n_blank": n_blank,
            "n_out_of_range_excluded": int(len(oor)),
            "out_of_range_rows": oor[["participant", "item", "question", "rating_1to7"]]
                .assign(question=lambda d: d.question.map(QNAME)).to_dict("records"),
            "n_valid": int(r.rating.notna().sum()),
            "calib_rows": int(len(c)), "calib_out_of_range": c_oor}
    return r.dropna(subset=["rating"]), c.dropna(subset=["rating"]), info


# ------------------------------------------------------------------ B
def mixed_models(r):
    import statsmodels.formula.api as smf
    out = {}
    for q in QNAME.values():
        d = r[r.question == q].copy()
        d["one"] = 1
        res_q = {"n": int(len(d)), "raw_means": {}, "contrasts": {}}
        for cnd in CONDS:
            x = d[d.condition == cnd].rating
            res_q["raw_means"][cnd] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "n": int(len(x))}
        cons = []
        for a, b in CONTRASTS:
            # crossed random effects: single group, participant + item variance components
            m = smf.mixedlm(f"rating ~ C(condition, Treatment('{b}'))", d, groups=d["one"],
                            vc_formula={"participant": "0 + C(participant)", "item": "0 + C(item)"})
            f = m.fit(reml=True, method="lbfgs", maxiter=500)
            name = f"C(condition, Treatment('{b}'))[T.{a}]"
            est, se, p = float(f.params[name]), float(f.bse[name]), float(f.pvalues[name])
            cons.append({"contrast": f"{a} - {b}", "estimate": est, "se": se,
                         "ci95": [est - 1.96 * se, est + 1.96 * se], "z": est / se, "p": p})
            if (a, b) == ("rule", "jump"):
                vnames = list(m.exog_vc.names)
                vc = {n: float(v) for n, v in zip(vnames, np.asarray(f.vcomp).ravel())}
                res_q["variance"] = {"participant": vc.get("participant"), "item": vc.get("item"),
                                     "residual": float(f.scale)}
                res_q["model_means"] = {
                    "jump": float(f.params["Intercept"]),
                    "rule": float(f.params["Intercept"] + f.params["C(condition, Treatment('jump'))[T.rule]"]),
                    "rule+BoN": float(f.params["Intercept"] + f.params["C(condition, Treatment('jump'))[T.rule+BoN]"])}
        adj = holm([c_["p"] for c_ in cons])
        for c_, pa in zip(cons, adj):
            c_["p_holm"] = pa
            res_q["contrasts"][c_["contrast"]] = c_
        out[q] = res_q
    return out


# ------------------------------------------------------------------ C
def participant_paired(r):
    out = {}
    pm = r.groupby(["participant", "question", "condition"]).rating.mean().unstack("condition")
    for q in QNAME.values():
        t = pm.xs(q, level="question").dropna()
        res = {"n_participants": int(len(t))}
        cons = []
        for a, b in CONTRASTS:
            dlt = t[a] - t[b]
            try:
                pw = float(stats.wilcoxon(t[a], t[b]).pvalue)
            except ValueError:
                pw = 1.0
            cons.append({"contrast": f"{a} - {b}", "mean_diff": float(dlt.mean()),
                         "dz": float(dlt.mean() / dlt.std(ddof=1)) if dlt.std(ddof=1) > 0 else None,
                         "n_positive": int((dlt > 0).sum()), "n_negative": int((dlt < 0).sum()),
                         "wilcoxon_p": pw})
        for c_, pa in zip(cons, holm([c_["wilcoxon_p"] for c_ in cons])):
            c_["p_holm"] = pa
        res["contrasts"] = {c_["contrast"]: c_ for c_ in cons}
        out[q] = res
    return out


# ------------------------------------------------------------------ D
def deam_out_of_sample_proxy(kd):
    """Refit the deploy recipe without the 8 DEAM calibration songs and
    score the actual 20-s clips the raters heard."""
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    from features import extract_features, extract_egemaps
    ids = json.load(open(os.path.join(LS, "calib_domains", "d_item_song_ids.json")))
    pack = joblib.load(os.path.join(VDIR, "verifier_arousal_mean_deploy.joblib"))
    cols = pack["cols"]
    df = pd.read_csv(os.path.join(RES, "deam_features.csv"))
    excl = {v["song_id"] for v in ids.values()}
    tr = df[~df.song_id.isin(excl)].dropna(subset=["arousal_mean"])
    m = HistGradientBoostingRegressor(random_state=42, max_iter=300)
    m.fit(tr[cols].values, tr["arousal_mean"].values)
    out = {}
    for item, v in ids.items():
        path = os.path.join(LS, "calib_domains", f"{item}.flac")
        f = extract_features(path); f.update(extract_egemaps(path))
        X = np.array([[f.get(c, np.nan) for c in cols]])
        out[item] = {"song_id": v["song_id"], "proxy_oos_20s": float(m.predict(X)[0]),
                     "proxy_insample_full": float(pack["model"].predict(X)[0]),
                     "true_arousal": v["true_arousal"]}
    return out


def proxy_validity(c, key, kd):
    hum = c.groupby("item").rating.agg(["mean", "std", "count"]).rename(columns={"mean": "human"})
    tab = []
    for _, k in key[key.kind == "calibration"].iterrows():
        tab.append({"item": k["item"], "domain": "midi", "proxy": k["verifier_arousal_traj"],
                    "intended": k["s0"], "true": None})
    for _, k in kd.iterrows():
        tab.append({"item": k["item"], "domain": "generated" if k["domain"].startswith("generated") else "real",
                    "proxy": k["proxy_arousal"] if pd.notna(k["proxy_arousal"]) else None,
                    "intended": None, "true": k["true_arousal"] if pd.notna(k["true_arousal"]) else None})
    tab = pd.DataFrame(tab).set_index("item").join(hum)
    oos = deam_out_of_sample_proxy(kd)
    for item, v in oos.items():
        tab.loc[item, "proxy"] = v["proxy_oos_20s"]
        tab.loc[item, "proxy_insample"] = v["proxy_insample_full"]
    res = {"per_item": tab.reset_index().to_dict("records"), "domains": {}}

    def rho(x, y):
        m = pd.notna(x) & pd.notna(y)
        if m.sum() < 3:
            return None
        r_, p_ = stats.spearmanr(x[m], y[m]); pr = stats.pearsonr(x[m], y[m])
        return {"spearman": float(r_), "p": float(p_), "pearson": float(pr[0]), "n_items": int(m.sum())}
    for dom in ["midi", "generated", "real"]:
        t = tab[tab.domain == dom]
        d = {"rho_proxy_human": rho(t.proxy, t.human),
             "raters_per_item_mean": float(t["count"].mean()),
             "human_range": [float(t.human.min()), float(t.human.max())],
             "proxy_range": [float(t.proxy.min()), float(t.proxy.max())]}
        if dom == "midi":
            d["rho_intended_human"] = rho(t.intended, t.human)
            d["rho_intended_proxy"] = rho(t.intended, t.proxy)
        if dom == "real":
            d["rho_deamlabel_human"] = rho(t["true"], t.human)
            d["rho_proxy_insample_human"] = rho(t.proxy_insample, t.human)
            d["rho_proxy_oos_deamlabel"] = rho(t.proxy, t["true"])
        res["domains"][dom] = d
    res["pooled_all_items"] = rho(tab.proxy, tab.human)
    # rater-level: each rater's Spearman over their own items (all domains pooled, n=16)
    per_rater = []
    for pid, g in c.groupby("participant"):
        g = g.join(tab[["proxy"]], on="item")
        if g.proxy.notna().sum() >= 5:
            per_rater.append(float(stats.spearmanr(g.proxy, g.rating).correlation))
    res["per_rater_rho"] = {"median": float(np.median(per_rater)), "iqr": [float(np.percentile(per_rater, 25)), float(np.percentile(per_rater, 75))],
                            "n_raters": len(per_rater), "frac_positive": float(np.mean(np.array(per_rater) > 0))}
    # inter-rater reliability of the calibration ratings (ICC-like: item variance share)
    grand = c.rating.mean(); im = c.groupby("item").rating.transform("mean")
    res["calib_item_variance_share"] = float(((im - grand) ** 2).mean() / ((c.rating - grand) ** 2).mean())
    return res


# ------------------------------------------------------------------ E
def stimulus_proxy_trajectories(key):
    """Score each 5-s segment of every trajectory stimulus with the deploy
    verifier -> proxy trajectory -> Eq.(track) metrics."""
    import librosa
    from score import score_array
    cache = os.path.join(RES, "listening_stimulus_proxy.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    out = {}
    for _, k in key[key.kind == "trajectory"].iterrows():
        y, sr = librosa.load(os.path.join(LS, "stimuli_flac", f"{k['item']}.flac"), sr=22050, mono=True)
        K = 5; seg = len(y) // K
        A = [score_array(y[i * seg:(i + 1) * seg], sr, VDIR)["arousal_pred"] for i in range(K)]
        s0 = float(k["s0"]); ks = np.arange(K)
        sch = 3.0 + (s0 - 3.0) * 0.5 * (1 + np.cos(np.pi * ks / (K - 1)))
        A = np.array(A)
        out[k["item"]] = {"condition": k["condition"], "s0": s0, "proxy_traj": A.tolist(),
                          "schedule": sch.tolist(),
                          "track_rmse": float(np.sqrt(np.mean((A - sch) ** 2))),
                          "start_err": float(abs(A[0] - s0)),
                          "max_step": float(np.max(np.abs(np.diff(A)))),
                          "endpoint_err": float(abs(A[-1] - 3.0)),
                          "endpoint_arousal": float(A[-1])}
    json.dump(out, open(cache, "w"), indent=1)
    return out


def trajectory_level(r, key):
    px = stimulus_proxy_trajectories(key)
    hum = r.groupby(["item", "question"]).rating.mean().unstack("question")
    t = pd.DataFrame(px).T.join(hum)
    t = t.astype({c: float for c in ["track_rmse", "start_err", "max_step", "endpoint_err", "endpoint_arousal"]})
    res = {"per_item": t.reset_index().rename(columns={"index": "item"}).to_dict("records"), "correlations": {}}
    pairs = [("endpoint_calm", "endpoint_arousal"), ("smoothness", "max_step"),
             ("start_match", "start_err"), ("overall_traj", "track_rmse")]
    for hq, pm in pairs:
        r_, p_ = stats.spearmanr(t[hq], t[pm])
        res["correlations"][f"{hq}~{pm}"] = {"spearman": float(r_), "p": float(p_), "n_items": int(len(t))}
    # proxy metric means per condition for the stimuli actually rated
    res["proxy_by_condition"] = {c_: {m: float(t[t.condition == c_][m].mean()) for m in ["track_rmse", "start_err", "max_step", "endpoint_err"]} for c_ in CONDS}
    return res


# ------------------------------------------------------------------ F
def bon_human(r):
    pm = r.groupby(["participant", "question", "condition"]).rating.mean().unstack("condition")
    out = {}
    for q in QNAME.values():
        t = pm.xs(q, level="question").dropna(subset=["rule", "rule+BoN"])
        d = t["rule+BoN"] - t["rule"]
        out[q] = {"rule_N1": float(t["rule"].mean()), "bon_N4": float(t["rule+BoN"].mean()),
                  "diff": float(d.mean()), "ci95_diff": [float(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d))),
                                                        float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(len(d)))],
                  "wilcoxon_p": float(stats.wilcoxon(t["rule+BoN"], t["rule"]).pvalue) if (d != 0).any() else 1.0,
                  "n": int(len(d))}
    return out


def main():
    r, c, p, key, kd = load()
    r, c, info = clean(r, c)
    r = r.merge(key[["item", "condition", "s0"]], on="item")
    summary = {"A_data": info}
    summary["A_data"].update({
        "n_participants": int(p.participant.nunique()),
        "age": {"min": int(p.age_band.min()), "max": int(p.age_band.max()), "mean": float(p.age_band.mean())},
        "music_training_years": p.music_background_years.value_counts().sort_index().to_dict(),
        "headphones_reported": int((p.headphones_model != "N/A").sum()),
        "items_per_participant": {"trajectories": 9, "calibration": 16}})
    summary["B_mixed"] = mixed_models(r)
    summary["C_paired"] = participant_paired(r)
    summary["D_proxy_validity"] = proxy_validity(c, key, kd)
    summary["E_trajectory_level"] = trajectory_level(r, key)
    summary["F_bon_human"] = bon_human(r)
    # sensitivity: clip the three '8' ratings to 7 instead of excluding
    r2, c2, _ = clean(*load()[:2])
    raw = pd.read_csv(os.path.join(LS, "data", "ratings_template-24.csv"), keep_default_na=False)
    raw["rating"] = pd.to_numeric(raw.rating_1to7, errors="coerce").clip(upper=7)
    raw["question"] = raw.question.map(QNAME)
    raw = raw.dropna(subset=["rating"]).merge(key[["item", "condition", "s0"]], on="item")
    summary["G_sensitivity_clip8to7"] = {q: {k: v["estimate"] for k, v in mm["contrasts"].items()}
                                        for q, mm in mixed_models(raw).items()}
    json.dump(summary, open(os.path.join(RES, "listening_summary.json"), "w"), indent=1)
    # compact printout
    print(json.dumps(summary["A_data"], indent=1))
    for q, m in summary["B_mixed"].items():
        print(f"\n== {q}: raw means", {k: round(v["mean"], 2) for k, v in m["raw_means"].items()},
              "var", {k: (round(v, 3) if v is not None else None) for k, v in m["variance"].items()})
        for k, v in m["contrasts"].items():
            print(f"   {k:18s} est {v['estimate']:+.2f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}] p={v['p']:.2e} holm={v['p_holm']:.2e}")
        for k, v in summary["C_paired"][q]["contrasts"].items():
            print(f"   paired {k:12s} d={v['mean_diff']:+.2f} dz={v['dz']:+.2f} +{v['n_positive']}/-{v['n_negative']} p={v['wilcoxon_p']:.2e} holm={v['p_holm']:.2e}")
    print("\n== proxy validity"); print(json.dumps(summary["D_proxy_validity"]["domains"], indent=1)); print("pooled", summary["D_proxy_validity"]["pooled_all_items"], "per-rater", summary["D_proxy_validity"]["per_rater_rho"], "item var share", round(summary["D_proxy_validity"]["calib_item_variance_share"], 3))
    print("\n== trajectory-level"); print(json.dumps(summary["E_trajectory_level"]["correlations"], indent=1)); print(summary["E_trajectory_level"]["proxy_by_condition"])
    print("\n== BoN human"); print(json.dumps(summary["F_bon_human"], indent=1))
    print("\n== sensitivity (clip 8->7)"); print(json.dumps(summary["G_sensitivity_clip8to7"], indent=1))


if __name__ == "__main__":
    main()
