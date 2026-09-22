"""Train the arousal/valence verifier on DEAM features.

Design decisions (stated in paper):
- Split at SONG level (each row is one song; DEAM static labels are
  crowd-averaged per song). 5-fold CV + one held-out test split (20%),
  fixed seed. No generated audio is ever seen in training.
- Models: Ridge (linear, interpretable) and HistGradientBoosting
  (nonlinear). Report Pearson r, R^2, CCC on held-out data.
- Model selection uses CV ONLY; the held-out test set is touched exactly
  once, by the CV-selected configuration (no selection-on-test).
- Interpretability: permutation importance of the hand-crafted
  (literature-grounded) features; used for the CONVERGENT-validity check
  against published acoustic-arousal findings (literature consistency —
  NOT external validity; that requires an external dataset / listeners).
- The VERIFIER SCORE used downstream = predicted arousal (1-9, lower is
  calmer) and predicted valence. We deliberately do NOT call this an
  'anxiety score': it is an arousal/valence proxy validated on listener
  annotations.

Usage:
  python3 train_verifier.py --features amg/results/deam_features.csv \
      --outdir amg/results/verifier
"""
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
import joblib

HAND = ["tempo_bpm", "onset_density", "pulse_clarity", "rms_mean", "rms_std",
        "rms_db_range", "attack_slope_mean", "spectral_centroid_mean",
        "spectral_centroid_std", "spectral_rolloff_mean",
        "spectral_flatness_mean", "spectral_flux_mean", "spectral_flux_std",
        "roughness", "mode_majorness", "zcr_mean", "f0_median"]


def ccc(y, p):
    """Concordance correlation coefficient."""
    my, mp = y.mean(), p.mean()
    vy, vp = y.var(), p.var()
    cov = ((y - my) * (p - mp)).mean()
    return float(2 * cov / (vy + vp + (my - mp) ** 2 + 1e-12))


def evaluate(model, X, y):
    p = model.predict(X)
    return {"pearson_r": float(np.corrcoef(y, p)[0, 1]),
            "r2": float(r2_score(y, p)), "ccc": ccc(y, p),
            "rmse": float(np.sqrt(np.mean((y - p) ** 2)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.features)
    eg_cols = [c for c in df.columns if c.startswith("eg_")]
    feature_sets = {"hand": HAND,
                    "egemaps": eg_cols,
                    "hand+egemaps": HAND + eg_cols}
    targets = ["arousal_mean", "valence_mean"]
    df = df.dropna(subset=[c for c in HAND + targets if c in df.columns])
    print(f"{len(df)} songs usable")

    report = {"n_songs": int(len(df)), "seed": args.seed, "results": {},
              "protocol": "model selection by 5-fold CV on train only; "
                          "held-out test evaluated once for the CV-winner"}
    tr, te = train_test_split(df, test_size=0.2, random_state=args.seed)

    def make_models():
        return {"ridge": Pipeline([("sc", StandardScaler()),
                                    ("m", Ridge(alpha=1.0))]),
                "hgb": HistGradientBoostingRegressor(
                    random_state=args.seed, max_iter=300)}

    for tgt in targets:
        report["results"][tgt] = {}
        cv_scores = {}
        for fs_name, cols in feature_sets.items():
            cols = [c for c in cols if c in df.columns]
            if not cols:
                continue
            for mname, model in make_models().items():
                kf = KFold(5, shuffle=True, random_state=args.seed)
                cv = []
                Xtr_all = tr[cols].values
                ytr_all = tr[tgt].values
                for itr, iva in kf.split(Xtr_all):
                    model.fit(Xtr_all[itr], ytr_all[itr])
                    cv.append(evaluate(model, Xtr_all[iva], ytr_all[iva]))
                key = f"{fs_name}/{mname}"
                cv_scores[key] = (np.mean([c["pearson_r"] for c in cv]), cols, mname)
                report["results"][tgt][key] = {
                    "cv_pearson_r_mean": float(np.mean([c["pearson_r"] for c in cv])),
                    "cv_pearson_r_std": float(np.std([c["pearson_r"] for c in cv])),
                    "n_features": len(cols)}
                print(f"{tgt} {key}: CV r={cv_scores[key][0]:.3f}", flush=True)

        # CV winner -> single touch of the held-out test set
        win_key = max(cv_scores, key=lambda k: cv_scores[k][0])
        _, win_cols, win_m = cv_scores[win_key]
        winner = make_models()[win_m]
        winner.fit(tr[win_cols].values, tr[tgt].values)
        test = evaluate(winner, te[win_cols].values, te[tgt].values)
        report["results"][tgt]["cv_selected"] = win_key
        report["results"][tgt]["test_once"] = test
        print(f"{tgt} CV-selected={win_key} -> TEST(once): "
              f"r={test['pearson_r']:.3f} R2={test['r2']:.3f} "
              f"ccc={test['ccc']:.3f}", flush=True)

        # interpretability on the hand-crafted set with the better model
        cols = [c for c in HAND if c in df.columns]
        best = HistGradientBoostingRegressor(random_state=args.seed, max_iter=300)
        best.fit(tr[cols].values, tr[tgt].values)
        pi = permutation_importance(best, te[cols].values, te[tgt].values,
                                    n_repeats=20, random_state=args.seed)
        imp = sorted(zip(cols, pi.importances_mean, pi.importances_std),
                     key=lambda x: -x[1])
        report["results"][tgt]["permutation_importance_hand_hgb"] = [
            {"feature": f, "mean": float(m), "std": float(s)} for f, m, s in imp]
        joblib.dump(best, os.path.join(args.outdir, f"verifier_{tgt}_hgb_hand.joblib"))

        # deployable verifier: hand+egemaps HGB trained on ALL data
        cols_full = [c for c in HAND + eg_cols if c in df.columns]
        deploy = HistGradientBoostingRegressor(random_state=args.seed, max_iter=300)
        deploy.fit(df[cols_full].values, df[tgt].values)
        joblib.dump({"model": deploy, "cols": cols_full},
                    os.path.join(args.outdir, f"verifier_{tgt}_deploy.joblib"))

        # INDEPENDENT evaluator (anti-circularity judge for the planner):
        # deliberately different model family (linear Ridge, not boosted
        # trees) AND different feature set (17 hand features only, not the
        # 105-dim deploy set). Shares only the training corpus — that
        # remaining dependence is stated in the paper and later removed by
        # the PMEmo external check and the listener study. Never used
        # inside any planner reward.
        cols_hand = [c for c in HAND if c in df.columns]
        indep = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
        indep.fit(df[cols_hand].values, df[tgt].values)
        joblib.dump({"model": indep, "cols": cols_hand},
                    os.path.join(args.outdir, f"verifier_{tgt}_independent.joblib"))
        # cross-agreement on held-out (sanity: correlated but not identical)
        pa = deploy.fit(tr[cols_full].values, tr[tgt].values).predict(te[cols_full].values)
        pb = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]) \
            .fit(tr[cols_hand].values, tr[tgt].values).predict(te[cols_hand].values)
        report["results"][tgt]["deploy_vs_independent_heldout_r"] = \
            float(np.corrcoef(pa, pb)[0, 1])

    json.dump(report, open(os.path.join(args.outdir, "report.json"), "w"), indent=1)
    print("wrote", os.path.join(args.outdir, "report.json"))


if __name__ == "__main__":
    main()
