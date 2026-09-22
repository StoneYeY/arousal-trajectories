"""Regenerate paper figures from results/ (no numbers typed by hand).

fig_audit.pdf   : per-category proxy arousal of the 840 SA3 clips, human
                  reference bands, LUFS-normalised means, MusicGen-small
                  replication means (step 3).
fig_planner.pdf : one MIDI episode (rule+BoN vs jump vs schedule) and the
                  best-of-N dual curve (planning proxy vs independent).
"""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results"); P = os.path.join(ROOT, "paper")
plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
BLUE, ORANGE, GREY = "#1f77b4", "#e8622a", "#555555"

CATS = ["sleep", "anxiety_relief", "meditation", "study_calm", "nature_calm",
        "neutral", "high_arousal"]
LAB = ["sleep", "anxiety\nrelief", "medita-\ntion", "study\ncalm", "nature\ncalm",
       "neutral", "high\narousal"]


def fig_audit():
    """Audit figure on an out-of-fold footing: clips scored by the five
    fold models (results/audit_clips_foldmodels.csv), human reference
    bands = out-of-fold predictions of the reference excerpts
    (results/oof_reference.json)."""
    oof = json.load(open(os.path.join(R, "oof_reference.json")))
    clips = pd.read_csv(os.path.join(R, "audit_clips_foldmodels.csv"))
    hc, ht = oof["refs"]["human_calm"]["oof"], oof["refs"]["human_tense"]["oof"]
    cat_sa3 = oof["sa3"]["per_category"]; cat_ln = oof["sa3_lufs"]["per_category"]; cat_mg = oof["musicgen"]["per_category"]

    fig, ax = plt.subplots(figsize=(3.45, 1.5))
    rng = np.random.default_rng(0)
    for i, cat in enumerate(CATS):
        y = clips[clips.category == cat].pred_oofmodels.values
        ax.scatter(i + rng.uniform(-0.22, 0.22, len(y)), y, s=2, c="#b0b0b0", alpha=0.6,
                   linewidths=0, zorder=1)
        col = ORANGE if cat == "high_arousal" else (GREY if cat == "neutral" else BLUE)
        m = cat_sa3[cat]
        ax.errorbar(i, m["mean"], yerr=[[m["mean"] - m["lo"]], [m["hi"] - m["mean"]]],
                    fmt="o", ms=4, c=col, capsize=2, lw=1, zorder=3)
        ax.plot(i + 0.3, cat_ln[cat]["mean"], "o", ms=3, mfc="none", mec=col, mew=0.9, zorder=3)
        ax.plot(i - 0.3, cat_mg[cat]["mean"], "^", ms=3.5, mfc="white", mec=col, mew=0.9, zorder=3)
    ax.axhspan(hc["lo"], hc["hi"], color=BLUE, alpha=0.18, lw=0)
    ax.axhline(hc["mean"], color=BLUE, lw=0.6, alpha=0.6)
    ax.axhspan(ht["lo"], ht["hi"], color=ORANGE, alpha=0.18, lw=0)
    ax.axhline(ht["mean"], color=ORANGE, lw=0.8)
    ax.text(-0.5, ht["mean"] + 0.12, "human high-arousal (out-of-fold)", ha="left", color=ORANGE, fontsize=6.5)
    ax.text(-0.5, hc["mean"] + 0.14, "human calm (out-of-fold)", ha="left", color=BLUE, fontsize=6.5)
    ax.axvline(4.5, color="#cccccc", lw=0.6, ls=":")
    ax.set_xticks(range(7)); ax.set_xticklabels(LAB)
    ax.set_ylabel("proxy arousal (1–9)"); ax.set_ylim(0.6, 7.0)
    ax.set_xlim(-0.6, 6.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", c=BLUE, ms=4, label="SA3 mean, 95% cluster CI"),
         Line2D([], [], marker="o", ls="", mfc="none", mec=BLUE, ms=3, label="−23 LUFS normalised"),
         Line2D([], [], marker="^", ls="", mfc="white", mec=BLUE, ms=3.5, label="MusicGen-small")]
    ax.legend(handles=h, loc="lower left", frameon=False, handletextpad=0.2,
              borderaxespad=0.1, labelspacing=0.15, ncol=2, columnspacing=0.6,
              fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(P, "fig_audit.pdf"))
    print("fig_audit (OOF): human_calm", hc["mean"], "tense", ht["mean"])


def fig_planner():
    eps = []
    for f in ["iso_episodes.jsonl", "iso_episodes_b.jsonl"]:
        p = os.path.join(R, f)
        if os.path.exists(p):
            eps += [json.loads(l) for l in open(p)]
    # pick the episode seed with the largest s0 among those having all planners
    by_seed = {}
    for e in eps:
        by_seed.setdefault(e["seed"], {})[e["planner"]] = e
    seeds = [s for s, d in by_seed.items() if {"rule+BoN", "jump"} <= set(d)]
    seed = min(seeds, key=lambda s: abs(by_seed[s]["rule+BoN"]["s0"] - 7.4))
    ep = by_seed[seed]
    bon = json.load(open(os.path.join(R, "bon_dual_curve.json")))

    fig, (a, b) = plt.subplots(1, 2, figsize=(3.45, 1.7), layout="constrained")
    k = np.arange(1, 6)
    e = ep["rule+BoN"]
    a.plot(k, e["schedule"], "--", c=GREY, lw=1, label="schedule")
    a.plot(k, e["arousal_trajectory"], "o-", c=BLUE, ms=3, lw=1.2, label="rule+BoN")
    a.plot(k, ep["jump"]["arousal_trajectory"], "o-", c=ORANGE, ms=3, lw=1.2, label="jump")
    a.plot([1], [e["s0"]], "*", c="k", ms=7)
    a.text(1.15, e["s0"] - 0.15, r"$s_0$", fontsize=7)
    a.text(2.6, e["schedule"][2] + 0.35, "schedule", color=GREY, fontsize=6.5)
    a.text(2.9, e["arousal_trajectory"][2] - 0.75, "rule+BoN", color=BLUE, fontsize=6.5)
    a.text(1.4, ep["jump"]["arousal_trajectory"][1] - 0.7, "jump", color=ORANGE, fontsize=6.5)
    a.set_xlabel("segment $k$"); a.set_ylabel("proxy arousal"); a.set_xticks(k)
    a.set_ylim(2.5, 8.2)
    Ns = sorted(int(n) for n in bon)
    b.plot(Ns, [bon[str(n)]["deploy"]["mean"] for n in Ns], "o-", c=BLUE, ms=3, lw=1.2)
    b.plot(Ns, [bon[str(n)]["independent"]["mean"] for n in Ns], "o-", c=ORANGE, ms=3, lw=1.2)
    b.set_xscale("log", base=2); b.set_xticks(Ns); b.set_xticklabels(Ns)
    b.set_xlabel("candidates $N$"); b.set_ylabel("tracking error")
    b.text(2.1, bon["8"]["independent"]["mean"] + 0.06, "independent\nestimator", color=ORANGE, fontsize=6.5)
    b.text(3.2, bon["4"]["deploy"]["mean"] - 0.13, "planning\nproxy", color=BLUE, fontsize=6.5, ha="center")
    b.set_ylim(0.75, 1.4)
    for ax in (a, b):
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.savefig(os.path.join(P, "fig_planner.pdf"))
    print("fig_planner: episode s0", e["s0"], "seed", seed)


if __name__ == "__main__":
    fig_audit(); fig_planner()
