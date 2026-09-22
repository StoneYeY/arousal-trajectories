"""ISO-trajectory planning: formalisation + planners + evaluation.

FORMALISATION (paper Sec. 3)
  A session is a sequence of K music segments a_1..a_K, each a parameter
  vector in the interpretable action space of midi_synth (tempo, register,
  density, mode, velocity, program, spread, ambient).
  Given an initial listener-state estimate s_0 (arousal in [1,9], e.g.
  self-report or physiological proxy) and a target s* (calm, e.g. 3.0),
  the ISO principle prescribes a trajectory whose *musical arousal*
  starts near s_0 (matching phase) and moves gradually to s* (leading
  phase).  We define a reference schedule r_k and three verifiable costs:

    match     C_m = |A(a_1) - s_0|              (start where the listener is)
    smooth    C_s = max_k |A(a_{k+1}) - A(a_k)| (no jumps; gradual leading)
    target    C_t = |A(a_K) - s*|               (end calm)

  where A(.) is the verifier's predicted arousal of the rendered audio.
  Total objective J = w_m C_m + w_s C_s + w_t C_t  (+ optional valence
  floor). All three terms are computable from audio alone -> a verifiable
  reward for best-of-N and, later, RL.

  NOTE ON CLAIMS: this evaluates whether a planner can REALISE a
  prescribed ISO schedule in rendered audio (trajectory fidelity), not
  whether ISO improves listener outcomes; the latter needs human/physio
  data and is future work.

PLANNERS
  rule         : linear interpolation in parameter space from a preset
                 matched to s_0 down to the calm preset (classic ISO
                 heuristic, no verifier access).
  rule+BoN     : same schedule, but each segment samples N candidates
                 (param jitter + seed) and keeps the one whose verifier
                 arousal is closest to the schedule point r_k.
  jump         : control/ablation - all segments at the calm preset
                 (ignores matching phase; tests whether C_m/C_s matter).
  endpoint-BoN : ablation - candidates selected ONLY to minimise final
                 arousal (no schedule tracking). Answers: does the ISO-
                 shaped reward matter, or does BoN just pick the calmest
                 endpoint?
  random       : floor baseline - uniformly random anchor preset per
                 segment, no verifier access.

NAMING: this is a SEQUENTIAL PLANNER (deterministic schedule + candidate
selection), not an agent — no learned policy, no observation loop. The
paper uses "planner" throughout.

EVALUATION: costs are reported under BOTH the planning verifier (deploy)
and an independent evaluator (different model family + feature set,
never used in any reward) to avoid purely circular evaluation.
"""
import os, sys, json, argparse, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from midi_synth import render_trajectory, PRESETS
from score import score_array
import librosa

VDIR_DEFAULT = "/home/claude/amg/results/verifier"

# anchor presets ordered on the arousal axis
ANCHORS = ["very_calm", "calm", "neutral", "tense", "agitated"]
NUMERIC = ["tempo_bpm", "register", "density", "velocity", "voicing_spread"]


def lerp_params(p_lo, p_hi, alpha):
    """Interpolate between two presets. alpha=0 -> lo, 1 -> hi."""
    out = {}
    for k in NUMERIC:
        out[k] = float(p_lo[k] + alpha * (p_hi[k] - p_lo[k]))
    out["register"] = int(round(out["register"]))
    out["velocity"] = int(round(out["velocity"]))
    out["mode"] = p_hi["mode"] if alpha > 0.5 else p_lo["mode"]
    out["program"] = p_hi["program"] if alpha > 0.5 else p_lo["program"]
    out["ambient"] = (p_hi if alpha > 0.5 else p_lo).get("ambient", False)
    return out


def schedule(s0, s_star, K):
    """Reference arousal schedule: cosine ease from s0 to s*."""
    ks = np.arange(K)
    return s_star + (s0 - s_star) * 0.5 * (1 + np.cos(np.pi * ks / (K - 1)))


def preset_for_level(level_01):
    """Map normalised arousal level [0,1] onto the anchor-preset axis."""
    x = level_01 * (len(ANCHORS) - 1)
    i = int(np.clip(np.floor(x), 0, len(ANCHORS) - 2))
    return lerp_params(PRESETS[ANCHORS[i]], PRESETS[ANCHORS[i + 1]], x - i)


def plan_rule(s0, s_star, K):
    sch = schedule(s0, s_star, K)
    # normalise DEAM 1-9 arousal to [0,1] on the anchor axis
    return [preset_for_level((a - 1) / 8.0) for a in sch], sch


def render_and_score_segment(params, dur, seed, vdir, sr=22050):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav = f.name
    try:
        render_trajectory([(params, dur)], wav, seed=seed, sr=sr)
        y, _ = librosa.load(wav, sr=sr)
        sc = score_array(y, sr, vdir)
        return y, sc
    finally:
        if os.path.exists(wav):
            os.unlink(wav)


def jitter(params, rng, scale=0.12):
    q = dict(params)
    q["tempo_bpm"] = float(np.clip(q["tempo_bpm"] * (1 + rng.uniform(-scale, scale)), 40, 180))
    q["velocity"] = int(np.clip(q["velocity"] + rng.integers(-10, 11), 20, 110))
    q["density"] = float(np.clip(q["density"] * (1 + rng.uniform(-scale, scale)), 0.25, 4))
    q["register"] = int(np.clip(q["register"] + rng.integers(-4, 5), 36, 84))
    return q


def costs_from_traj(A, sch, s0, s_star):
    K = len(A)
    c = {"match": float(abs(A[0] - s0)),
         "smooth": float(np.max(np.abs(np.diff(A)))) if K > 1 else 0.0,
         "target": float(abs(A[-1] - s_star)),
         "schedule_rmse": float(np.sqrt(np.mean((A - sch) ** 2)))}
    c["J"] = c["match"] + c["smooth"] + c["target"]
    return c


def run_episode(s0, s_star=3.0, K=5, seg_dur=12.0, planner="rule",
                n_candidates=1, vdir=VDIR_DEFAULT, seed=0,
                score_independent=True):
    rng = np.random.default_rng(seed)
    if planner == "jump":
        plan = [preset_for_level((s_star - 1) / 8.0)] * K
        sch = np.full(K, s_star)
    elif planner == "random":
        plan = [dict(PRESETS[ANCHORS[int(rng.integers(len(ANCHORS)))]])
                for _ in range(K)]
        _, sch = plan_rule(s0, s_star, K)  # judged against the ISO schedule
    else:
        plan, sch = plan_rule(s0, s_star, K)

    seg_audio, seg_arousal = [], []
    for k, base in enumerate(plan):
        best = None
        n = n_candidates if "BoN" in planner else 1
        for c in range(n):
            cand = base if c == 0 else jitter(base, rng)
            y, sc = render_and_score_segment(cand, seg_dur,
                                             seed=int(rng.integers(1e6)),
                                             vdir=vdir)
            if planner == "endpoint-BoN":
                gap = sc["arousal_pred"]           # greedy: just be calm
            else:
                gap = abs(sc["arousal_pred"] - sch[k])  # ISO: track schedule
            if best is None or gap < best[0]:
                best = (gap, y, sc)
        seg_audio.append(best[1])
        seg_arousal.append(best[2]["arousal_pred"])

    A = np.array(seg_arousal)
    out = {"planner": planner, "s0": s0, "s_star": s_star, "K": K,
           "n_candidates": n_candidates, "seed": seed,
           "arousal_trajectory": [float(a) for a in A],
           "schedule": [float(x) for x in sch],
           "costs": costs_from_traj(A, sch, s0, s_star)}

    # anti-circularity: re-score the SELECTED segments with the
    # independent evaluator (never used in selection above)
    if score_independent:
        try:
            Ai = np.array([score_array(y, 22050, vdir, variant="independent")
                           ["arousal_pred"] for y in seg_audio])
            out["arousal_trajectory_independent"] = [float(a) for a in Ai]
            out["costs_independent"] = costs_from_traj(Ai, sch, s0, s_star)
        except FileNotFoundError:
            pass  # independent verifier not trained yet
    return out


PLANNER_SET = [("rule", 1), ("rule+BoN", None), ("jump", 1),
               ("endpoint-BoN", None), ("random", 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vdir", default=VDIR_DEFAULT)
    ap.add_argument("--out", default="/home/claude/amg/results/iso_episodes.jsonl")
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--n_candidates", type=int, default=4)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--seg_dur", type=float, default=12.0)
    ap.add_argument("--ep_start", type=int, default=0)
    ap.add_argument("--ep_end", type=int, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(7)
    s0s = rng.uniform(5.5, 8.5, size=args.n_episodes)  # anxious starts
    ep_end = args.ep_end if args.ep_end is not None else args.n_episodes

    # resume: skip (episode, planner) pairs already in the output file
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                done.add((r["seed"], r["planner"]))
            except (json.JSONDecodeError, KeyError):
                pass

    with open(args.out, "a") as f:
        for i in range(args.ep_start, ep_end):
            s0 = float(s0s[i])
            for planner, nc in PLANNER_SET:
                nc = nc or args.n_candidates
                if (1000 + i, planner) in done:
                    continue
                ep = run_episode(s0, 3.0, args.K, args.seg_dur,
                                 planner, nc, args.vdir, seed=1000 + i)
                f.write(json.dumps(ep) + "\n")
                f.flush()
                print(f"ep{i} {planner:12s} s0={s0:.2f} "
                      f"traj={[f'{a:.2f}' for a in ep['arousal_trajectory']]} "
                      f"J={ep['costs']['J']:.3f}", flush=True)


if __name__ == "__main__":
    main()
