"""Build the listening-study stimulus package.

- 8 s0 conditions (episodes 0-7 of the ISO experiment seeds), each
  rendered as full trajectories under 3 conditions:
  direct-jump / rule / rule+BoN (N=4). K=5 segments x 5 s = ~25 s/clip.
  Filenames are anonymised (t01.wav ... t24.wav) - the condition map
  lives ONLY in key.csv so raters stay blind.
- 12 single 8-s segments spanning the arousal axis (perceived-arousal
  calibration items, c01..c12.wav) with verifier scores recorded for the
  rho(proxy, perceived) analysis.
- ratings_template.csv: one row per (participant, item, question).
- randomization.csv: per-participant Latin-square-ish item order.

Output: listening_study/stimuli/ + zip.
"""
import json, os, sys, zipfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iso_agent import plan_rule, preset_for_level, jitter, PRESETS, ANCHORS
from midi_synth import render_trajectory
from score import score_file

BASE = "/home/claude/amg/listening_study"
STIM = os.path.join(BASE, "stimuli")
VDIR = "/home/claude/amg/results/verifier"
SEG_DUR = 5.0
K = 5


def build_plan(s0, cond, rng):
    if cond == "jump":
        return [preset_for_level((3.0 - 1) / 8.0)] * K, None
    plan, sch = plan_rule(s0, 3.0, K)
    return plan, sch


def main():
    os.makedirs(STIM, exist_ok=True)
    rng_master = np.random.default_rng(7)
    s0s = rng_master.uniform(5.5, 8.5, size=20)[:8]

    # trajectory stimuli
    items = []  # (item_id, kind, cond, s0, wav)
    idx = 1
    key_rows = ["item,kind,condition,s0,verifier_arousal_traj"]
    for i, s0 in enumerate(s0s):
        for cond in ["jump", "rule", "rule+BoN"]:
            rng = np.random.default_rng(5000 + i)
            plan, sch = build_plan(float(s0), cond, rng)
            segs = []
            for k, base in enumerate(plan):
                if cond == "rule+BoN":
                    # N=4 selection against the schedule, same as main exp
                    best = None
                    for c in range(4):
                        cand = base if c == 0 else jitter(base, rng)
                        wav_tmp = os.path.join(STIM, "_tmp.wav")
                        render_trajectory([(cand, SEG_DUR)], wav_tmp,
                                          seed=int(rng.integers(1e6)))
                        sc = score_file(wav_tmp, VDIR)
                        gap = abs(sc["arousal_pred"] - sch[k])
                        if best is None or gap < best[0]:
                            best = (gap, cand)
                    segs.append((best[1], SEG_DUR))
                else:
                    segs.append((base, SEG_DUR))
            out = os.path.join(STIM, f"t{idx:02d}.wav")
            render_trajectory(segs, out, seed=5000 + i)
            sc = score_file(out, VDIR)
            key_rows.append(f"t{idx:02d},trajectory,{cond},{s0:.2f},"
                            f"{sc['arousal_pred']:.2f}")
            idx += 1
            print(f"t{idx-1:02d} {cond} s0={s0:.2f}", flush=True)
    if os.path.exists(os.path.join(STIM, "_tmp.wav")):
        os.unlink(os.path.join(STIM, "_tmp.wav"))

    # calibration single segments across the arousal axis
    rng = np.random.default_rng(99)
    for j in range(12):
        level = j / 11.0
        p = preset_for_level(level)
        out = os.path.join(STIM, f"c{j+1:02d}.wav")
        render_trajectory([(p, 8.0)], out, seed=9000 + j)
        sc = score_file(out, VDIR)
        key_rows.append(f"c{j+1:02d},calibration,single,{1+8*level:.2f},"
                        f"{sc['arousal_pred']:.2f}")
        print(f"c{j+1:02d} level={level:.2f}", flush=True)

    open(os.path.join(BASE, "key.csv"), "w").write("\n".join(key_rows) + "\n")

    # ratings template + per-participant randomization (24 participants)
    qs = ["endpoint_calm", "smoothness", "start_match", "overall_traj"]
    with open(os.path.join(BASE, "ratings_template.csv"), "w") as f:
        f.write("participant,item,question,rating_1to7\n")
    rand_rows = ["participant,order"]
    all_traj = [f"t{i:02d}" for i in range(1, 25)]
    rng = np.random.default_rng(123)
    for p in range(1, 25):
        # each participant: 9 trajectories (3 per condition, balanced) + 4 calib
        order = []
        for cond_block in range(3):
            picks = rng.choice(8, size=3, replace=False)
            for pk in picks:
                order.append(all_traj[pk * 3 + cond_block])
        calibs = [f"c{c+1:02d}" for c in rng.choice(12, size=4, replace=False)]
        seq = order + calibs
        rng.shuffle(seq)
        rand_rows.append(f"P{p:02d},{' '.join(seq)}")
    open(os.path.join(BASE, "randomization.csv"), "w").write(
        "\n".join(rand_rows) + "\n")

    zpath = os.path.join(BASE, "stimuli.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(STIM)):
            z.write(os.path.join(STIM, fn), f"stimuli/{fn}")
        for fn in ["key.csv", "ratings_template.csv", "randomization.csv",
                   "protocol.md"]:
            z.write(os.path.join(BASE, fn), fn)
    print("wrote", zpath)


if __name__ == "__main__":
    main()
