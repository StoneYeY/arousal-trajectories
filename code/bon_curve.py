"""Best-of-N scaling curve: rule+BoN at N in {1,2,4,8} on episodes 0-4.

Resume-safe by (seed, N). Output: results/bon_curve.jsonl.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from iso_agent import run_episode

OUT = "/home/claude/amg/results/bon_curve.jsonl"

def main():
    rng = np.random.default_rng(7)
    s0s = rng.uniform(5.5, 8.5, size=20)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
                done.add((r["seed"], r["n_candidates"]))
            except (json.JSONDecodeError, KeyError):
                pass
    with open(OUT, "a") as f:
        for i in range(5):
            for N in [1, 2, 4, 8]:
                if (1000 + i, N) in done:
                    continue
                ep = run_episode(float(s0s[i]), 3.0, 5, 12.0, "rule+BoN",
                                 N, seed=1000 + i)
                f.write(json.dumps(ep) + "\n"); f.flush()
                print(f"ep{i} N={N} rmse={ep['costs']['schedule_rmse']:.3f}",
                      flush=True)

if __name__ == "__main__":
    main()
