"""Stage 1 of the audit: extract features for every audit clip, RESUMABLE.

Per clip: hand+eGeMAPS features on (a) the raw audio and (b) the
LUFS-normalised audio (loudness-confound sensitivity). One extraction
pass each — predictions and statistics happen later in audit_score.py
from these cached features, so nothing is ever recomputed.

Sharded like prep_deam.py: --shard k/n appends to a shard jsonl;
finished clip_ids are skipped on restart.

Usage:
  python3 audit_extract.py --audit_dir data/sa3_audit/sa322 \
      --out results/sa3_features.jsonl --shard 0/2
"""
import argparse, glob, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features, extract_egemaps

TARGET_LUFS = -23.0


def feats_for_wave(y, sr):
    """hand + eGeMAPS for an in-memory waveform (temp wav for openSMILE)."""
    import soundfile as sf, tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, y, sr)
    try:
        d = extract_features(f.name)
        d.update(extract_egemaps(f.name))
        return d
    finally:
        os.unlink(f.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--no_lufs", action="store_true",
                    help="skip the LUFS-normalised second pass (step-3 sets)")
    args = ap.parse_args()

    import librosa, pyloudnorm
    k, nsh = map(int, args.shard.split("/"))
    rows_path = args.out if nsh == 1 else \
        args.out.replace(".jsonl", f".shard{k}.jsonl")

    man = os.path.join(args.audit_dir, "manifest.jsonl")
    recs = [json.loads(l) for l in open(man)]
    done = set()
    for shard in glob.glob(args.out.replace(".jsonl", "*.jsonl")):
        for line in open(shard):
            try:
                done.add(json.loads(line)["clip_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [r for i, r in enumerate(recs)
            if i % nsh == k and r["clip_id"] not in done]
    print(f"total={len(recs)} done={len(done)} todo(shard {k}/{nsh})={len(todo)}",
          flush=True)

    meter = pyloudnorm.Meter(22050)
    t0 = time.time()
    with open(rows_path, "a") as f:
        for n, rec in enumerate(todo, 1):
            path = None
            for ext in (".flac", ".wav"):
                c = os.path.join(args.audit_dir, rec["clip_id"] + ext)
                if os.path.exists(c):
                    path = c; break
            if path is None:
                continue
            try:
                y, sr = librosa.load(path, sr=22050, mono=True)
                feat_raw = feats_for_wave(y, sr)
                lufs = meter.integrated_loudness(y)
                y_ln = y
                if args.no_lufs:
                    feat_ln = None
                elif np.isfinite(lufs):
                    y_ln = pyloudnorm.normalize.loudness(y, lufs, TARGET_LUFS)
                    peak = np.max(np.abs(y_ln))
                    if peak > 0.99:
                        y_ln = y_ln * (0.99 / peak)
                if not args.no_lufs:
                    feat_ln = feats_for_wave(y_ln.astype(np.float32), sr)
                out = {"clip_id": rec["clip_id"], "group": rec.get("group"),
                       "category": rec.get("category"), "prompt": rec["prompt"],
                       "gen_index": rec.get("gen_index"),
                       "model": rec.get("model"),
                       "meta": {k: v for k, v in rec.items()
                                if k not in ("clip_id", "prompt", "wav")},
                       "peak_in": float(np.max(np.abs(y))) if len(y) else None,
                       "lufs_in": float(lufs) if np.isfinite(lufs) else None,
                       "feat_raw": feat_raw, "feat_ln": feat_ln}
            except Exception as e:
                out = {"clip_id": rec["clip_id"], "_error": str(e)[:200]}
            f.write(json.dumps(out) + "\n")
            f.flush()
            if n % 20 == 0:
                rate = n / (time.time() - t0)
                print(f"{n}/{len(todo)} ({rate:.2f} clips/s, "
                      f"ETA {(len(todo)-n)/rate/60:.0f} min)", flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
