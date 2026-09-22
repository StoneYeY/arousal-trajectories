"""Prepare DEAM: join static annotations + extract features. RESUMABLE.

Each song's features are appended to <out>.rows.jsonl the moment they are
computed; on restart, songs already present are skipped. The final CSV is
assembled at the end (or by --finalize alone).

Prefers 22.05 kHz mono wavs in <data_dir>/wav22/ (fast soundfile path);
falls back to the original mp3s.

Usage:
  python3 prep_deam.py --data_dir amg/data/deam --out amg/results/deam_features.csv
  python3 prep_deam.py ... --finalize   # just rebuild csv from jsonl
"""
import argparse, os, sys, glob, json, time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features, extract_egemaps


def find_annotations(data_dir):
    hits = glob.glob(os.path.join(
        data_dir, "**", "static_annotations_averaged_songs_1_2000.csv"),
        recursive=True)
    if not hits:
        raise FileNotFoundError("static annotations csv not found under " + data_dir)
    frames = []
    for f in [hits[0], hits[0].replace("songs_1_2000", "songs_2000_2058")]:
        if os.path.exists(f):
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            frames.append(df)
    ann = pd.concat(frames, ignore_index=True)
    keep = ["song_id", "valence_mean", "valence_std", "arousal_mean", "arousal_std"]
    return ann[[c for c in keep if c in ann.columns]]


def audio_index(data_dir):
    idx = {}
    wavdir = os.path.join(data_dir, "wav22")
    if os.path.isdir(wavdir):
        for p in glob.glob(os.path.join(wavdir, "*.wav")):
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem.isdigit():
                idx[int(stem)] = p
    if not idx:
        for p in glob.glob(os.path.join(data_dir, "**", "*.mp3"), recursive=True):
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem.isdigit():
                idx[int(stem)] = p
    return idx


def finalize(ann, rows_path, out):
    rows = []
    for shard in glob.glob(rows_path.replace(".jsonl", "*.jsonl")):
        for line in open(shard):
            try:
                r = json.loads(line)
                if "_error" not in r:
                    rows.append(r)
            except json.JSONDecodeError:
                pass  # torn write from an interrupted append
    feats = pd.DataFrame(rows).drop_duplicates(subset="song_id", keep="last")
    df = ann.merge(feats, on="song_id", how="inner")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}: {df.shape[0]} rows x {df.shape[1]} cols", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_egemaps", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--shard", default="0/1",
                    help="k/n: process songs where song_id %% n == k, "
                         "writing to a shard-specific jsonl")
    args = ap.parse_args()

    ann = find_annotations(args.data_dir)
    base_rows = args.out + ".rows.jsonl"
    if args.finalize:
        finalize(ann, base_rows, args.out)
        return

    k, nsh = map(int, args.shard.split("/"))
    rows_path = base_rows if nsh == 1 else \
        args.out + f".rows.shard{k}.jsonl"

    idx = audio_index(args.data_dir)
    done = set()
    for shard in glob.glob(base_rows.replace(".jsonl", "*.jsonl")):
        for line in open(shard):
            try:
                done.add(json.loads(line)["song_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [(int(s), idx[int(s)]) for s in ann["song_id"]
            if int(s) in idx and int(s) not in done and int(s) % nsh == k]
    if args.limit:
        todo = todo[: args.limit]
    print(f"annotations={len(ann)} audio={len(idx)} done={len(done)} todo={len(todo)}",
          flush=True)

    t0 = time.time()
    with open(rows_path, "a") as f:
        for n, (sid, path) in enumerate(todo, 1):
            try:
                feats = extract_features(path)
                if not args.no_egemaps:
                    feats.update(extract_egemaps(path))
                feats["song_id"] = sid
            except Exception as e:
                feats = {"song_id": sid, "_error": str(e)[:200]}
            f.write(json.dumps(feats) + "\n")
            f.flush()
            if n % 25 == 0:
                rate = n / (time.time() - t0)
                eta = (len(todo) - n) / rate / 60
                print(f"{n}/{len(todo)} ({rate:.2f} songs/s, ETA {eta:.0f} min)",
                      flush=True)

    finalize(ann, rows_path, args.out)


if __name__ == "__main__":
    main()
