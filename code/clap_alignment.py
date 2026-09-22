"""CLAP text-audio alignment for the audit clips — RUNS ON THE GPU SERVER
(CLAP weights come from HuggingFace, unreachable from the cloud sandbox).

Purpose (prompt-compliance baseline): separates two explanations of a
high-arousal 'calm' clip —
  (a) the model ignored the prompt        -> low CLAP alignment
  (b) the model followed the prompt but the acoustics are still aroused
                                          -> high CLAP alignment
Case (b) is the interesting finding: semantic alignment does not
guarantee arousal alignment.

Usage (on the GPU server, after generation):
  pip install laion-clap
  python3 scripts/clap_alignment.py \
      --audit_dir results/audit_audio_sao \
      --out results/clap_alignment.jsonl
Output: one JSON line per clip {clip_id, prompt, clap_similarity}.
Copy the jsonl back with sao22.zip.
"""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import numpy as np
    import laion_clap

    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()  # default 630k-audioset checkpoint

    recs = [json.loads(l) for l in
            open(os.path.join(args.audit_dir, "manifest.jsonl"))]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["clip_id"] for l in open(args.out)}

    with open(args.out, "a") as f:
        for i, rec in enumerate(recs):
            if rec["clip_id"] in done:
                continue
            wav = rec["wav"] if os.path.exists(rec["wav"]) else \
                os.path.join(args.audit_dir, rec["clip_id"] + ".wav")
            if not os.path.exists(wav):
                continue
            ae = model.get_audio_embedding_from_filelist([wav], use_tensor=False)
            te = model.get_text_embedding([rec["prompt"], ""], use_tensor=False)[:1]
            sim = float(np.dot(ae[0], te[0]) /
                        (np.linalg.norm(ae[0]) * np.linalg.norm(te[0]) + 1e-9))
            f.write(json.dumps({"clip_id": rec["clip_id"],
                                "prompt": rec["prompt"],
                                "clap_similarity": sim}) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"{i+1}/{len(recs)}", flush=True)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
