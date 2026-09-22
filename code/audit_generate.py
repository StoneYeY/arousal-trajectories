"""Generation audit: text-to-music model vs. anxiolytic prompt set.

Loads MusicGen from a LOCAL directory (weights staged from the user's
machine; HF hub is unreachable from this sandbox). CPU inference.

For each prompt in prompts_audit.json, generate `per_prompt_generations`
clips of `duration_s` seconds, save wavs + a manifest. Scoring happens
separately (audit_score.py) once the verifier is trained.

Usage:
  python3 audit_generate.py --model_dir amg/data/musicgen-small \
      --out_dir amg/results/audit_audio [--subset anxiolytic.sleep]
"""
import argparse, json, os, time, itertools
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--prompts", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "prompts_audit.json"))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--duration_s", type=float, default=None)
    ap.add_argument("--per_prompt", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_clips", type=int, default=0,
                    help="stop after N clips (for CPU time budgeting)")
    args = ap.parse_args()

    import torch, soundfile as sf
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    torch.manual_seed(args.seed)

    spec = json.load(open(args.prompts))
    dur = args.duration_s or spec["duration_s"]
    dur = min(dur, 30.0)  # MusicGen context limit (~30 s)
    npp = args.per_prompt or spec["per_prompt_generations"]

    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = MusicgenForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=torch.float32)
    model.eval()
    sr = model.config.audio_encoder.sampling_rate
    frame_rate = model.config.audio_encoder.frame_rate
    max_new = int(dur * frame_rate)

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")
    done = set()
    if os.path.exists(manifest_path):
        for line in open(manifest_path):
            done.add(json.loads(line)["clip_id"])

    tasks = []
    for group, cats in [("anxiolytic", spec["anxiolytic"]),
                        ("controls", spec["controls"])]:
        for cat, prompts in cats.items():
            for pi, prompt in enumerate(prompts):
                for g in range(npp):
                    tasks.append((group, cat, pi, prompt, g))

    n_done = 0
    with open(manifest_path, "a") as mf:
        for group, cat, pi, prompt, g in tasks:
            clip_id = f"{group}.{cat}.p{pi}.g{g}"
            if clip_id in done:
                continue
            if args.max_clips and n_done >= args.max_clips:
                print("clip budget reached"); break
            t0 = time.time()
            inputs = processor(text=[prompt], padding=True, return_tensors="pt")
            with torch.no_grad():
                audio = model.generate(**inputs, do_sample=True,
                                       guidance_scale=3.0,
                                       max_new_tokens=max_new)
            y = audio[0, 0].cpu().numpy()
            wav = os.path.join(args.out_dir, clip_id + ".wav")
            sf.write(wav, y, sr)
            rec = {"clip_id": clip_id, "group": group, "category": cat,
                   "prompt": prompt, "gen_index": g, "wav": wav, "sr": sr,
                   "duration_s": float(len(y) / sr),
                   "gen_seconds": round(time.time() - t0, 1),
                   "model": os.path.basename(args.model_dir.rstrip("/")),
                   "guidance_scale": 3.0, "seed": args.seed}
            mf.write(json.dumps(rec) + "\n"); mf.flush()
            n_done += 1
            print(f"[{n_done}] {clip_id} ({rec['gen_seconds']}s gen)", flush=True)


if __name__ == "__main__":
    main()
