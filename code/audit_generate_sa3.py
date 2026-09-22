"""Generation audit with Stable Audio 3 (stable-audio-tools).

Runs on the GPU server. Same manifest format as audit_generate_sao.py /
audit_generate.py so audit_score.py works on all of them.

Setup (once):
  pip install ./stable-audio-tools   # patched local clone (py3.12 pins)
  hf download stabilityai/stable-audio-3-medium --local-dir stable-audio-3-medium

Usage:
  python3 scripts/audit_generate_sa3.py --model_dir stable-audio-3-medium \
      --out_dir results/audit_audio_sa3 --device cuda
Resume-safe: already-generated clip_ids in manifest.jsonl are skipped.

Seed note: per-clip seed = base_seed * 100003 + md5(clip_id) % 100003
(md5, not Python hash(), so it is stable across processes; the derived
seed is recorded per clip in the manifest).
"""
import argparse, hashlib, json, os, time
import numpy as np


def stable_clip_seed(base_seed: int, clip_id: str) -> int:
    h = int(hashlib.md5(clip_id.encode()).hexdigest(), 16) % 100003
    return (base_seed * 100003 + h) % (2**31 - 1)


def build_conditioning(model_config, prompt, dur):
    """Fill conditioning ids declared in the model config."""
    cond = {}
    for c in model_config["model"]["conditioning"]["configs"]:
        cid = c["id"]
        if cid == "prompt":
            cond[cid] = prompt
        elif cid == "seconds_start":
            cond[cid] = 0
        elif cid in ("seconds_total", "duration"):
            cond[cid] = float(dur)
        else:
            raise ValueError(f"Unhandled conditioning id: {cid}")
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--model_name", default=None,
                    help="model name recorded in manifest; defaults to basename of --model_dir")
    ap.add_argument("--prompts", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "prompts_audit.json"))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--duration_s", type=float, default=None)
    ap.add_argument("--per_prompt", type=int, default=None)
    # Official SA3 recipe (model card): adversarially post-trained model,
    # few-step pingpong sampler, no CFG.
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg_scale", type=float, default=1.0)
    ap.add_argument("--sampler_type", default="pingpong")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_clips", type=int, default=0)
    ap.add_argument("--fp32", action="store_true",
                    help="keep model in float32 (default: half precision on cuda)")
    args = ap.parse_args()

    import torch, soundfile as sf
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import load_ckpt_state_dict
    from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint

    model_dir = os.path.abspath(args.model_dir)
    model_name = args.model_name or os.path.basename(model_dir.rstrip("/"))
    model_config = json.load(open(os.path.join(model_dir, "model_config.json")))

    # Point a t5gemma conditioner at the copy bundled inside the model repo
    # so no extra (gated) download is needed.
    for c in model_config["model"]["conditioning"]["configs"]:
        if c.get("type") == "t5gemma":
            local_t5 = os.path.join(model_dir, "t5gemma-b-b-ul2")
            if os.path.isdir(local_t5):
                c.setdefault("config", {})
                c["config"]["model_path"] = local_t5
                c["config"].pop("repo_id", None)
                c["config"].pop("subfolder", None)

    model = create_model_from_config(model_config)
    ckpt = os.path.join(model_dir, "model.safetensors")
    model.load_state_dict(load_ckpt_state_dict(ckpt))
    model = model.to(args.device).eval().requires_grad_(False)
    if args.device.startswith("cuda") and not args.fp32:
        model = model.to(torch.float16)

    sr = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    spec = json.load(open(args.prompts))
    dur = args.duration_s or spec["duration_s"]
    npp = args.per_prompt or spec["per_prompt_generations"]

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
            clip_seed = stable_clip_seed(args.seed, clip_id)
            t0 = time.time()
            with torch.no_grad():
                audio = generate_diffusion_cond_inpaint(
                    model,
                    steps=args.steps,
                    cfg_scale=args.cfg_scale,
                    conditioning=[build_conditioning(model_config, prompt, dur)],
                    sample_size=sample_size,
                    sampler_type=args.sampler_type,
                    seed=clip_seed,
                    device=args.device,
                )
            audio = audio[0].to(torch.float32).cpu()        # (ch, samples)
            audio = audio[:, : int(dur * sr)]
            audio = audio.clamp(-1, 1).T.numpy()            # (samples, ch)
            wav = os.path.join(args.out_dir, clip_id + ".wav")
            sf.write(wav, audio, sr)
            rec = {"clip_id": clip_id, "group": group, "category": cat,
                   "prompt": prompt, "gen_index": g, "wav": wav, "sr": int(sr),
                   "duration_s": float(audio.shape[0] / sr),
                   "gen_seconds": round(time.time() - t0, 1),
                   "model": model_name,
                   "num_inference_steps": args.steps,
                   "cfg_scale": args.cfg_scale,
                   "sampler_type": args.sampler_type,
                   "seed": args.seed, "clip_seed": clip_seed}
            mf.write(json.dumps(rec) + "\n"); mf.flush()
            n_done += 1
            print(f"[{n_done}] {clip_id} ({rec['gen_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
