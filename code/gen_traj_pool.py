"""SA3 trajectory candidate pools — RUNS ON THE GPU SERVER.

Purpose (transfer experiment): show the same planning objective steers
the AUDITED text-to-music model, not only the MIDI synthesiser.
For each of 10 episodes (same seeds/s0 as the MIDI experiment) and each
segment k of a cosine schedule r_1..r_5, generate N=4 candidate 12-s
clips whose PROMPT verbalises the scheduled arousal level. Selection
happens later in the cloud (verifier picks per Eq. (sel)); jump uses
the r_K-level pool for every segment. Nothing is selected here.

Usage (after the SA3 model is set up as for audit_generate_sa3.py):
  python3 scripts/gen_traj_pool.py --model_dir stable-audio-3-medium \
      --out_dir results/traj_pool --device cuda
Then transcode + zip like step2 (22.05k mono FLAC + manifest).
"""
import argparse, hashlib, json, os, time
import numpy as np

# verbal anchors along the proxy arousal axis (1-9); linear pick by level
ANCHORS = [
    (2.0, "extremely calm, still ambient music, very slow, soft sustained warm pads, quiet"),
    (3.5, "gentle relaxed instrumental music, slow tempo, soft dynamics, warm and peaceful"),
    (5.0, "moderately energetic instrumental music, steady mid-tempo groove, balanced dynamics"),
    (6.5, "energetic driving instrumental music, fast tempo, prominent percussion, bright"),
    (8.0, "very intense aggressive music, very fast, loud pounding drums, distorted, urgent"),
]

def prompt_for_level(r):
    best = min(ANCHORS, key=lambda a: abs(a[0] - r))
    return best[1]

def cosine_sched(s0, s_star=3.0, K=5):
    ks = np.arange(K)
    return s_star + (s0 - s_star) * 0.5 * (1 + np.cos(np.pi * ks / (K - 1)))

def stable_seed(tag):
    return int(hashlib.md5(tag.encode()).hexdigest(), 16) % (2**31 - 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_candidates", type=int, default=4)
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--seg_dur", type=float, default=12.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg_scale", type=float, default=1.0)
    ap.add_argument("--sampler_type", default="pingpong")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch, soundfile as sf
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import load_ckpt_state_dict
    from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint

    model_dir = os.path.abspath(args.model_dir)
    cfg = json.load(open(os.path.join(model_dir, "model_config.json")))
    for c in cfg["model"]["conditioning"]["configs"]:
        if c.get("type") == "t5gemma":
            local = os.path.join(model_dir, "t5gemma-b-b-ul2")
            if os.path.isdir(local):
                c.setdefault("config", {})
                c["config"]["model_path"] = local
                c["config"].pop("repo_id", None); c["config"].pop("subfolder", None)
    model = create_model_from_config(cfg)
    model.load_state_dict(load_ckpt_state_dict(os.path.join(model_dir, "model.safetensors")))
    model = model.to(args.device).eval().requires_grad_(False).to(torch.float16)
    sr, sample_size = cfg["sample_rate"], cfg["sample_size"]

    rng = np.random.default_rng(7)                # same as MIDI experiment
    s0s = rng.uniform(5.5, 8.5, size=20)[: args.n_episodes]

    os.makedirs(args.out_dir, exist_ok=True)
    man = os.path.join(args.out_dir, "manifest.jsonl")
    done = {json.loads(l)["clip_id"] for l in open(man)} if os.path.exists(man) else set()

    def build_cond(prompt, dur):
        cond = {}
        for c in cfg["model"]["conditioning"]["configs"]:
            cid = c["id"]
            cond[cid] = prompt if cid == "prompt" else (0 if cid == "seconds_start" else float(dur))
        return cond

    with open(man, "a") as mf:
        for i, s0 in enumerate(s0s):
            sch = cosine_sched(float(s0), K=args.K)
            for k, r in enumerate(sch):
                prompt = prompt_for_level(r)
                for c in range(args.n_candidates):
                    clip_id = f"ep{i:02d}.k{k}.c{c}"
                    if clip_id in done:
                        continue
                    seed = stable_seed(clip_id)
                    t0 = time.time()
                    with torch.no_grad():
                        audio = generate_diffusion_cond_inpaint(
                            model, steps=args.steps, cfg_scale=args.cfg_scale,
                            conditioning=[build_cond(prompt, args.seg_dur)],
                            sample_size=sample_size,
                            sampler_type=args.sampler_type,
                            seed=seed, device=args.device)
                    y = audio[0].to(torch.float32).cpu()[:, : int(args.seg_dur * sr)]
                    y = y.clamp(-1, 1).T.numpy()
                    sf.write(os.path.join(args.out_dir, clip_id + ".wav"), y, sr)
                    mf.write(json.dumps({
                        "clip_id": clip_id, "episode": i, "segment": k,
                        "candidate": c, "s0": float(s0), "r_k": float(r),
                        "prompt": prompt, "clip_seed": seed, "sr": int(sr),
                        "steps": args.steps, "cfg_scale": args.cfg_scale,
                        "sampler_type": args.sampler_type,
                        "model": os.path.basename(model_dir)}) + "\n")
                    mf.flush()
                    print(f"{clip_id} r={r:.2f} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
