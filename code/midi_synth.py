"""Parametric MIDI segment generator + fluidsynth renderer.

This is the ACTION SPACE of the ISO-trajectory agent: each segment is
defined by interpretable musical parameters; the agent plans a sequence
of segments (a trajectory). Rendering: FluidR3_GM via fluidsynth.

Parameters per segment:
  tempo_bpm      : beats per minute
  register       : MIDI note of tonal center (e.g., 48 = C3)
  density        : notes per beat (0.25 .. 4)
  mode           : 'major' | 'minor'
  velocity       : 20..110 (loudness)
  program        : GM instrument (0=piano, 48=strings, 89=warm pad...)
  voicing_spread : semitone spread of chord voicing (wide = consonant feel)
  ambient        : if True, sustained overlapping tones; else articulated
"""
import subprocess, os, tempfile
import numpy as np
import pretty_midi

SCALE = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}
SF2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "FluidR3_GM.sf2")


def segment_to_midi(pm, start_t, params, dur_s, rng):
    tempo = params["tempo_bpm"]
    beat = 60.0 / tempo
    scale = SCALE[params["mode"]]
    root = params["register"]
    inst = pretty_midi.Instrument(program=params.get("program", 0))
    dens = params["density"]
    vel = int(params["velocity"])
    spread = params.get("voicing_spread", 12)
    ambient = params.get("ambient", False)

    t = start_t
    end_t = start_t + dur_s
    if ambient:
        # sustained overlapping chord tones, very slow harmonic rhythm
        chord_dur = max(8 * beat, 6.0)
        degrees = [0, 2, 4]  # triad in scale steps
        while t < end_t:
            for k, d in enumerate(degrees):
                pitch = root + scale[d % 7] + 12 * (d // 7)
                pitch += (spread // 12) * 12 * (k % 2)
                nt = pretty_midi.Note(
                    velocity=max(20, vel + rng.integers(-5, 6)),
                    pitch=int(np.clip(pitch, 24, 96)),
                    start=t, end=min(t + chord_dur * 1.1, end_t))
                inst.notes.append(nt)
            t += chord_dur
    else:
        step = beat / dens
        degrees = [0, 2, 4, 5, 7]
        while t < end_t:
            d = int(rng.choice(degrees))
            pitch = root + scale[d % 7] + 12 * (d // 7)
            if rng.random() < 0.3:
                pitch += int(rng.choice([-12, 12])) * (spread >= 12)
            nt = pretty_midi.Note(
                velocity=max(20, vel + int(rng.integers(-8, 9))),
                pitch=int(np.clip(pitch, 24, 96)),
                start=t, end=t + step * (1.6 if dens <= 1 else 0.9))
            inst.notes.append(nt)
            t += step
    pm.instruments.append(inst)
    return end_t


def render_trajectory(segments, out_wav, seed=0, sr=22050):
    """segments: list of (params_dict, duration_s). Returns out_wav path."""
    rng = np.random.default_rng(seed)
    pm = pretty_midi.PrettyMIDI(initial_tempo=segments[0][0]["tempo_bpm"])
    t = 0.0
    for params, dur in segments:
        t = segment_to_midi(pm, t, params, dur, rng)
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        mid_path = f.name
    pm.write(mid_path)
    subprocess.run(
        ["fluidsynth", "-ni", "-g", "0.7", "-F", out_wav, "-r", str(sr),
         SF2, mid_path],
        check=True, capture_output=True)
    os.unlink(mid_path)
    return out_wav


# Convenience presets spanning the arousal axis (used for calibration tests)
PRESETS = {
    "very_calm": dict(tempo_bpm=56, register=48, density=0.5, mode="major",
                      velocity=30, program=88, voicing_spread=19, ambient=True),
    "calm": dict(tempo_bpm=72, register=55, density=1.0, mode="major",
                 velocity=52, program=0, voicing_spread=12, ambient=False),
    "neutral": dict(tempo_bpm=100, register=60, density=2.0, mode="major",
                    velocity=70, program=0, voicing_spread=12, ambient=False),
    "tense": dict(tempo_bpm=132, register=65, density=3.0, mode="minor",
                  velocity=88, program=30, voicing_spread=7, ambient=False),
    "agitated": dict(tempo_bpm=160, register=70, density=4.0, mode="minor",
                     velocity=105, program=30, voicing_spread=6, ambient=False),
}

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/traj_test.wav"
    segs = [(PRESETS["agitated"], 8), (PRESETS["neutral"], 8),
            (PRESETS["calm"], 8), (PRESETS["very_calm"], 8)]
    render_trajectory(segs, out)
    print("rendered", out)
