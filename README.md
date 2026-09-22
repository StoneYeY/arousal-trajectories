# Auditing and Steering Arousal Trajectories in Text-to-Music Generation

Audio companion for the ICASSP 2027 submission, served from `docs/` via GitHub Pages:
https://stoneyey.github.io/arousal-trajectories/ (fully self-contained, all audio embedded:
six audit generations with proxy scores, two blinded listening sets and one stable-audio-3-medium
steering episode revealed on demand, eight calibration clips comparing the proxy with listener ratings).

Every number in the paper is computed by the scripts in `code/` from the files in `data/`
(plus the DEAM features, which `code/prep_deam.py` and `code/features.py` regenerate from the
public DEAM corpus). Scores are on the DEAM 1–9 arousal scale.

## data/
| file | rows | what it is |
|---|---|---|
| `prompts.csv` | 70 | audit prompts (5 anxiolytic categories x 10, neutral x 10, high-arousal x 10); `in_decoding_sweep` marks the 20 reused in Table 1 |
| `audit_scores.csv` | 840 | stable-audio-3-medium generations (45 s, 12 seeds/prompt): sampler settings and seeds; `proxy_arousal_deploy` (proxy refit on all DEAM), `proxy_arousal_foldmean` (mean of the five fold models = the paper's out-of-fold footing), `*_lufs23_*` (after -23 LUFS normalisation), `aux_arousal` (auxiliary ridge estimator), CLAP similarity, integrated loudness, six descriptors |
| `audit_sweep_scores.csv` | 400 | decoding sweep (5 configurations x 20 prompts x 4 seeds), same columns plus `config`, `peak_in` |
| `audit_musicgen_scores.csv` | 280 | MusicGen-small replication (70 prompts x 4 seeds, 30 s) |
| `deam_reference_oof.csv` | 1795 | every DEAM excerpt: fold (KFold(5, shuffle, seed 42)), labels, out-of-fold and in-sample proxy/auxiliary scores, and the reference-group flags (`human_calm_ref` = lowest-arousal quartile & above-median valence, n=82; `human_tense_ref` = highest quartile, n=466; the other flags give the three alternative definitions) |
| `planning_synth.json` | 100 episodes | synthesiser planning: per-episode schedule, proxy and auxiliary trajectories, costs; best-of-N dual curve |
| `planning_sa3.json` | 200 candidates | stable-audio-3-medium candidate pool (10 episodes x 5 segments x 4 candidates) and the per-planner selection summary (Table 2) |
| `listening_ratings.csv` | 864 | trajectory ratings, 24 listeners (P01–P24), 9 items each, 4 questions (7-point); `valid` flags the 857 rows used (3 out-of-range, 4 blank) |
| `listening_calibration_ratings.csv` | 384 | single-clip perceived-arousal ratings (1–9), 16 items per listener |
| `listening_participants.csv` | 24 | age, years of musical training, headphones yes/no (no other personal data was collected) |
| `listening_items.csv` | 24 | trajectory stimuli: planner, s0, cosine schedule, proxy trajectory, tracking metrics, number of raters, mean ratings |
| `calibration_items.csv` | 32 | single-clip items: domain, proxy score (and how it was computed), DEAM song id and label for the 8 real-music items, intended level for the 12 synthesiser presets, number of raters, listener mean |
| `listening_protocol.md`, `listening_key*.csv`, `listening_randomization.csv`, `listening_calib_randomization.csv`, `listening_item_start_levels.csv` | | the protocol and the per-listener presentation orders |
| `listening_summary.json`, `round3_analyses.json`, `oof_reference_summary.json` | | the analysis outputs the paper quotes (Sections 3–4) |

Listening scales, as read to participants (translated from the Chinese protocol): endpoint calmness 1 = not calm at all … 7 = very calm;
transition smoothness 1 = abrupt … 7 = very smooth; start match 1 = does not match the stated start at all … 7 = matches very well
(too high or too low both count as mismatch); overall trajectory 1 = not realised at all … 7 = realised very well;
single clips: 1 = extremely calm … 9 = extremely aroused. Before each trajectory the stated starting level s0 was read out.

## stimuli/
`t01–t24`: the 24 synthesiser trajectories (five 5-s segments joined by hard cuts, not loudness-matched; jump items average
about -43 LUFS and scheduled items about -25 LUFS). `c01–c12`: the 12 synthesiser calibration segments (12 s).
`g01–g12`: the 12 stable-audio-3 calibration generations (45 s; listeners heard the first 20 s). The 8 DEAM excerpts
(`d01–d08`) are not redistributed; they are DEAM songs 745, 167, 186, 1749, 500, 1379, 1629, 343 (first 20 s of the
45-s excerpt), see `calibration_items.csv`.

## models/
`verifier_*_deploy.joblib`: the deployed proxy (HistGradientBoosting on 17 descriptors + 88 eGeMAPS functionals, refit on all
1,795 DEAM excerpts; `cols` lists the feature order). `verifier_*_independent.joblib`: the auxiliary ridge estimator on the 17
descriptors. `report.json`: the training protocol and held-out metrics. Score new audio with `code/score.py`.

## code/
Feature extraction (`features.py`), proxy training (`train_verifier.py`), audit scoring and statistics (`audit_extract.py`,
`audit_score.py`, `audit_stats.py`, `oof_reference.py`), planning (`midi_synth.py`, `iso_agent.py`, `bon_curve.py`,
`gen_traj_pool.py`, `step3_analysis.py`), listening-study stimuli and analysis (`make_stimuli.py`, `analyze_listening.py`),
the round-3 analyses (`round3_analysis.py`), figures (`make_figs.py`), the companion page (`build_demo.py`) and this release
(`make_release.py`). Generation scripts (`audit_generate_sa3.py`, `audit_generate.py`) were run on a separate GPU host with
stable-audio-tools 0.0.20 / audiocraft; sampler settings and seeds are in the score tables.

## Ethics
The listening test involved 24 adult volunteers; no compensation was paid, no personal data beyond age, years of musical
training and headphone use were recorded, and ratings are released under pseudonyms P01–P24.
