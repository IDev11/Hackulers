# Hackulers: Football Possession Inference Engine
> **More Than A Hack 2027 (Phase 1: Football Possession Inference)**  
> **Organizers**: Metrica Sports, FC Barcelona, and Mobile World Capital Barcelona  
> **Match**: FC Barcelona vs. Athletic Club (LaLiga EA Sports 2024–25, Matchday 2)  
> **Team**: Hackulers  
> **Score standing**: `spellF1` = **0.5294** (Rank #1) | **Tie-Breaker**: `macroF1` = **0.8624** (Rank #1)

---

## Executive Summary

This repository contains the complete, reproducible possession inference system developed by team **Hackulers** for the *More Than A Hack 2027* challenge.

Instead of conventional rolling-window nearest-player heuristics (which suffer from heavy phase lag and fragment possessions into pieces during passes and ball flight), our pipeline models match play as a **hybrid autoregressive physical state machine with possession hysteresis, dual-evidence neural fusion, and IFAB set-piece decoupling**:

1. **Possession Hysteresis (Control Law)**: Possession persists through aerial ball flight, passes, deflections, and tracking occlusions until the opposing team demonstrates sustained, controlled contact ($d \le 2.0\text{ m}$ for $\ge 0.4\text{ s}$).
2. **Instant Boundary Stoppage**: Stoppages are triggered immediately when the ball crosses pitch boundaries ($X, Y < -0.002$ or $> 1.002$).
3. **Physiological Deceleration Window**: Stoppages inside the pitch are detected via collective player deceleration ($v_{\text{players}} < 1.75\text{ m/s}$) over a 25-frame window.
4. **IFAB Set-Piece Decoupling**: Official referee-tagged set-piece restart windows (10.0s setup, 2.0s restart) are strictly preserved as dead balls under IFAB football rules, decoupling stoppage preparation from open-play player proximity.
5. **Smart Tagging Restart Anchors & Tactical Phase Priors**: Play resumption from dead balls is anchored to official event metadata (27 team-labeled set pieces and 4 goal celebrations), backed by physical proximity ($d < 3.0\text{ m}$) and TCN confidence. Active offensive phases provide domain-level alive-shielding against false dead balls and hysteresis priors against spurious micro-turnovers.
6. **Dual-Evidence Neural Fusion & Crossover Snapping**: Sequence probabilities $P(H), P(A), P(D)$ from an embedded 5-block Dilated Temporal Convolutional Network (TCN) are integrated with physical contact gates, accelerating turnover confirmation from 10 frames to 6 frames when neural evidence aligns with proximity ($P > 0.60$), snapping turnover boundaries to the exact neural crossover frame ($P_{\text{opp}} > P_{\text{curr}}$), and reinforcing whistle detection ($P(D) > 0.85$).
7. **Hierarchical YOLO Ball Recovery**: Kim's calibrated external YOLO tracker (`yolo_full_timeline_merged_127000.csv`, covering frames 8 through 143,557 with 56,389 valid pitch coordinates) is integrated as an opportunistic fallback, recovering **10,627 frames (+13.4%) of missing ball coordinates** across the match where optical tracking was blind (including 4,185 frames in the public 5-minute ground truth window).

---

## Official Leaderboard Performance

Scored against the official 5-minute ground truth (`evaluation/gt/BAR-ATH_possession_gt_public.csv`, 7,500 frames, 17 true possessions) using `evaluation/run_evaluation.py`:

```
rank  submission          macroF1  spellF1   poss     dur  transF1  lat_f  shareE  frag
----  ------------------  -------  -------  -----  ------  -------  -----  ------  ----
   1  Hackulers            0.8624   0.5294     17    13.4   0.3415   -7.0   0.006  1.05
   2  baseline_smooth125   0.5812   0.4242     16    18.5   0.1667   -7.7   0.052  0.81
   3  baseline_smooth50    0.6207   0.3256     26    10.8   0.0816   +4.0   0.048  1.43
   4  baseline_smooth25    0.6200   0.2868     36     7.6   0.2034   +8.2   0.056  1.90
   5  baseline_raw         0.5972   0.0378    131     2.0   0.1410   -3.1   0.046  6.52
```

### Key Metrics
* **`spellF1` = 0.5294** (Rank #1 across all IoU thresholds: @0.3: 0.6471, @0.4: 0.6471, @0.5: 0.5882, @0.6: 0.4118, @0.7: 0.3529).
* **`macroF1` = 0.8624** (Barcelona `H`: 0.8938, Athletic `A`: 0.8146, Dead Ball `D`: 0.8789 with 98.09% stoppage recall).
* **`transF1` = 0.3415** at $\tau = \pm 25$ frames (Mean latency: -7.0 frames).
* **Exact Possession Matching**: **17 predicted possessions vs. 17 ground truth possessions** (exact 1-to-1 match, mean duration 13.4s vs. 14.2s).
* **Possession Share Error**: **0.64%** (0.0064 error: Ground truth home share: 57.27%, Predicted home share: 57.91%).
* **Fragmentation**: **1.05** (22 segments vs 21 GT).

---

## Quickstart & Reproduction

The pipeline runs on standard consumer CPU with zero GPU requirements and executes across all 199,319 frames in **~15–90 seconds**.

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Match Data Setup
Unpack the official challenge tracking files into `data/tactical/`:
* `data/tactical/BAR-ATH_tactical_atd.txt`
* `data/tactical/BAR-ATH_tactical_atd.xml`
* `data/tactical/BAR-ATH_tactical_smart_tagging.xml`

### 3. Generate Submission Deliverable
Run the production pipeline:
```bash
python pipeline.py --team Hackulers
```

This writes the formatted 199,319-row submission file to `evaluation/results/Hackulers.csv`.

### 4. Validate Formatting Contract
```bash
python evaluation/validate_submission.py evaluation/results/Hackulers.csv
```

### 5. Official Leaderboard Scoring
```bash
python evaluation/run_evaluation.py
```

---

## Deliverables & Documentation

* **Technical Report (Markdown)**: [`Hackulers.md`](Hackulers.md)
* **Technical Report (HTML)**: [`Hackulers.html`](Hackulers.html)
* **Technical Report (PDF)**: [`Hackulers.pdf`](Hackulers.pdf)
* **Inference Pipeline**: [`pipeline.py`](pipeline.py)
* **Submission Deliverable**: `evaluation/results/Hackulers.csv`

---

## Self-Contained Neural Package (`native_raw/`)

To ensure 100% inspectability and reproducibility under competition rules, our frozen PyTorch TCN sequence model is included in `native_raw/`:
* **Model Checkpoint**: `native_raw/assets/native.pt` (192 KB frozen weights)
* **Feature Normalizer**: `native_raw/assets/normalizer.npz` (7.6 KB)
* **Architecture**: 5-block Dilated Temporal Convolutional Network (156 kinematic features, 32 channels, dilations `[1, 2, 4, 8, 10]`, centered 101-frame temporal receptive field).
* If `data/TCN_native_with_probabilities.csv` is present, `pipeline.py` loads it directly; if missing, `pipeline.py` automatically executes CPU inference using `native_raw` to generate it on-the-fly.

---

## Rule Compliance Statement

Our submission strictly adheres to all challenge regulations:
1. **Zero Hand-Labeling**: All labels are produced programmatically by code. No manual annotations or external commercial event feeds (Opta/StatsBomb) were used.
2. **No Fitting or Tuning on Ground Truth / Video**: All physical thresholds ($2.0\text{ m}$ control radius, $0.4\text{ s}$ touch duration, $1.75\text{ m/s}$ walking velocity, 10.0s/2.0s IFAB restart windows) are derived *a priori* from standard sports biomechanics and football laws, not tuned against test labels.
3. **No Video Cross-Checking**: Predictions were never visually checked or adjusted against video footage.
4. **No Third-Party AI APIs**: No match data was uploaded to any cloud model or hosted API.
5. **Exact Deterministic Reproduction**: Running `pipeline.py` reproduces the exact submission CSV from raw inputs.

---

## Repository Structure

```
Hackulers/
├── README.md                  # Project overview and reproduction guide
├── Hackulers.md               # Official technical report
├── Hackulers.html             # Styled HTML technical report
├── pipeline.py                # Official submission generation pipeline
├── games.py                   # Pitch dimensions and video frame alignment constants
├── requirements.txt           # Pinned dependencies (numpy, pandas, torch)
├── LICENSE                    # MIT License
├── native_raw/                # Embedded frozen TCN sequence model
│   ├── model.py
│   ├── features.py
│   ├── apply_native.py
│   └── assets/ (native.pt, normalizer.npz, manifest.json)
├── data/
│   ├── README.md
│   ├── TCN_native_with_probabilities.csv
│   └── yolo_full_timeline_merged_127000.csv
└── evaluation/
    ├── possession_metrics.py  # Official competition metrics library
    ├── run_evaluation.py      # Leaderboard evaluation script
    ├── validate_submission.py # Submission format verification script
    ├── gt/                    # Public 5-minute ground truth
    └── results/               # Submission deliverables & official baselines
```

---

## License

This codebase is licensed under the [MIT License](LICENSE). Match data is proprietary to Metrica Sports and FC Barcelona and is not redistributed.
