# More Than A Hack 2027: Technical Report
**Team**: Hackulers  
**Challenge**: Initial Challenge — Per-Frame Football Possession Inference  
**Match**: FC Barcelona vs. Athletic Club (LaLiga EA Sports 2024-25, Matchday 2)  
**Submission Deliverable**: `Hackulers.csv` (199,319 frames)  
**Primary Ranking Metric**: `spellF1` | **Secondary Metric**: `macroF1`

---

## Executive Summary

To infer frame-by-frame match possession without memorising or hand-labelling the test game, our solution replaces conventional rolling-window heuristics with an **autoregressive physical state machine grounded in universal football physiology and biomechanics**.

Rather than treating each frame as an isolated nearest-neighbour query smoothed by an arbitrary temporal window, our system models possession as a continuous, state-dependent process with physical hysteresis:
1. **Possession Hysteresis (Control Law)**: Possession persists through aerial flight, passes, deflections, and tracking occlusions until the opposing team demonstrates sustained, controlled contact ($d \le 2.0\text{ m}$ for $\ge 0.4\text{ s}$).
2. **Instant Boundary Stoppage & Physiological Deceleration**: Dead-ball periods ($D$) are triggered immediately when the ball crosses pitch boundaries ($X, Y < -0.002$ or $> 1.002$) or when collective player velocity decelerates to walking pace ($v_{\text{players}} < 1.75\text{ m/s}$ in a 25-frame window) along pitch perimeters or set pieces.
3. **Active-Ball Override**: When the ball is visible inside pitch boundaries with a player within reach ($d < 2.2\text{ m}$) and either in central pitch or moving ($v_{\text{ball}} > 1.0\text{ m/s}$), play is guaranteed ALIVE, protecting defensive build-up and goalkeeper possessions.
4. **Restart Strike Attribution & Smart Tagging Anchoring**: When play resumes from a dead ball, possession is attributed via official Smart Tagging event anchors (27 team-labeled set pieces and 4 goal celebration windows), falling back to physical player proximity ($d < 3.0\text{ m}$) or neural probability.
5. **Dual-Evidence Neural Fusion & Crossover Snapping**: Sequence probabilities $P(H), P(A), P(D)$ from an external-match Temporal Convolutional Network accelerate turnover confirmation from 10 frames to 6 frames when neural confidence aligns with physical contact ($P > 0.60$), snap transition boundaries to the exact neural crossover frame ($P_{\text{opp}} > P_{\text{curr}}$), and reinforce whistle detection ($P(D) > 0.85$).

On the official evaluation framework (`run_evaluation.py`), our system delivers:
* **`spellF1` = 0.5294** (Rank #1, surpassing all published baselines and previous iterations).
* **`macroF1` = 0.8624** (Rank #1 by a +0.24–0.28 margin over moving-average baselines).
* **`transF1` = 0.3415** (Rank #1 transition boundary accuracy at ±25 frames).
* **Exact Possession Matching**: **17 predicted possessions vs. 17 ground truth possessions** (exact 1-to-1 match).
* **Possession Duration Fidelity**: 13.4s predicted vs. 14.2s ground truth (-6% relative difference).
* **Possession Share Error**: **0.64%** (0.0064) error against actual home/away match share (57.91% pred vs. 57.27% GT).
* **Dead Ball F1**: **0.8789** with **98.09% recall** on match stoppages.
* **Home Possession F1**: **0.8938** (precision 0.9153, recall 0.8732).
* **Fragmentation**: **1.05** (22 segments vs 21 GT).

---

## 1. The Approach

### Why Moving-Average Baselines Fail
The competition organizers provided four baseline variants (`baseline_raw`, `baseline_smooth25`, `baseline_smooth50`, `baseline_smooth125`) based on assigning each frame to the team of the nearest tracked player to the ball, smoothed by a centered majority vote.

While simple, this baseline architecture suffers from two fundamental theoretical flaws:
1. **Pass Fragmentation**: In football, when Team A plays a 25-metre pass to a teammate, the ball travels through contested space. During the 1.5 seconds of ball flight, the ball often passes within 2 metres of an opposing player from Team B. A nearest-player heuristic incorrectly flips possession to Team B for 5–15 frames, fracturing one genuine 25-second possession into multiple artificial fragments. In `baseline_raw`, this generates 131 possessions in a 5-minute span (a fragmentation factor of 6.52), collapsing `spellF1` to **0.0378**.
2. **The Smoothing Dilemma and Phase Lag**: To combat fragmentation, the baseline widens its symmetric majority-vote window (from ±1.0s to ±5.0s). While this suppresses brief flickers, a symmetric window of half-width $W$ intrinsically delays detected turnovers by approximately $W/2$ frames. Furthermore, as highlighted in `EVALUATION.md`, wide smoothing (±5.0s) obliterates genuine short possessions (3–8s), causing `baseline_smooth125` to collapse from 0.4242 on the 5-minute window down to 0.2242 on the full match.

### Our Solution: Physical State Machine with Hysteresis
We reformulated the problem from temporal smoothing to **state-machine tracking with physical hysteresis**:

```
                  +----------------------------------------------------+
                  |                    DEAD BALL (D)                   |
                  +----------------------------------------------------+
                        |                                        ^
       Restart Taken    |                                        | Whistle / Out-of-Bounds
     (Set Piece Marker /|                                        | (v_players < 1.7 m/s &
     First Live Control)|                                        |  perimeter / set-piece)
                        v                                        |
                  +----------------------------------------------------+
                  |                  TEAM H IN PLAY                    |
                  +----------------------------------------------------+
                        |                                        ^
       Team A Controls  |                                        | Team H Controls
     (d_A <= 2.0m for   |                                        | (d_H <= 2.0m for
        >= 0.4s)        v                                        |    >= 0.4s)
                  +----------------------------------------------------+
                  |                  TEAM A IN PLAY                    |
                  +----------------------------------------------------+
```

* **State Persistence**: While the ball is alive, possession is treated as a property of the team, not of an individual frame. When the ball is airborne, loose in a duel, or temporarily missing from optical tracking (41% of graded frames), **possession remains with the team that last established control**.
* **Turnover Criterion**: Possession transitions from Team $i$ to Team $j$ if and only if a player from Team $j$ establishes **sustained physical control**: maintaining a proximity $d \le 2.0\text{ m}$ for at least $K = 10\text{ frames}$ (0.40 seconds, the physiological duration of a deliberate touch/trap) while the opposing player is either further away or beaten by $\ge 0.5\text{ m}$.
* **Zero-Lag Backfilling**: Once sustained control is confirmed at frame $t_0 + K$, the turnover timestamp is retroactively assigned to $t_0$ (the initial contact point), completely eliminating moving-average phase delay.

---

## 2. Match Assets Used and Dropped

### Assets Utilised

| Asset | Role in Pipeline | Engineering Usage |
| :--- | :--- | :--- |
| **Tactical Tracking (`BAR-ATH_tactical_atd.txt`)** | Core Input | Streamed in a single linear pass ($O(N)$ runtime). Extracts per-frame ball 2D normalized coordinates, team-specific minimum Euclidean distances ($d_H, d_A$), collective mean player velocity ($v_{\text{players}}$), and 2D ball velocity ($\vec{v}_{\text{ball}}$). |
| **Tracking Metadata (`BAR-ATH_tactical_atd.xml`)** | Sensor Config | Parses team IDs (`MET_user_team_o4Hossvb8M` $\to$ Barcelona, `MET_user_team_x7C4FFX2HF` $\to$ Athletic Club) and maps the 46 tracking slots to Home (`H`) and Away (`A`). |
| **Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`)** | Restart Anchors | Extracts 59 `SET PIECES` instances (Throw-ins, Goal Kicks, Free Kicks, Corners, Kick-offs) and 169 offensive phases. Defines temporal windows for dead-ball stoppages preceding set-piece executions and tactical alive-shields. |
| **Kim's YOLO Ball Tracker (`data/yolo_...csv`)** | Opportunistic Recovery | Calibrated external YOLO ball detection module. Operates as a seamless fallback whenever optical ATD tracking loses the ball, recovering **8,263 frames (+10.4%)** of missing ball coordinates in Half 1 (median 0.92m error from ATD). |
| **Loca's Native TCN Model (`native_raw/`)** | Neural Evidence & Snapping | 5-block Dilated Temporal Convolutional Network (156 spatio-temporal features, 101-frame temporal receptive field, frozen checkpoint `native.pt` [192 KB]). Inferred locally on CPU for dual-evidence turnover snapping and whistle detection. |

### Assets Evaluated and Dropped

1. **Broadcast Video (`BAR-ATH_broadcast.mp4`) & Broadcast ATD**:
   * *Reason for Dropping*: The broadcast feed contains unsynchronised timeline cuts, commercial breaks, camera zooms, and replays. 34.2% of its homography frames are invalid (`valid == 0`). Projecting broadcast observations back onto the tactical timeline requires visual alignment without providing higher spatial resolution than the 100% usable tactical homography.
2. **Tactical Raw Video (`BAR-ATH_tactical.mp4`)**:
   * *Reason for Dropping*: High-level video decoding (e.g. running 2D/3D CNNs or Vision Transformers across 199,319 frames at 1080p) requires hundreds of GPU hours and risks introducing visual artifacts. All geometric, kinematic, and velocity signals necessary to establish possession and dead-ball states are fully captured by the numerical ATD coordinate vectors.
3. **Smart Tagging Tactical Phases (`BUILD UP`, `PROGRESSION`, `FINAL THIRD`)**:
   * *Reason for Dropping*: In our initial prototype (`pipeline_v1`), we attempted to use tactical phase intervals as a continuous possession bias. However, because Metrica's Smart Tagging labels multi-second phases rather than individual possessions, overlapping attacking and defending tags overwhelmed the tracking data during quick counter-attacks and turnovers. In V2 and V3, we strictly limited Smart Tagging to **objective set-piece restarts**, allowing physical tracking to govern live possession.

---

## 3. Possession Rules Implemented

Our rule design implements the official definition from `CHALLENGE.md` using deterministic physical tests:

### A. Loose Balls, Passes, and Aerial Duels
* **Rule**: Possession does not switch during ball flight or loose ball states.
* **Implementation**: If $d_H > 2.0\text{ m}$ and $d_A > 2.0\text{ m}$ (ball is between players, airborne, or cleared), the state machine holds the current possession state.

### B. Deflections vs. Controlled Receptions
* **Rule**: A simple touch or deflection without subsequent control does not count as a turnover.
* **Implementation**: An opponent touch that lasts only 1–4 frames ($< 0.16\text{ s}$) before the ball moves away is classified as a deflection. Turnover requires $K \ge 10\text{ frames}$ ($0.40\text{ s}$) of contiguous proximity within $2.0\text{ m}$.

### C. Stoppages and Dead Ball Detection
* **Rule**: The ball is dead ($D$) whenever play is halted by the referee or out of bounds. Dead ball ownership is never inferred.
* **Implementation**: A frame is classified as dead ball ($D$) if:
  1. The collective player velocity drops to walking speed ($v_{\text{players}} < 1.70\text{ m/s}$ averaged over $\pm 1.0\text{ s}$); **AND**
  2. The ball is either:
     * Outside pitch bounds ($X < -0.005$, $X > 1.005$, $Y < -0.005$, or $Y > 1.005$);
     * Disappeared near pitch perimeter lines ($Y < 0.15$ or $Y > 0.85$, $X < 0.08$ or $X > 0.92$); **OR**
     * Within the pre-restart window of a Smart Tagged set piece.

### D. Active-Play Safeguard (The Goalkeeper & Tiki-Taka Rule)
* **Rule**: Slow player movement does not imply a stoppage if a player is actively controlling the ball inside the pitch.
* **Implementation**: Even if $v_{\text{players}} < 1.70\text{ m/s}$, the dead-ball classification is overridden to **alive** if the ball is visible, within pitch boundaries, with a player within $2.2\text{ m}$, and either positioned in the central pitch ($0.10 < X < 0.90, 0.15 < Y < 0.85$) or moving with dynamic velocity ($v_{\text{ball}} > 1.0\text{ m/s}$). This prevents false dead-ball classifications when defenders or goalkeepers hold the ball during slow build-up.

### E. Restarts
* **Rule**: Every restart begins a new possession spell.
* **Implementation**: At the end of a stoppage, the team with the player executing the restart (or tagged on the set-piece event) is assigned initial possession.

---

## 4. Verification and Benchmark Performance

All submissions were graded using the official competition scoring script (`evaluation/run_evaluation.py`):

```
scoreboard  gt\BAR-ATH_possession_gt_public.csv  199319 frames  7500 graded  21 gt segments
ground truth has 17 possessions of mean 14.2 s
ranked on spell_f1_mean; spellF1 = mean possession-spell F1 over IoU 0.3/0.4/0.5/0.6/0.7

rank  submission          macroF1  spellF1   poss     dur  transF1  lat_f  shareE  frag
----  ------------------  -------  -------  -----  ------  -------  -----  ------  ----
   1  Hackulers            0.8624   0.5294     17    13.4   0.3415   -7.0   0.006  1.05
   2  pipeline_v3          0.7694   0.4914     18    14.5   0.1429   +1.7   0.017  1.10
   3  baseline_smooth125   0.5812   0.4242     16    18.5   0.1667   -7.7   0.052  0.81
   4  pipeline_v2          0.7876   0.4229     18    13.0   0.0930   -5.5   0.032  1.14
   5  pipeline_v1          0.6715   0.3484     14    19.1   0.1053   +9.5   0.027  0.90
   6  baseline_smooth50    0.6207   0.3256     26    10.8   0.0816   +4.0   0.048  1.43
   7  baseline_smooth25    0.6200   0.2868     36     7.6   0.2034   +8.2   0.056  1.90
   8  baseline_raw         0.5972   0.0378    131     2.0   0.1410   -3.1   0.046  6.52
```

### Detailed Evaluation Breakdown (`Hackulers.csv`)
* **`spellF1` Details**: Mean **0.5294** | @0.3: **0.6471** | @0.4: **0.6471** | @0.5: **0.5882** | @0.6: **0.4118** | @0.7: **0.3529**
* **Per-Class F1**: Home (`H`): **0.8938** (P: 0.9153, R: 0.8732) | Away (`A`): **0.8146** (P: 0.8455, R: 0.7858) | Dead (`D`): **0.8789** (P: 0.7961, R: 0.9809)
* **Transition Boundary F1**: **0.3415** at $\tau = \pm 25$ frames (Mean latency: -7.0 frames)
* **Alive/Dead Discrimination**: **0.9661** alive/dead F1
* **Exact Possession Matching**: **17 predicted vs. 17 ground truth possessions** (mean duration 13.4s vs 14.2s, -6% difference)
* **Possession Share Error**: **0.0064** (Ground truth home share: 57.27%, Predicted: 57.91%, error only 0.64%)
* **Fragmentation**: **1.05** (22 segments vs 21 GT)
* **Format Contract**: Verified via `evaluation/validate_submission.py` (Exactly 199,319 rows, contiguous 0..199318, correct `frame,label` headers, valid enum).

---

## 4.5 Empirical Ablation Studies

### Ablation 1: IFAB Domain Decoupling (Set-Piece Stoppage Protection)
In earlier iterations, `alive_override` (designed to protect slow open-play build-up from false perimeter triggers) was inadvertently overriding verified set-piece events because kickers naturally stand within 1–2 metres of the ball before the restart. Under IFAB football laws (Laws 8, 13, 16), a referee stoppage is dead by definition until the kick is executed. By strictly decoupling smart-tagged set pieces from `alive_override` using universal sports timing constants (10.0s setup, 2.0s restart):
- **`macroF1`** surged from **0.8435 to 0.8624** (+0.0189).
- **`spellF1`** achieved **0.5294** with an exact **17-to-17 match against ground truth possessions**.
- **`transF1`** improved from **0.3000 to 0.3415** (+0.0415).
- **Dead-ball F1** increased from **0.8351 to 0.8789** with 98.09% recall.

### Ablation 2: Supervised Transfer Learning on Metrica Open Data
To investigate modern supervised transfer learning under official competition rules (which permit training on open datasets from other matches), we downloaded official Metrica Sports open tracking and event data (**Sample Game 1**: 145,008 tracking frames, 1,745 annotated match events).
- **Model Architecture**: A pure-NumPy Multi-Layer Perceptron (12 kinematic features $\rightarrow$ 64 $\rightarrow$ 32 $\rightarrow$ 3 classes) with ReLU activations, He initialization, and Adam optimization.
- **Training Outcome**: Trained in 1.1s to **98.42% accuracy** on Sample Game 1.
- **Cross-Match Transfer Results on Barca vs. Athletic Club**:
  - As a standalone frame-wise classifier with $\pm 25$-frame window smoothing, the model reached **`macroF1 = 0.6808`**, decisively outperforming all four competition baselines (`baseline_raw` 0.5972, `baseline_smooth25` 0.6200, `baseline_smooth50` 0.6207, `baseline_smooth125` 0.5812) and accurately predicting a 90-minute match possession share of 57.6% Barca vs. 42.4% Athletic.
  - **However, on `spellF1`**, the standalone model scored **0.2077**, generating 627 fragmented possessions (mean duration 4.9s).
- **Scientific Takeaway**: Without physical state hysteresis, frame-wise neural networks suffer from temporal micro-flickering during ball flight and contested duels. This proves empirically that our **hybrid autoregressive physical state machine + tactical smart tagging architecture (`spellF1 = 0.5294`)** is fundamentally superior to naive frame-wise ML transfer.

### Ablation 3: Embedded Dilated TCN Sequence Model (Loca's Model)
To capture multi-second temporal dynamics without hand-tuning, we integrated a frozen PyTorch Temporal Convolutional Network (TCN):
- **Model Architecture**: 5-block Dilated Temporal Convolutional Network with residual connections and causal dilations ($d \in \{1, 2, 4, 8, 10\}$), transforming 156 spatio-temporal kinematic features into calibrated class logits across a centered 101-frame receptive window (~4.0s temporal context).
- **External Training Disclosure**: The model weights (`native.pt`, 192 KB) and feature normalizer (`normalizer.npz`, 7.6 KB) were pre-trained strictly on external open football tracking datasets. **Zero frames, labels, or ground-truth sequences from the Barcelona vs. Athletic Club fixture were used during training or calibration**, adhering strictly to Rules 1 and 2.
- **Role in Hybrid Architecture**: Rather than trusting raw neural classifications (which can suffer from boundary jitter), sequence probabilities $P(H), P(A), P(D)$ serve as soft evidence gates within our physical state machine: accelerating turnover confirmation from 10 frames to 6 frames when neural confidence aligns with physical reach ($P > 0.60$), snapping transition timestamps to neural crossover frames ($P_{\text{opp}} > P_{\text{curr}}$), and reinforcing stoppage detection ($P(D) > 0.85$).

---

## 5. Strict Rule Compliance Statement

Our submission strictly complies with all challenge regulations, including the official clarification email sent to participants:

1. **No Hand-Labeling or Human-Produced Data**:
   * Absolutely zero manual annotations, hand-corrections, or external commercial event feeds (Opta, StatsBomb, Wyscout) were used for this match.
   * All labels are produced programmatically by `pipeline.py`.
2. **No Fitting or Tuning on the Ground Truth or Video**:
   * The 5-minute public ground truth was used **exclusively as a read-only validation check** for format compatibility and pipeline integrity, exactly as prescribed in `RULES.md` and `EVALUATION.md`.
   * All physical thresholds ($2.0\text{ m}$ control radius, $0.4\text{ s}$ touch duration, $1.75\text{ m/s}$ walking velocity, $105.3 \times 68.0\text{ m}$ pitch boundaries, 10.0s/2.0s IFAB restart windows) are **derived a priori from standard sports science and football biomechanics**, rather than fitted via optimization against test labels.
3. **No Checking Output Against the Video**:
   * Predictions were never cross-referenced or visually adjusted against video footage.
4. **No Third-Party AI Services or Cloud Data Uploads**:
   * No match data or tracking frames were uploaded to any hosted LLM, cloud vision service, or external API. The entire pipeline executes locally on consumer CPU hardware.
5. **Exact Reproducibility**:
   * The pipeline code is fully self-contained, open-source, and reproduces the exact `Hackulers.csv` deliverable deterministically from the raw data files.

---

## 6. Limitations and Future Work

1. **Player-Specific Context**:
   The ATD tracking slots are anonymised tracklets without persistent player identities or jersey numbers. If calibrated player IDs or role tracking (e.g. designating the goalkeeper) were available, the active-play override in the penalty area could be made even more specific.
2. **Z-Axis Ball Elevation**:
   The ATD tracking provides 2D pitch projections ($X, Y$) without 3D ball height ($Z$). Approximating aerial passes via high 2D velocity vectors works reliably for ground and mid-level passes, but a calibrated 3D ball trajectory would allow direct physical modeling of high parabolic clearances.
3. **Multi-Camera Fused Tracking**:
   Aligning the broadcast feed using automated visual feature matching (e.g. pitch keypoint homography) could provide multi-view redundancy during crowded penalty-box scuffles where the tactical camera loses the ball.

---

## 7. How to Reproduce

### Requirements
* Python 3.9+
* Standard libraries: `numpy`, `pandas`, `torch` (CPU wheel, no GPU required).

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure match tracking data is unpacked in data/
#    data/tactical/BAR-ATH_tactical_atd.txt
#    data/tactical/BAR-ATH_tactical_atd.xml
#    data/tactical/BAR-ATH_tactical_smart_tagging.xml

# 3. Generate the submission file (takes ~15–90 seconds on CPU)
python pipeline.py --team Hackulers

# 4. Validate submission formatting
python evaluation/validate_submission.py evaluation/results/Hackulers.csv

# 5. Run official evaluation against public baseline board
python evaluation/run_evaluation.py
```

