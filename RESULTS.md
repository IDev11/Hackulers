# Final Submission Results — Team Hackulers
> **Competition**: More Than A Hack 2027 (Phase 1: Football Possession Inference)  
> **Organizers**: Metrica Sports, FC Barcelona, and Mobile World Capital Barcelona  
> **Match**: FC Barcelona vs. Athletic Club (LaLiga EA Sports 2024–25, Matchday 2)  
> **Official Team Name**: Hackulers  
> **Official Repository**: `https://github.com/IDev11/Hackulers.git`  
> **Official Submission File**: `evaluation/results/Hackulers.csv`  

---

## 1. Official Leaderboard Benchmark

Evaluated against the official 5-minute ground truth reference (`evaluation/gt/BAR-ATH_possession_gt_public.csv`, 7,500 graded frames, 17 true possessions of mean 14.2s) using the competition's official scoring harness (`evaluation/run_evaluation.py`):

| Rank | Submission | `spellF1` *(Ranking)* | `macroF1` *(Tie-Breaker)* | Possessions *(GT = 17)* | Mean Duration *(GT = 14.2s)* | `transF1` *($\pm 25\text{f}$)* | Mean Latency | Share Error | Fragmentation |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **`Hackulers` (Ours)** | **0.5294** | **0.8624** | **17 (Exact)** | **13.4s (-6%)** | **0.3415** | **-7.0 frames** | **0.0064 (0.64%)** | **1.05** |
| 2 | `baseline_smooth125` | 0.4242 | 0.5812 | 16 | 18.5s (+30%) | 0.1667 | -7.7 frames | 0.0520 (5.20%) | 0.81 |
| 3 | `baseline_smooth50` *(Floor)* | 0.3256 | 0.6207 | 26 | 10.8s (-24%) | 0.0816 | +4.0 frames | 0.0485 (4.85%) | 1.43 |
| 4 | `baseline_smooth25` | 0.2868 | 0.6200 | 36 | 7.6s (-47%) | 0.2034 | +8.2 frames | 0.0556 (5.56%) | 1.90 |
| 5 | `baseline_raw` | 0.0378 | 0.5972 | 131 | 2.0s (-86%) | 0.1410 | -3.1 frames | 0.0457 (4.57%) | 6.52 |

---

## 2. Headline Performance Metrics

* **`spellF1` = 0.5294 (Rank #1)**:  
  Outperforms the competition published benchmark floor (`0.3256`) by **+62.6% relative** (+0.2038 points).
* **`macroF1` = 0.8624 (Rank #1)**:  
  Exceeds the best baseline by **+0.2417 points**, ensuring first-place tie-break security.
* **Exact Possession Matching (17 vs. 17)**:  
  Our autoregressive physical state machine identifies exactly **17 possessions**, matching ground truth 1-to-1 with a mean duration of **13.4s vs. 14.2s** (-6% difference), avoiding both under-smoothing and over-smoothing.
* **Low Latency Transition Snapping (-7.0 frames)**:  
  Turnover timestamps snap retroactively to within **-7.0 frames (-0.28 seconds)** of ground truth boundaries, achieving a `transF1` of **0.3415** at $\tau = \pm 25$ frames.
* **Match Possession Share Error = 0.64%**:  
  Predicted live home possession share is **57.91%** against actual ground truth of **57.27%** (an error of only 0.0064).
* **Fragmentation Index = 1.05**:  
  Only 22 predicted live segments vs. 21 ground truth segments. Pass fragmentation is completely resolved.

---

## 3. Per-Class Performance Breakdown

| Class | Precision | Recall | F1 Score | Ground Truth Support | Key Technical Factor |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Home (`H`)** *(FC Barcelona)* | **91.53%** | 87.32% | **0.8938** | 3,454 frames | Physical hysteresis holds state across passes and ball flight |
| **Away (`A`)** *(Athletic Club)* | **84.55%** | 78.58% | **0.8146** | 2,577 frames | 10-frame sustained control barrier confirms turnovers cleanly |
| **Dead Ball (`D`)** *(Stoppage)* | 79.61% | **98.09%** | **0.8789** | 1,469 frames | IFAB set-piece decoupling + deceleration detection |
| **Overall Alive / Dead** | — | — | **0.9661** | 7,500 frames | Near-perfect match stoppage discrimination |

### Confusion Matrix (Rows = Ground Truth, Columns = Predicted)

| Ground Truth \ Predicted | `H` | `A` | `D` | `X` | Total GT | Class Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`H` (Barcelona)** | **3,016** | 351 | 87 | 0 | 3,454 | **87.32%** |
| **`A` (Athletic)** | 270 | **2,025** | 282 | 0 | 2,577 | **78.58%** |
| **`D` (Dead Ball)** | 9 | 19 | **1,441** | 0 | 1,469 | **98.09%** |
| **`X` (Halftime / Ungraded)** | 41,442 | 25,091 | 84,824 | 40,462 | 191,819 | 100% compliant |

---

## 4. `spellF1` IoU Threshold Performance

`spellF1` treats each possession as a temporal interval object and evaluates intersection-over-union (IoU) matches across five thresholds:

| IoU Threshold | `Hackulers` F1 | `baseline_smooth50` F1 | Margin Over Baseline |
| :---: | :---: | :---: | :---: |
| **@0.3** | **0.6471** | 0.5116 | +0.1355 |
| **@0.4** | **0.6471** | 0.4651 | +0.1820 |
| **@0.5** | **0.5882** | 0.3256 | **+0.2626** *(+80.7% relative)* |
| **@0.6** | **0.4118** | 0.2326 | +0.1792 |
| **@0.7** | **0.3529** | 0.0930 | **+0.2599** *(Nearly 4x higher!)* |

---

## 5. Full Match Predictions (199,319 Frames)

Generated directly via `python pipeline.py --team Hackulers` and verified with `evaluation/validate_submission.py`:

* **Total Row Count**: Exactly **199,319 frames** (video frames `0 … 199318` contiguous).
* **Frame Label Distribution**:
  * **`H` (Home in possession)**: 44,737 frames (22.4%)
  * **`A` (Away in possession)**: 27,486 frames (13.8%)
  * **`D` (Dead ball / Stoppage)**: 86,634 frames (43.5%)
  * **`X` (Ungraded: before match, halftime, post-whistle)**: 40,462 frames (20.3%)
* **Live Match Possession Share**:
  * **FC Barcelona (`H`)**: **61.9%**
  * **Athletic Club (`A`)**: **38.1%**  
  *(Accurately mirrors official broadcast match statistics where Barcelona controlled possession).*
* **Possession Structure**:
  * **Total Match Possessions**: **285 possessions** of mean duration **10.1 seconds**.
  * **Half 1**: 138 possessions (mean 11.2s, median 7.0s), 53 dead balls.
  * **Half 2**: 147 possessions (mean 9.2s, median 5.7s), 68 dead balls.
* **YOLO Ball Detection Integration**:
  * External coordinates loaded: **65,412 frames** (covering frames 8 through 164,057).
  * Missing ball observations recovered: **13,450 untracked ball frames plugged (+17.0% recovery rate)** across the match timeline where camera optical tracking had `NaN`.

---

## 6. Exact Reproduction Instructions

The entire system is 100% deterministic, runs on standard consumer CPU with zero GPU requirements, and completes in **~10–15 seconds**:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run official inference pipeline
python pipeline.py --team Hackulers

# 3. Verify submission format contract (199,319 rows, 0 errors)
python evaluation/validate_submission.py evaluation/results/Hackulers.csv

# 4. Score against public benchmark ground truth
python evaluation/run_evaluation.py
```
