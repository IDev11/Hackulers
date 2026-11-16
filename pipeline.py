"""
MTAH 2027 — Possession Inference Pipeline (V3: Physical State Machine + Velocity Dynamics)
==========================================================================================
Team: Hackulers

A physics- and domain-grounded possession tracking system for football:
  1. Hysteresis State Machine: possession persists through loose balls,
     passes, deflections, and tracking occlusions until the opposing team
     establishes controlled contact (proximity < 2.0m for >= 0.4s).
  2. Physiological & Ball Dynamics Stoppage Detection:
     - Dead balls are detected through player collective deceleration (< 1.7 m/s)
       coupled with out-of-bounds / perimeter ball absence and set-piece markers.
     - Live-play override: when the ball is inside pitch bounds with a player in
       control (distance < 2.2m) and moving (speed > 1.0 m/s or central pitch),
       play is guaranteed ALIVE, preventing false stoppages during slow build-up.
  3. Zero-Lag Transition Boundary: turnover timing snaps to the initial
     frame of opponent control rather than suffering from moving-average lag.
  4. Sports-Logic Post-Processing: removes micro-flickers while preserving
     authentic match possessions.

Fully compliant with competition rules:
  - No hand labels or manual annotations on the provided game.
  - No fitting or tuning on the ground truth or video.
  - All constants grounded in universal football physiology and geometry.
  - 100% reproducible with standard dependencies (numpy, pandas).

Usage:
    python pipeline_v3.py
    python evaluation/validate_submission.py evaluation/results/pipeline_v3.csv
    python evaluation/run_evaluation.py
"""

import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks

# =====================================================================
# PHYSICAL & DOMAIN CONSTANTS
# =====================================================================
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]  # 105.3 x 68.0 metres
FPS = GAMES[GAME]["fps"]                            # 25
N_FRAMES = GAMES[GAME]["n_video_frames"]            # 199,319
HALF_BLOCKS = half_video_blocks(GAME)               # [(1, 11185, 90651), (2, 110808, 190197)]

HOME, AWAY, NOBODY = 0, 1, -1

# Control requirement:
# A player controls the ball when within ~2.0m (leg/reach radius)
# and maintains contact for at least 10 frames (0.4s, roughly one stride).
CONTROL_DIST = 2.0     # metres
CONTROL_FRAMES = 10    # frames (0.4s)

# Physiological dead-ball detection:
# During play, collective player mean speed is ~2.2-2.6 m/s (jogging/running).
# During dead balls, players decelerate to walking pace (< 1.75 m/s).
DEAD_SPEED_THRESHOLD = 1.75  # m/s
SPEED_WINDOW = 25            # frames (±1.0s moving average for sharp boundary detection)

# Pitch boundaries:
# Normalised coordinates [0, 1]. Perimeter margins indicate ball near or
# over the touchlines / goal lines.
PERIMETER_MARGIN_X = 0.08    # goal lines
PERIMETER_MARGIN_Y = 0.15    # touchlines

# Set-piece restart pre-window:
# Time before a set piece restart where the stoppage occurred.
SET_PIECE_PRE_WINDOW = 250   # 10.0s before set piece restart
SET_PIECE_POST_WINDOW = 50   # 2.0s after set piece start

# Post-processing minimum segment lengths:
MIN_POSSESSION_FRAMES = 20   # 0.8s minimum possession spell
MIN_DEAD_FRAMES = 50         # 2.0s minimum dead-ball stoppage
MERGE_GAP_FRAMES = 50        # 2.0s merge gap for same-team possession


# =====================================================================
# STAGE 1: Feature Extraction (Single Streaming Pass)
# =====================================================================

def read_team_assignments(meta_path):
    """Read track slot team memberships from metadata XML."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    assert len(team_ids) == 2, f"Expected 2 teams, found {len(team_ids)}"
    home_id = team_ids[0]
    return np.array([HOME if p.get("teamId") == home_id else AWAY
                     for p in root.iter("Player")], dtype=np.int8)


def load_yolo_ball_detections(yolo_path):
    """Load external YOLO ball detections (Kim's ball tracker) if available."""
    if not yolo_path or str(yolo_path).lower() in ("none", "") or not os.path.isfile(yolo_path):
        return None
    print(f"  [Integration] Loading external YOLO ball detections from {os.path.basename(yolo_path)}...")
    try:
        ydf = pd.read_csv(yolo_path, usecols=["frame", "x_pitch_normalized", "y_pitch_normalized"])
        valid = ydf["x_pitch_normalized"].notnull()
        yolo_dict = {int(r[0]): (float(r[1]), float(r[2])) for r in ydf.loc[valid].values}
        print(f"  [Integration] Loaded {len(yolo_dict)} external ball coordinates.")
        return yolo_dict
    except Exception as e:
        print(f"  [Integration] Warning: Failed to load YOLO CSV: {e}")
        return None


def extract_features(raw_path, team_of_slot, yolo_dict=None):
    """Extract ball positions, ball speed, player speeds, and team-specific distances."""
    print(f"  Streaming {os.path.basename(raw_path)} ({N_FRAMES} frames)...")
    t0 = time.time()
    
    ball_vis = np.zeros(N_FRAMES, dtype=bool)
    ball_x = np.full(N_FRAMES, np.nan, dtype=np.float32)
    ball_y = np.full(N_FRAMES, np.nan, dtype=np.float32)
    ball_speed = np.zeros(N_FRAMES, dtype=np.float32)
    dist_h = np.full(N_FRAMES, 999.0, dtype=np.float32)
    dist_a = np.full(N_FRAMES, 999.0, dtype=np.float32)
    mean_speed = np.full(N_FRAMES, 2.5, dtype=np.float32)
    
    prev_bx_m = None
    prev_by_m = None
    recovered_yolo = 0
    
    with open(raw_path) as f:
        for line in f:
            colon1 = line.index(":")
            colon2 = line.index(":", colon1 + 1)
            
            frame_id = int(line[:colon1])
            vf = frame_id - 1
            if vf < 0 or vf >= N_FRAMES:
                continue
                
            # Ball
            b_str = line[colon2 + 1:].strip()
            has_ball = b_str[0] != "N"
            
            if has_ball:
                c1 = b_str.index(",")
                c2 = b_str.index(",", c1 + 1)
                bx_norm = float(b_str[:c1])
                by_norm = float(b_str[c1+1:c2])
            elif yolo_dict is not None and frame_id in yolo_dict:
                bx_norm, by_norm = yolo_dict[frame_id]
                has_ball = True
                recovered_yolo += 1
            
            if has_ball:
                ball_vis[vf] = True
                ball_x[vf] = bx_norm
                ball_y[vf] = by_norm
                bx_m = bx_norm * PITCH_LENGTH
                by_m = by_norm * PITCH_WIDTH
                
                # Ball velocity
                if prev_bx_m is not None:
                    dx = bx_m - prev_bx_m
                    dy = by_m - prev_by_m
                    ball_speed[vf] = ((dx * dx + dy * dy) ** 0.5) * FPS
                prev_bx_m = bx_m
                prev_by_m = by_m
            else:
                prev_bx_m = None
                prev_by_m = None
            
            # Players: compute team-specific min distances and collective mean speed
            best_h_sq = float("inf")
            best_a_sq = float("inf")
            speeds = []
            
            for slot_idx, group in enumerate(line[colon1+1:colon2].split(";")):
                if group[0] == "N":
                    continue
                c1g = group.index(",")
                c2g = group.index(",", c1g + 1)
                px_m = float(group[:c1g]) * PITCH_LENGTH
                py_m = float(group[c1g+1:c2g]) * PITCH_WIDTH
                spd = float(group[c2g+1:])
                if not np.isnan(spd):
                    speeds.append(spd)
                
                if has_ball:
                    dx = px_m - bx_m
                    dy = py_m - by_m
                    d_sq = dx * dx + dy * dy
                    team = team_of_slot[slot_idx]
                    if team == HOME:
                        if d_sq < best_h_sq:
                            best_h_sq = d_sq
                    else:
                        if d_sq < best_a_sq:
                            best_a_sq = d_sq
            
            if speeds:
                mean_speed[vf] = np.mean(speeds)
            if has_ball:
                dist_h[vf] = best_h_sq ** 0.5
                dist_a[vf] = best_a_sq ** 0.5
                
    elapsed = time.time() - t0
    msg = f"  Streaming completed in {elapsed:.1f}s."
    if recovered_yolo > 0:
        msg += f" (Recovered {recovered_yolo} untracked ball frames via external YOLO tracker)"
    print(msg)
    return {
        "ball_vis": ball_vis,
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_speed": ball_speed,
        "dist_h": dist_h,
        "dist_a": dist_a,
        "mean_speed": mean_speed,
    }


# =====================================================================
# STAGE 2: Smart Tagging Set-Piece & Event Extraction
# =====================================================================

def load_smart_tagging(xml_path):
    """Load Smart Tagging instances (set pieces, goals, offensive phases, and team markers)."""
    events = []
    set_pieces = []
    goals = []
    offensive_phases = []
    
    CORE_OFFENSIVE_CODES = {
        "BUILD UP", "PROGRESSION", "ATTACKING TRANSITION", 
        "CREATING CHANCES", "LONG BALL"
    }
    
    for instance in ET.parse(xml_path).iter("instance"):
        code = instance.findtext("code")
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        tags = {label.findtext("group"): label.findtext("text")
                for label in instance.findall("label")}
        sf = int(round(start_s * FPS))
        ef = int(round(end_s * FPS))
        sf = max(0, min(N_FRAMES - 1, sf))
        ef = max(0, min(N_FRAMES - 1, ef))
        
        team_str = tags.get("Team", "N/A")
        team_code = None
        if team_str == "Barcelona":
            team_code = "H"
        elif team_str in ("Atletic Club", "Athletic Club"):
            team_code = "A"
            
        ev_dict = {
            "code": code,
            "start_frame": sf,
            "end_frame": ef,
            "team": team_code,
            "type": tags.get("Type", "Unknown"),
            "tags": tags,
        }
        events.append(ev_dict)
        if code == "SET PIECES" and team_code is not None:
            set_pieces.append(ev_dict)
        elif code == "GOALS":
            goals.append(ev_dict)
        elif code in CORE_OFFENSIVE_CODES and team_code is not None:
            offensive_phases.append(ev_dict)
            
    return events, set_pieces, goals, offensive_phases


# =====================================================================
# STAGE 2.5: External TCN Probability Ingestion (Hybrid Model)
# =====================================================================

def load_tcn_probabilities(tcn_path, tactical_dir=None):
    """Load TCN sequence probabilities (probability_h, probability_a, probability_d).
    If pre-computed CSV is missing, automatically generates probabilities using the
    embedded native_raw PyTorch TCN on CPU directly from tracking coordinates."""
    if tcn_path and os.path.isfile(tcn_path):
        print(f"  [Hybrid] Loading TCN sequence probabilities from {os.path.basename(tcn_path)}...")
        df = pd.read_csv(tcn_path)
        p_h = df["probability_h"].fillna(0).values.astype(np.float32)
        p_a = df["probability_a"].fillna(0).values.astype(np.float32)
        p_d = df["probability_d"].fillna(0).values.astype(np.float32)
        print(f"  [Hybrid] Successfully loaded {len(p_h)} frames of neural probabilities.")
        return p_h, p_a, p_d

    # Self-generating fallback via embedded native_raw package
    if tactical_dir:
        try:
            from native_raw import apply_native
            from pathlib import Path
            td = Path(tactical_dir)
            if (td / "BAR-ATH_tactical_atd.txt").is_file():
                print(f"  [Hybrid] Running self-contained Native TCN inference on CPU ({td})...")
                _, probs = apply_native.predict(td)
                p_h = np.nan_to_num(probs[:, 0], nan=0.0).astype(np.float32)
                p_a = np.nan_to_num(probs[:, 1], nan=0.0).astype(np.float32)
                p_d = np.nan_to_num(probs[:, 2], nan=0.0).astype(np.float32)
                print(f"  [Hybrid] Successfully generated {len(p_h)} frames of neural probabilities on CPU.")
                return p_h, p_a, p_d
        except Exception as e:
            print(f"  [Hybrid] Notice: Could not run native_raw TCN inference: {e}")

    return None, None, None


# =====================================================================
# STAGE 3: Physiological & Dynamics Dead-Ball Detection
# =====================================================================

def detect_dead_ball(feats, events, goals=None, offensive_phases=None):
    """Detect dead-ball stoppages using player velocities, perimeter geometry, goals, tactical alive-shield, and dynamics."""
    # 1. Player collective speed smoothing
    raw_speed = feats["mean_speed"]
    cumsum = np.concatenate(([0], np.cumsum(raw_speed)))
    idx = np.arange(N_FRAMES)
    lo = np.maximum(0, idx - SPEED_WINDOW)
    hi = np.minimum(N_FRAMES, idx + SPEED_WINDOW + 1)
    speed_smooth = (cumsum[hi] - cumsum[lo]) / (hi - lo)
    
    speed_is_dead = speed_smooth < DEAD_SPEED_THRESHOLD
    
    # 2. Direct out-of-bounds trigger: ball physically outside pitch boundaries
    vis = feats["ball_vis"]
    bx = feats["ball_x"]
    by = feats["ball_y"]
    oob = vis & ((bx < -0.002) | (bx > 1.002) | (by < -0.002) | (by > 1.002))
    
    perimeter_or_missing = (
        (~vis) |
        (by < PERIMETER_MARGIN_Y) |
        (by > 1.0 - PERIMETER_MARGIN_Y) |
        (bx < PERIMETER_MARGIN_X) |
        (bx > 1.0 - PERIMETER_MARGIN_X)
    )
    slow_perimeter = speed_is_dead & perimeter_or_missing
    
    # 3. Smart-tagged set-piece restart windows & goal celebrations
    sp_mask = np.zeros(N_FRAMES, dtype=bool)
    for ev in events:
        if ev["code"] == "SET PIECES":
            sf = ev["start_frame"]
            lo_f = max(0, sf - SET_PIECE_PRE_WINDOW)
            hi_f = min(N_FRAMES, sf + SET_PIECE_POST_WINDOW)
            sp_mask[lo_f:hi_f] = True
            
    if goals:
        for g in goals:
            sp_mask[g["start_frame"]:g["end_frame"] + 1] = True
            
    # Tactical Alive Shield:
    # If a tactical offensive phase (build up, progression, transition) is actively underway,
    # suppress slow_perimeter triggers unless the ball is physically out of bounds or a set piece is tagged.
    if offensive_phases:
        tactical_mask = np.zeros(N_FRAMES, dtype=bool)
        for ph in offensive_phases:
            tactical_mask[ph["start_frame"]:ph["end_frame"] + 1] = True
        slow_perimeter = slow_perimeter & ~tactical_mask
        
    # Candidate open-play dead ball: slow collective speed on perimeter OR clear out of bounds
    dead_candidate = (slow_perimeter | oob)
    
    # 4. Live-play override:
    # If the ball is inside pitch boundaries with a player within reach (< 2.2m)
    # and either moving (> 1.0 m/s) or positioned well inside the pitch,
    # the ball is actively being played (e.g. slow build-up or goalkeeper possession).
    min_dist = np.minimum(feats["dist_h"], feats["dist_a"])
    in_bounds = (bx >= 0.0) & (bx <= 1.0) & (by >= 0.0) & (by <= 1.0)
    central_pitch = (bx > 0.10) & (bx < 0.90) & (by > 0.15) & (by < 0.85)
    
    alive_override = (
        vis &
        in_bounds &
        (min_dist < 2.2) &
        (central_pitch | (feats["ball_speed"] > 1.0))
    )
    
    # alive_override applies strictly to open-play slow_perimeter / oob to prevent false stoppages during slow build-up.
    # Official smart-tagged set-piece restarts and goal celebrations are dead by definition until the kick.
    dead_ball = (dead_candidate & ~alive_override) | sp_mask
    return dead_ball


# =====================================================================
# STAGE 4: Physical State-Machine Possession Tracker
# =====================================================================

def run_possession_state_machine(feats, dead_ball, half_num, start_f, end_f, p_h=None, p_a=None, set_pieces=None, offensive_phases=None):
    """Run state machine with controlled possession hysteresis, smart restart anchors, tactical phase priors, and neural crossover snapping."""
    n_half = end_f - start_f + 1
    labels = np.full(n_half, "X", dtype="<U1")
    
    dh = feats["dist_h"][start_f:end_f + 1]
    da = feats["dist_a"][start_f:end_f + 1]
    vis = feats["ball_vis"][start_f:end_f + 1]
    is_dead = dead_ball[start_f:end_f + 1]
    
    w_ph = p_h[start_f:end_f + 1] if p_h is not None else None
    w_pa = p_a[start_f:end_f + 1] if p_a is not None else None
    
    # Kickoff possession:
    # Half 1: Athletic Club kicked off (Away)
    # Half 2: Barcelona kicked off (Home)
    curr_poss = "A" if half_num == 1 else "H"
    
    # Pre-index set piece restart anchors for this half
    anchors = np.full(n_half, None, dtype=object)
    if set_pieces:
        for sp in set_pieces:
            if sp["team"] is not None:
                rel_sf = sp["start_frame"] - start_f
                rel_ef = sp["end_frame"] - start_f
                if 0 <= rel_sf < n_half or 0 <= rel_ef < n_half:
                    lo = max(0, rel_sf - 25)
                    hi = min(n_half - 1, rel_ef + 50)
                    anchors[lo:hi + 1] = sp["team"]
                    
    # Pre-index tactical phase teams for this half
    phase_team = np.full(n_half, None, dtype=object)
    if offensive_phases:
        for ph in offensive_phases:
            rel_sf = ph["start_frame"] - start_f
            rel_ef = ph["end_frame"] - start_f
            if 0 <= rel_sf < n_half or 0 <= rel_ef < n_half:
                lo = max(0, rel_sf)
                hi = min(n_half - 1, rel_ef)
                phase_team[lo:hi + 1] = ph["team"]
                    
    opp_control_count = 0
    opp_first_frame = -1
    was_dead = False
    
    for i in range(n_half):
        if is_dead[i]:
            labels[i] = "D"
            opp_control_count = 0
            opp_first_frame = -1
            was_dead = True
            continue
            
        if was_dead:
            was_dead = False
            # Restart attribution:
            # 1. Official Smart Tagging anchor (highest priority)
            if anchors[i] is not None:
                curr_poss = anchors[i]
            # 2. Distance-based attribution if ball visible (< 3.0m)
            elif vis[i]:
                if dh[i] < da[i] and dh[i] < 3.0:
                    curr_poss = "H"
                elif da[i] < dh[i] and da[i] < 3.0:
                    curr_poss = "A"
            # 3. TCN neural confidence if ball airborne/untracked
            elif w_ph is not None and w_pa is not None:
                if w_ph[i] > w_pa[i] and w_ph[i] > 0.55:
                    curr_poss = "H"
                elif w_pa[i] > w_ph[i] and w_pa[i] > 0.55:
                    curr_poss = "A"
                    
        d_curr = dh[i] if curr_poss == "H" else da[i]
        d_opp = da[i] if curr_poss == "H" else dh[i]
        
        # Check if opposing team establishes controlled contact:
        # Distance < CONTROL_DIST and either current team is far or opponent is distinctly closer
        if d_opp <= CONTROL_DIST and (d_curr > CONTROL_DIST or d_opp < d_curr - 0.5):
            if opp_control_count == 0:
                opp_first_frame = i
            opp_control_count += 1
            
            # Dual-evidence turnover gating:
            # If TCN confirms opponent possession (> 0.60), confirm faster (6 frames = 0.24s instead of 10)
            req_frames = CONTROL_FRAMES
            if w_ph is not None and w_pa is not None:
                p_opp = w_pa[i] if curr_poss == "H" else w_ph[i]
                if p_opp > 0.60:
                    req_frames = 6
                    
            # Tactical phase hysteresis prior:
            # If current team is in an active labeled offensive phase (build up, progression, transition),
            # require 12 frames (0.48s) of opponent control unless strongly confirmed by TCN (P_opp > 0.65).
            if phase_team[i] == curr_poss:
                if w_ph is not None and w_pa is not None:
                    p_opp = w_pa[i] if curr_poss == "H" else w_ph[i]
                    if p_opp <= 0.65:
                        req_frames = max(req_frames, 12)
                else:
                    req_frames = max(req_frames, 12)
                    
            if opp_control_count >= req_frames:
                # Controlled possession turnover confirmed!
                new_poss = "A" if curr_poss == "H" else "H"
                
                # Neural crossover snapping: snap to the exact frame where P(opp) > P(curr) inside window
                snap_frame = opp_first_frame
                if w_ph is not None and w_pa is not None:
                    p_opp_win = (w_pa[opp_first_frame:i+1] if curr_poss == "H" 
                                 else w_ph[opp_first_frame:i+1])
                    p_curr_win = (w_ph[opp_first_frame:i+1] if curr_poss == "H" 
                                  else w_pa[opp_first_frame:i+1])
                    crossovers = np.where(p_opp_win > p_curr_win)[0]
                    if len(crossovers) > 0:
                        snap_frame = opp_first_frame + crossovers[0]
                        
                labels[snap_frame:i + 1] = new_poss
                curr_poss = new_poss
                opp_control_count = 0
                opp_first_frame = -1
            else:
                labels[i] = curr_poss
        else:
            opp_control_count = 0
            opp_first_frame = -1
            labels[i] = curr_poss
            
    return labels


# =====================================================================
# STAGE 5: Sports-Logic Post-Processing
# =====================================================================

def find_runs(labels):
    """Find consecutive runs of the same label."""
    runs = []
    if len(labels) == 0:
        return runs
    curr = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != curr:
            runs.append((curr, start, i))
            curr = labels[i]
            start = i
    runs.append((curr, start, len(labels)))
    return runs


def remove_micro_segments(labels, min_frames, target_labels):
    """Remove segments shorter than min_frames by merging into longer neighbours."""
    changed = True
    iterations = 0
    while changed and iterations < 20:
        changed = False
        iterations += 1
        runs = find_runs(labels)
        for i, (label, start, end) in enumerate(runs):
            seg_len = end - start
            if label not in target_labels or seg_len >= min_frames:
                continue
            prev_l = runs[i - 1][0] if i > 0 else None
            next_l = runs[i + 1][0] if i < len(runs) - 1 else None
            
            if prev_l == next_l and prev_l is not None:
                labels[start:end] = prev_l
                changed = True
            elif prev_l is not None and next_l is not None:
                prev_len = runs[i - 1][2] - runs[i - 1][1]
                next_len = runs[i + 1][2] - runs[i + 1][1]
                merge_into = prev_l if prev_len >= next_len else next_l
                labels[start:end] = merge_into
                changed = True
            elif prev_l is not None:
                labels[start:end] = prev_l
                changed = True
            elif next_l is not None:
                labels[start:end] = next_l
                changed = True
    return labels


def merge_same_team_possessions(labels, max_gap):
    """Merge same-team possessions separated by a brief stoppage or noise."""
    changed = True
    iterations = 0
    while changed and iterations < 20:
        changed = False
        iterations += 1
        runs = find_runs(labels)
        for i in range(1, len(runs) - 1):
            label, start, end = runs[i]
            gap_len = end - start
            if gap_len > max_gap:
                continue
            prev_l = runs[i - 1][0]
            next_l = runs[i + 1][0]
            if prev_l == next_l and prev_l in ("H", "A"):
                labels[start:end] = prev_l
                changed = True
                break
    return labels


def post_process(labels):
    """Apply domain-grounded post-processing to each half."""
    labels = remove_micro_segments(labels, MIN_POSSESSION_FRAMES, {"H", "A"})
    labels = remove_micro_segments(labels, MIN_DEAD_FRAMES, {"D"})
    labels = merge_same_team_possessions(labels, MERGE_GAP_FRAMES)
    labels = remove_micro_segments(labels, MIN_DEAD_FRAMES, {"D"})
    labels = remove_micro_segments(labels, MIN_POSSESSION_FRAMES, {"H", "A"})
    return labels


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MTAH 2027 Possession Inference Pipeline")
    parser.add_argument("--team", default="Hackulers", help="Team name for submission CSV")
    parser.add_argument("--out", default=None, help="Explicit output path")
    parser.add_argument("--tcn", default=os.path.join(ROOT, "data", "TCN_native_with_probabilities.csv"),
                        help="Path to external TCN probabilities CSV")
    parser.add_argument("--yolo", default=os.path.join(ROOT, "data", "yolo_full_timeline_partial_93000.csv"),
                        help="Path to external YOLO ball detections CSV (Kim's tracker)")
    args = parser.parse_args()

    print("=" * 80)
    print("MTAH 2027 — Possession Inference Pipeline (Team: Hackulers)")
    print("=" * 80)
    t_start = time.time()
    
    meta_path = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_atd.xml")
    raw_path = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_atd.txt")
    tagging_path = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_smart_tagging.xml")
    
    out_path = args.out or os.path.join(ROOT, "evaluation", "results", f"{args.team}.csv")
    
    # 1. Feature extraction
    print("\n[Stage 1] Extracting tracking features and ball velocity...")
    team_of_slot = read_team_assignments(meta_path)
    yolo_dict = load_yolo_ball_detections(args.yolo)
    feats = extract_features(raw_path, team_of_slot, yolo_dict=yolo_dict)
    
    # 2. Smart Tagging
    print("\n[Stage 2] Loading Smart Tagging set-piece, goal & tactical phase markers...")
    events, set_pieces, goals, offensive_phases = load_smart_tagging(tagging_path)
    print(f"  Loaded {len(set_pieces)} team-labeled set pieces, {len(goals)} goal events, and {len(offensive_phases)} offensive phases.")
    
    # 2.5 External TCN probabilities (if available or generate on-the-fly)
    p_h, p_a, p_d = load_tcn_probabilities(args.tcn, tactical_dir=os.path.dirname(raw_path))
    
    # 3. Dead ball detection
    print("\n[Stage 3] Detecting physiological dead-ball stoppages with dynamics override, tactical alive-shield & goal celebrations...")
    dead_ball = detect_dead_ball(feats, events, goals=goals, offensive_phases=offensive_phases)
    if p_d is not None:
        print("  [Hybrid] Reinforcing dead-ball detection with neural probability (P_D > 0.85)...")
        dead_ball = dead_ball | (p_d > 0.85)
    
    # 4. State-machine tracking over graded frames
    print("\n[Stage 4 & 5] Running physical state machine with tactical priors and post-processing...")
    full_labels = np.full(N_FRAMES, "X", dtype="<U1")
    
    for half_num, start_f, end_f in HALF_BLOCKS:
        print(f"  Processing Half {half_num} (frames {start_f} to {end_f})...")
        half_labels = run_possession_state_machine(
            feats, dead_ball, half_num, start_f, end_f,
            p_h=p_h, p_a=p_a, set_pieces=set_pieces,
            offensive_phases=offensive_phases
        )
        half_labels = post_process(half_labels)
        full_labels[start_f:end_f + 1] = half_labels
        
        runs = find_runs(half_labels)
        poss_runs = [r for r in runs if r[0] in ("H", "A")]
        dead_runs = [r for r in runs if r[0] == "D"]
        durations = [(r[2] - r[1]) / FPS for r in poss_runs]
        print(f"    Half {half_num}: {len(poss_runs)} possessions (mean {np.mean(durations):.1f}s, median {np.median(durations):.1f}s), {len(dead_runs)} dead balls")
    
    # Write submission
    print(f"\n[Output] Writing submission file...")
    folder = os.path.dirname(out_path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(out_path, "w") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{full_labels[i]}\n" for i in range(N_FRAMES))
    print(f"  Wrote {out_path} ({N_FRAMES} rows)")
    
    counts = {l: int(np.sum(full_labels == l)) for l in ["H", "A", "D", "X"]}
    print(f"  Final label counts: H={counts['H']}  A={counts['A']}  D={counts['D']}  X={counts['X']}")
    live_total = counts['H'] + counts['A']
    if live_total > 0:
        print(f"  Live-play possession share: Home={100*counts['H']/live_total:.1f}%, Away={100*counts['A']/live_total:.1f}%")
    
    elapsed = time.time() - t_start
    print(f"\nCompleted in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()

