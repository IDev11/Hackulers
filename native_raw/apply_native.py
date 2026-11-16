"""Run the frozen Barcelona–Athletic Native TCN and export H/A/D scores."""

import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from .features import make_model_input
from .model import model_from_config, predict_half
from .temporal_features import blocks_for


ASSETS = Path(__file__).resolve().parent / "assets"
VIDEO_FRAMES = 199_319
HALVES = ((1, 11_185, 90_651), (2, 110_808, 190_197))
PITCH_M = (105.3, 68.0)
CLASSES = np.array(["H", "A", "D"])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_tactical_metadata(path):
    root = ET.parse(path).getroot()
    spec = root.find(".//DataFormatSpecification")
    fps = float(root.findtext(".//GlobalConfig/FrameRate"))
    require(fps == 25 and spec.get("separator") == ":" and spec.get("startFrame") == "1"
            and int(spec.get("endFrame")) == VIDEO_FRAMES, "Unexpected ATD time base")
    pitch_xml = [float(root.findtext(f".//FieldSize/{axis}")) for axis in ("Width", "Height")]
    require(pitch_xml == [105, 68], "Unexpected pitch metadata")
    frame_register, players, ball = list(spec)
    require(frame_register.get("name") == "frameCount" and players.get("separator") == ";"
            and ball.get("separator") == "," and [x.get("channelId") for x in ball] == ["x", "y", "s"],
            "Unexpected ATD register layout")

    score = root.find(".//Score")
    team_ids = [score.get("idLocalTeam"), score.get("idVisitingTeam")]
    teams = {team.get("id"): team.findtext("Name") for team in root.findall(".//Teams/Team")}
    require([teams[tid] for tid in team_ids] == ["Barcelona", "Atletic Club"], "Unexpected H/A mapping")
    player_team = {p.get("id"): p.get("teamId") for p in root.findall(".//Players/Player")}
    channels = {ch.get("id"): (ch.get("playerId"), ch.get("channelId"))
                for ch in root.findall(".//PlayerChannel")}
    slots, team_codes = [], []
    for group in players:
        linked = [channels[ref.get("playerChannelId")] for ref in group]
        require(group.get("separator") == "," and [code for _, code in linked] == ["x", "y", "s"]
                and len({pid for pid, _ in linked}) == 1, "Invalid player-channel group")
        pid = linked[0][0]
        slots.append(pid)
        team_codes.append(team_ids.index(player_team[pid]))
    require(len(slots) == len(set(slots)) == 46 and np.bincount(team_codes).tolist() == [24, 22],
            "Unexpected number of slots by team")
    return np.asarray(team_codes, dtype=np.int8)


def time_map(first, last):
    """Map each 10 Hz sample to ATD and each video frame back to its nearest sample."""
    grid = np.arange(2 * (last - first) // 5 + 1, dtype=np.int64)
    sampled = first + (5 * grid + 1) // 2
    offsets = np.arange(last - first + 1, dtype=np.int64)
    export_index = np.minimum((4 * offsets + 5) // 10, len(grid) - 1)
    require(np.array_equal(export_index[sampled - first], grid), "ATD/video round trip failed")
    return sampled, grid / 10.0, export_index


def to_centered_meters(points):
    xy = (points - 0.5) * np.array([PITCH_M[0], -PITCH_M[1]], dtype=np.float64)
    xy[~np.isfinite(points).all(-1)] = np.nan
    return xy.astype(np.float32)



def build_features(tactical, manifest, normalizer):
    player_team = read_tactical_metadata(tactical / "BAR-ATH_tactical_atd.xml")
    maps = [time_map(first, last) for _, first, last in HALVES]
    sample_frames = np.concatenate([sampled for sampled, _, _ in maps])
    sample_count = len(sample_frames)
    is_sampled = np.full(VIDEO_FRAMES, -1, dtype=np.int64)
    is_sampled[sample_frames] = np.arange(sample_count)
    ball_xy = np.empty((sample_count, 2), dtype=np.float64)
    players_xy = np.empty((sample_count, len(player_team), 2), dtype=np.float64)

    parsed = 0
    with (tactical / "BAR-ATH_tactical_atd.txt").open(encoding="utf-8-sig") as stream:
        for parsed, line in enumerate(stream, start=1):
            raw_frame, player_text, ball_text = line.strip().split(":")
            require(parsed <= VIDEO_FRAMES and int(raw_frame) == parsed, "Noncontiguous ATD frame IDs")
            sample = is_sampled[parsed - 1]
            if sample < 0:
                continue
            players = np.fromstring(player_text.replace(";", ","), sep=",").reshape(-1, 3)
            ball = np.fromstring(ball_text, sep=",")
            require(len(players) == len(player_team) and len(ball) == 3, f"Invalid ATD row {parsed}")
            ball_xy[sample] = ball[:2]
            players_xy[sample] = players[:, :2]
    require(parsed == VIDEO_FRAMES, "Incomplete ATD")

    tracking = {
        "frame_id": np.arange(sample_count, dtype=np.int64),
        "period": np.repeat([half for half, _, _ in HALVES], [len(m[0]) for m in maps]).astype(np.int8),
        "time_seconds": np.concatenate([times for _, times, _ in maps]),
        "within_half": np.ones(sample_count, dtype=bool),
        "ball_xy_m": to_centered_meters(ball_xy),
        "ball_available": np.isfinite(ball_xy).all(axis=1),
        "player_xy_m": to_centered_meters(players_xy),
        "player_team": player_team,
        "pitch_length_m": PITCH_M[0],
        "pitch_width_m": PITCH_M[1],
    }
    half_rows, _ = blocks_for(
        tracking["frame_id"], tracking["period"], tracking["time_seconds"], tracking["within_half"],
        manifest["temporal_feature_config"]["fps"],
    )
    second_half = len(maps[0][0])
    require(half_rows == [(0, second_half), (second_half, sample_count)], "Invalid temporal feature blocks")
    return make_model_input(tracking, manifest, normalizer), half_rows, maps


def predict(tactical_dir):
    """Return video-ordered labels and H/A/D probabilities, with NaN for X."""
    tactical = Path(tactical_dir)
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    with np.load(ASSETS / "normalizer.npz", allow_pickle=False) as normalizer:
        inputs, half_rows, maps = build_features(tactical, manifest, normalizer)

    checkpoint = torch.load(ASSETS / "native.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    require(checkpoint["arm"] == "native" and checkpoint["selected_epoch"] == 3
            and checkpoint["status"] == "complete" and config["input_channels"] == inputs.shape[1],
            "Expected the frozen Native checkpoint from epoch 3")
    model = model_from_config(config)
    model.load_state_dict(checkpoint["model_state"])

    labels = np.full(VIDEO_FRAMES, "X", dtype="<U1")
    probabilities = np.full((VIDEO_FRAMES, 3), np.nan, dtype=np.float64)
    for (_, first, last), (begin, end), (_, _, export) in zip(HALVES, half_rows, maps, strict=True):
        logits = predict_half(model, inputs[begin:end], radius=config["context_radius"])
        require(logits.shape == (end - begin, 3) and np.isfinite(logits).all(), "Invalid TCN logits")
        labels[first:last + 1] = CLASSES[logits.argmax(axis=1)[export]]
        probabilities[first:last + 1] = torch.from_numpy(logits).double().softmax(dim=1).numpy()[export]
    return labels, probabilities


def write_csv(stream, labels, probabilities):
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("frame", "label", "probability_h", "probability_a", "probability_d"))
    for frame, (label, scores) in enumerate(zip(labels, probabilities, strict=True)):
        values = ("", "", "") if label == "X" else tuple(f"{p:.10f}" for p in scores)
        writer.writerow((frame, label, *values))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tactical", type=Path, required=True, help="Directory containing the original ATD TXT and XML")
    parser.add_argument("--output", type=Path, required=True, help="Path for a new five-column CSV")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    labels, probabilities = predict(args.tactical)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stream = args.output.open("x", encoding="utf-8", newline="")
    try:
        with stream:
            write_csv(stream, labels, probabilities)
    except BaseException:
        args.output.unlink(missing_ok=True)
        raise
    print(f"Exported {len(labels):,} Native predictions to {args.output}")


if __name__ == "__main__":
    main()
