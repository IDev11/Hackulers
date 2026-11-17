"""Export tracker-selected observations from one or more runs on the same video."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from pipeline import load_geometry, resolve
from resumable_detections import publish


def export(config_path, runs, output):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["start"] != 0 or config["frames"] < 1:
        raise ValueError("Use a zero-based full-timeline config for export")
    stop = config["frames"]
    video = resolve(config["video"])
    geometry = resolve(config["homography"])
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened() or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != stop:
            raise ValueError("The export config must cover the entire source video")
    finally:
        capture.release()
    matrices, valid = load_geometry(geometry, stop)
    geometry_hash = hashlib.sha256(geometry.read_bytes()).hexdigest()

    segments = []
    for run in runs:
        manifest = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
        run_config = manifest["config"]
        if (resolve(run_config["video"]).resolve() != video.resolve()
                or manifest["homography_sha256"] != geometry_hash):
            raise ValueError(f"Source video or geometry differs in {run}")
        checkpoint = manifest.get("checkpoint_sha256")
        if checkpoint != config["expected_sha256"]:
            raise ValueError(f"Run {run} must record the configured checkpoint digest")
        start = run_config["start"]
        end = start + run_config["frames"]
        with (run / "selected.jsonl").open(encoding="utf-8") as source:
            rows = [json.loads(line) for line in source if line.strip()]
        if not 0 <= start < end <= stop or [row["frame"] for row in rows] != list(range(start, end)):
            raise ValueError(f"Invalid frame sequence in {run}")
        segments.append((start, end, rows))
    segments.sort(key=lambda segment: segment[0])
    if any(left[1] > right[0] for left, right in zip(segments, segments[1:])):
        raise ValueError("Run intervals overlap")
    fields = ("frame", "status", "x_px", "y_px", "x_pitch_normalized", "y_pitch_normalized",
              "confidence", "track_id", "projection_status")
    from io import StringIO

    counts = Counter()
    def lines():
        buffer = StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(fields)
        yield buffer.getvalue()
        for frame in range(stop):
            buffer.seek(0)
            buffer.truncate()
            row = [frame, "not_processed", "", "", "", "", "", "", ""]
            for segment, (start, end, selections) in enumerate(segments):
                if not start <= frame < end:
                    continue
                selected = selections[frame - start]
                row[1] = selected["status"]
                if row[1] == "observed":
                    x0, y0, x1, y1 = selected["xyxy"]
                    x, y = (x0 + x1) / 2, (y0 + y1) / 2
                    track_id = selected["track_id"]
                    if not 0 < track_id < 1_000_000:
                        raise ValueError(f"Track ID outside export range at frame {frame}")
                    row[2:4] = [x, y]
                    row[6:] = [selected["confidence"], track_id + segment * 1_000_000,
                               "invalid_geometry"]
                    if valid[frame] and np.isfinite(matrices[frame]).all():
                        try:
                            q = np.linalg.solve(matrices[frame], [x, y, 1])
                        except np.linalg.LinAlgError:
                            q = None
                        if q is not None and np.isfinite(q).all() and abs(q[2]) > 1e-12:
                            pitch = q[:2] / q[2]
                            row[-1] = "outside_pitch"
                            if ((pitch >= 0) & (pitch <= 1)).all():
                                row[4:6] = pitch.tolist()
                                row[-1] = "ground_plane_estimate"
                elif row[1] != "unknown":
                    raise ValueError("Use selected.jsonl from the original association")
                break
            counts[row[1]] += 1
            writer.writerow(row)
            yield buffer.getvalue()

    # Exclusive publication prevents replacing an earlier handoff.
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    publish(output, lines())
    return dict(counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Full-timeline config")
    parser.add_argument("--run", type=Path, action="append", required=True, help="Repeat for each interval")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.config, args.run, args.output), indent=2))
