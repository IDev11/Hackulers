import csv
import json
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import run
from trajectory_reconstruction import reconstruct_selected_track


def observation(frame, x, y=10, track_id=1):
    return dict(frame=frame, status="observed", track_id=track_id, candidate_index=0,
                xyxy=[x - 2, y - 2, x + 2, y + 2], confidence=0.9,
                geometry_available=True)


def unknown(frame, **fields):
    row = dict(frame=frame, status="unknown", track_id=None, candidate_index=None,
               xyxy=None, confidence=None, geometry_available=True)
    row.update(fields)
    return row


class ReconstructionTests(unittest.TestCase):
    def test_linear_gap_preserves_original_rows_and_provenance(self):
        rows = [unknown(0), observation(1, 10), unknown(2), unknown(3),
                observation(4, 16, 16), unknown(5)]
        original = deepcopy(rows)
        result = reconstruct_selected_track(rows)
        self.assertEqual(rows, original)
        for index in (0, 1, 4, 5):
            self.assertEqual(result[index], rows[index])
        for index, center in ((2, (12, 12)), (3, (14, 14))):
            row = result[index]
            self.assertEqual((row["x_px"], row["y_px"]), center)
            self.assertEqual(row["status"], "interpolated")
            self.assertEqual(row["track_id"], 1)
            self.assertEqual(row["reconstruction_method"], "linear")
            self.assertEqual(row["endpoint_frames"], [1, 4])
            for field in ("confidence", "candidate_index", "xyxy"):
                self.assertIsNone(row[field])

    def test_unsupported_gaps_stay_unknown(self):
        for overrides in (
            dict(track_id=2), dict(track_id=None), dict(geometry_available=False),
            dict(confidence=None), dict(xyxy=[0, 0, 0, 1]),
        ):
            with self.subTest(overrides=overrides):
                right = observation(2, 14)
                right.update(overrides)
                rows = [observation(0, 10), unknown(1), right]
                self.assertEqual(reconstruct_selected_track(rows), rows)

        for middle in (unknown(1, geometry_available=False), unknown(1, track_id=2)):
            rows = [observation(0, 10), middle, observation(2, 14)]
            self.assertEqual(reconstruct_selected_track(rows), rows)
        rows = [observation(0, 10), *(unknown(i) for i in range(1, 7)), observation(7, 24)]
        self.assertEqual(reconstruct_selected_track(rows), rows)

    def test_hermite_uses_trusted_adjacent_neighbors_or_falls_back(self):
        rows = [observation(0, 0), observation(1, 1), unknown(2),
                observation(3, 9), observation(4, 16)]
        result = reconstruct_selected_track(rows, method="hermite")
        self.assertEqual(result[2]["reconstruction_method"], "hermite")
        self.assertAlmostEqual(result[2]["x_px"], 3.5)
        rows[-1]["track_id"] = 2
        result = reconstruct_selected_track(rows, method="hermite")
        self.assertEqual(result[2]["reconstruction_method"], "linear")
        self.assertAlmostEqual(result[2]["x_px"], 5)

    def test_rejects_discontinuous_frames_and_unknown_method(self):
        with self.assertRaises(ValueError):
            reconstruct_selected_track([observation(0, 0), observation(2, 2)])
        with self.assertRaises(ValueError):
            reconstruct_selected_track([], method="cubic")

    def test_synthetic_csv_five_frame_gap(self):
        fixture = ROOT / "notebooks" / "synthetic_ball_tracking.csv"
        if not fixture.is_file():
            self.skipTest("Local synthetic notebook CSV is unavailable")
        with fixture.open(newline="") as source:
            points = list(csv.DictReader(source))[:11]
        self.assertTrue(all(point["match_id"].startswith("SYNTHETIC_") for point in points))
        measured = [
            observation(int(point["frame"]), float(point["x"]), float(point["y"]))
            for point in points
        ]
        selected = [*measured[:3], *(unknown(frame) for frame in range(3, 8)), *measured[8:]]
        result = reconstruct_selected_track(selected)
        self.assertEqual([row["status"] for row in result].count("observed"), 6)
        self.assertEqual([row["status"] for row in result].count("interpolated"), 5)
        self.assertEqual(result[:3] + result[8:], measured[:3] + measured[8:])
        for row in result[3:8]:
            self.assertEqual(row["endpoint_frames"], [2, 8])
            fraction = (row["frame"] - 2) / 6
            for axis in ("x", "y"):
                start, end = (float(points[i][axis]) for i in (2, 8))
                self.assertAlmostEqual(row[f"{axis}_px"], start + fraction * (end - start))


class SavedTrackTests(unittest.TestCase):
    def test_cli_reconstructs_from_saved_selection_and_refuses_overwrite(self):
        selected = [observation(10, 8), unknown(11), observation(12, 12)]
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "selected.jsonl"
            target = Path(folder) / "reconstructed.jsonl"
            original = "".join(json.dumps(row) + "\n" for row in selected)
            source.write_text(original)
            command = [sys.executable, "-B", str(ROOT / "src/trajectory_reconstruction.py"),
                       "--selected", str(source), "--output", str(target)]
            subprocess.run(command, capture_output=True, check=True)
            result = [json.loads(line) for line in target.read_text().splitlines()]
            self.assertEqual(result, reconstruct_selected_track(selected))
            saved = target.read_bytes()
            retry = subprocess.run(command, capture_output=True)
            self.assertEqual(retry.returncode, 1)
            self.assertEqual(target.read_bytes(), saved)
            self.assertEqual(source.read_text(), original)


class CachedPipelineTests(unittest.TestCase):
    def test_opt_in_adds_output_with_original_artifacts_unchanged(self):
        records = [
            dict(frame=i, detections=[] if i == 4 else [
                dict(xyxy=[10 + i, 40, 14 + i, 44], confidence=0.9)
            ])
            for i in range(8)
        ]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root / "detections.jsonl"
            cache.write_text("".join(json.dumps(row) + "\n" for row in records))
            geometry = root / "geometry.csv"
            geometry.write_text("frame,valid,h00,h01,h02,h10,h11,h12,h20,h21,h22\n"
                                + "".join(f"{i},1,100,0,0,0,100,0,0,0,1\n" for i in range(8)))
            config = root / "config.json"
            config.write_text(json.dumps(dict(
                video=str(root / "unused.mp4"), weights=str(root / "absent.pt"),
                homography=str(geometry), start=0, frames=8, output_root=str(root / "runs"),
            )))
            capture = Mock()
            capture.get.side_effect = {3: 100, 4: 100, 5: 25, 7: 8}.__getitem__
            with patch("pipeline.cv2.VideoCapture", return_value=capture):
                default = run(config, cache, render=False)
                enabled = run(config, cache, render=False, reconstruct="linear")
            self.assertEqual(
                {file.name for file in enabled.iterdir()} - {file.name for file in default.iterdir()},
                {"reconstructed.jsonl"},
            )
            for file in default.iterdir():
                self.assertEqual(file.read_bytes(), (enabled / file.name).read_bytes(), file.name)
            selected = [json.loads(line) for line in (enabled / "selected.jsonl").read_text().splitlines()]
            result = [json.loads(line) for line in (enabled / "reconstructed.jsonl").read_text().splitlines()]
            self.assertEqual(selected[4]["status"], "unknown")
            self.assertEqual(result[4]["status"], "interpolated")
            self.assertEqual(result[4]["x_px"], 16)
            self.assertEqual(result[:4] + result[5:], selected[:4] + selected[5:])


if __name__ == "__main__":
    unittest.main()
