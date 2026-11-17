import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import run
from resumable_detections import collect


class ChunkTests(unittest.TestCase):
    def test_rejects_changed_settings_and_corrupt_saved_chunk(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            detect = Mock(return_value=[dict(frame=10, detections=[])])
            progress = Mock()
            collect(root, 10, 11, 2, {"device": "cpu"}, detect, progress)
            with self.assertRaisesRegex(ValueError, "Resume settings differ"):
                collect(root, 10, 11, 2, {"device": "cuda"}, detect, progress)
            chunk = root / "000000010-000000010.jsonl"
            chunk.write_text('{"frame": 11, "detections": []}\n')
            with self.assertRaisesRegex(ValueError, "Invalid saved chunk"):
                collect(root, 10, 11, 2, {"device": "cpu"}, detect, progress)
            detect.assert_called_once_with(10, 11)


class ResumablePipelineTests(unittest.TestCase):
    def test_resumes_after_interruption_with_continuous_association(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "synthetic.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (100, 100))
            self.assertTrue(writer.isOpened())
            try:
                for frame in range(8):
                    writer.write(np.full((100, 100, 3), frame * 25, dtype=np.uint8))
            finally:
                writer.release()
            weights = root / "synthetic.pt"
            weights.write_bytes(b"synthetic")
            geometry = root / "geometry.csv"
            geometry.write_text("frame,valid,h00,h01,h02,h10,h11,h12,h20,h21,h22\n"
                                + "".join(f"{i},1,100,0,0,0,100,0,0,0,1\n" for i in range(8)))
            config = root / "config.json"
            config.write_text(json.dumps(dict(
                video=str(video), weights=str(weights), homography=str(geometry),
                start=0, frames=8, output_root=str(root / "runs"), device="cpu",
                confidence=0.25, imgsz=1280,
                expected_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
            )))
            chunks = root / "chunks"
            frames_seen = []

            def predict(image, **kwargs):
                frame = round(float(image.mean()) / 25)
                frames_seen.append(frame)
                if frame == 4 and frames_seen.count(4) == 1:
                    raise RuntimeError("simulated interruption")
                boxes = Mock()
                boxes.cpu.return_value.numpy.return_value = SimpleNamespace(
                    xyxy=np.empty((0, 4)) if frame == 3 else np.array([[10+frame, 40, 14+frame, 44]]),
                    conf=[] if frame == 3 else [0.9],
                )
                return [SimpleNamespace(boxes=boxes)]

            yolo = Mock(return_value=SimpleNamespace(names={0: "ball"}, predict=predict))
            with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=yolo)}):
                with patch("pipeline.version", return_value="synthetic"):
                    options = dict(render=False, reconstruct="linear",
                                   resume_detections=chunks, chunk_frames=3)
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        run(config, **options)
                    self.assertEqual([p.name for p in chunks.glob("*.jsonl")],
                                     ["000000000-000000002.jsonl"])
                    self.assertFalse((root / "runs").exists())
                    out = run(config, **options)
                    repeated = run(config, **options)
            self.assertEqual(yolo.call_count, 2)
            self.assertEqual(frames_seen, [0, 1, 2, 3, 4, 3, 4, 5, 6, 7])
            with (out / "reconstructed.jsonl").open() as source:
                rows = [json.loads(line) for line in source]
            self.assertEqual([row["frame"] for row in rows], list(range(8)))
            self.assertEqual(rows[3]["status"], "interpolated")
            self.assertEqual(rows[3]["endpoint_frames"], [2, 4])
            self.assertEqual(rows[2]["track_id"], rows[4]["track_id"])
            self.assertEqual((out / "reconstructed.jsonl").read_bytes(),
                             (repeated / "reconstructed.jsonl").read_bytes())
            self.assertEqual(json.loads((out / "run_config.json").read_text())["mode"],
                             "resumable_detector")


if __name__ == "__main__":
    unittest.main()
