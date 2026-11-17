# YOLO26 ball tracking reproduction

This folder reproduces local ball detection, candidate association, optional trajectory reconstruction, and the frame-aligned CSV used by the Hackulers possession pipeline. It can be run independently of the possession model.

## Environment

Use Python 3.10 in a separate environment. Run these commands from this folder after activating it:

```sh
python -m pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The tests use synthetic frames or synthetic detections for association, cache recovery and interpolation. They require no checkpoints or match assets.

## Local inputs

- Obtain the tactical video and its matching homography CSV through authorized challenge access. See the [repository data notes](../data/README.md) for the expected layout. Edit the config if your video has a different filename. The video must be the same timeline as the homography and downstream ATD.
- Obtain the existing `yolo26-best-ball.pt` checkpoint from the team via an authorized local handoff. The checkpoint is excluded from this repository because redistribution permission has not been established. Its filename, size and SHA-256 are recorded in [checkpoint.json](weights/checkpoint.json). The runner verifies the SHA-256 before fresh inference.
- Copy [configs/example.json](configs/example.json) to `configs/local.json` and edit the asset paths. Configured relative asset paths resolve from this folder. The homography CSV needs consecutive zero-based `frame` values, a numeric `valid` flag and `h00` through `h22`. Each matrix maps normalized pitch coordinates to source image pixels.

## Run detection and association

Start with a short interval to verify installation. Set `start` and `frames` in the local config, then run:

```sh
python src/pipeline.py --config configs/local.json --resume-detections outputs/smoke_chunks --no-render
```

For a longer run, use a new chunk directory for each interval. Rerun the same command with the same config and chunk size to resume. Completed 250-frame chunks are reused, and an unfinished chunk is rerun. Use a local filesystem that supports hard links, such as NTFS. The original CPU run was slow, so prefer a short execution check before attempting the match.

The terminal prints the new run directory after association. `detections.jsonl` records raw boxes and confidence. `selected.jsonl` has one row per processed frame, with either an original tracker-selected box or an unknown state. `candidate_decisions.csv` and `run_config.json` give the run's decision log and settings. Add `--reconstruct` for a separate `reconstructed.jsonl` with bounded same-track interpolation. Estimated positions carry endpoint provenance and have null YOLO confidence. The fallback CSV uses selected observations.

To reproduce the 127,000-processed-frame handoff, make two local configs with these intervals. Preserve the original source frame numbers.

| Interval | start | frames | Final source frame |
|---|---:|---:|---:|
| Initial run prefix | 0 | 94250 | 94249 |
| Second-half prefix | 110808 | 32750 | 143557 |

## Export the fallback CSV

Keep a copy of the full-timeline config at `configs/local_full.json` with `start=0` and `frames=199319`. Use the run directories printed by the two inference commands:

```sh
python src/export_fallback_csv.py --config configs/local_full.json --run outputs/FIRST_RUN --run outputs/SECOND_RUN --output outputs/yolo_fallback.csv
```

The CSV contains one row per original video frame. Frames outside the processed intervals, including the skipped halftime gap, are marked `not_processed`. Frames with no tracker-selected box are `unknown`. Both have blank detection fields. Track IDs are local to each interval and receive distinct numeric ranges in the merged CSV.

For possession integration, join on `frame`, keep ATD where it has a position, and use normalized pitch coordinates only where ATD is missing and `projection_status=ground_plane_estimate`. A projected aerial ball may land at the wrong place on the pitch. Pass the generated CSV explicitly to the main possession pipeline using its `--yolo` option.

## Reproduction scope and data handling

New files in this folder contain code, checkpoint metadata, and a synthetic fixture. Match inputs, cached match detections, generated exports, private configs and checkpoint binaries stay local. This workflow reads no possession ground truth. Keep assessment, training and parameter selection on synthetic or authorized external data in line with the challenge rules. The smoke tests establish execution and output semantics. Production metrics are outside the scope of this package.
