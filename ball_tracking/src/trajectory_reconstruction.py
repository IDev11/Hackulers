"""Offline reconstruction of short gaps in the selected ball track."""

import argparse
import json
from copy import deepcopy
from math import isfinite
from pathlib import Path


MAX_MISSING_FRAMES = 5


def _trusted_center(row):
    if (
        row["status"] != "observed"
        or row.get("track_id") is None
        or row.get("geometry_available") is not True
    ):
        return None

    confidence = row.get("confidence")
    box = row.get("xyxy")
    if not (
        isinstance(confidence, (int, float))
        and 0 <= confidence <= 1
        and isinstance(box, (list, tuple))
        and len(box) == 4
        and all(isinstance(value, (int, float)) and isfinite(value) for value in box)
        and box[0] < box[2]
        and box[1] < box[3]
    ):
        return None
    x0, y0, x1, y1 = box
    return x0 / 2 + x1 / 2, y0 / 2 + y1 / 2


def _hermite(start, end, before, after, span, t):
    t2, t3 = t * t, t * t * t
    return (
        (2 * t3 - 3 * t2 + 1) * start
        + (t3 - 2 * t2 + t) * span * (start - before)
        + (-2 * t3 + 3 * t2) * end
        + (t3 - t2) * span * (after - end)
    )


def reconstruct_selected_track(selected, *, method="linear"):
    """Return a copy of selected rows, adding image-space centers only to filled rows.

    Observed and unresolved rows retain their original fields and values.
    Hermite falls back to linear without adjacent trusted observations from
    the same track on both sides.
    """
    if method not in ("linear", "hermite"):
        raise ValueError(f"Unsupported reconstruction method: {method}")
    frames = [row["frame"] for row in selected]
    if any(type(frame) is not int or frame < 0 for frame in frames):
        raise ValueError("Reconstruction requires nonnegative integer frames")
    if any(right != left + 1 for left, right in zip(frames, frames[1:])):
        raise ValueError("Reconstruction requires consecutive frame records")

    output = deepcopy(selected)
    observed = [index for index, row in enumerate(selected) if row["status"] == "observed"]
    centers = {index: _trusted_center(selected[index]) for index in observed}
    for left, right in zip(observed, observed[1:]):
        gap = right - left - 1
        if not 1 <= gap <= MAX_MISSING_FRAMES:
            continue
        track_id = selected[left]["track_id"] if centers[left] is not None else None
        if track_id is None or centers[right] is None or selected[right]["track_id"] != track_id:
            continue
        missing = selected[left + 1:right]
        if any(
            row["status"] != "unknown"
            or row.get("geometry_available") is not True
            or row.get("track_id") not in (None, track_id)
            for row in missing
        ):
            continue

        before, after = centers.get(left - 1), centers.get(right + 1)
        use_hermite = (
            method == "hermite"
            and before is not None
            and after is not None
            and selected[left - 1]["track_id"] == selected[right + 1]["track_id"] == track_id
        )
        span = right - left
        for offset in range(1, span):
            t = offset / span
            center = tuple((1 - t) * a + t * b for a, b in zip(centers[left], centers[right]))
            chosen_method = "linear"
            if use_hermite:
                estimate = tuple(
                    _hermite(a, b, prev, following, span, t)
                    for a, b, prev, following in zip(centers[left], centers[right], before, after)
                )
                if all(isfinite(value) for value in estimate):
                    center = estimate
                    chosen_method = "hermite"
            output[left + offset].update(
                status="interpolated",
                track_id=track_id,
                candidate_index=None,
                xyxy=None,
                confidence=None,
                x_px=center[0],
                y_px=center[1],
                reconstruction_method=chosen_method,
                endpoint_frames=[frames[left], frames[right]],
            )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", required=True, type=Path, help="Existing selected.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="New reconstructed.jsonl")
    parser.add_argument("--method", default="linear", choices=("linear", "hermite"))
    args = parser.parse_args()
    try:
        with args.selected.open(encoding="utf-8") as source:
            selected = [json.loads(line) for line in source if line.strip()]
        if not selected:
            raise ValueError("The selected track contains no frames")
        if any(not isinstance(row, dict) or row.get("status") not in ("observed", "unknown")
               for row in selected):
            raise ValueError("Expected selected observations or unknown frames")
        reconstructed = reconstruct_selected_track(selected, method=args.method)
        payload = "".join(json.dumps(row, allow_nan=False) + "\n" for row in reconstructed)
        with args.output.open("x", encoding="utf-8", newline="\n") as target:
            target.write(payload)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Reconstruction failed: {error}\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
