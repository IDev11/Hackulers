"""Persist complete detector chunks for restartable local inference."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def publish(path, lines):
    """Publish a complete file atomically, keeping an existing writer's result."""
    temporary = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=".pending-", delete=False, mode="w",
            encoding="utf-8", newline="\n",
        ) as stream:
            temporary = Path(stream.name)
            stream.writelines(lines)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate(rows, start, stop, label):
    if len(rows) != stop - start or any(
        not isinstance(row, dict)
        or type(row.get("frame")) is not int
        or row["frame"] != frame
        or not isinstance(row.get("detections"), list)
        for frame, row in enumerate(rows, start)
    ):
        raise ValueError(f"{label}: expected one detection record per frame in [{start}, {stop})")


def load_chunk(path, start, stop):
    try:
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        validate(rows, start, stop, path.name)
    except ValueError as error:
        raise ValueError(f"Invalid saved chunk {path}. Move it aside and rerun. {error}") from error
    return rows


def iter_chunks(directory, start, stop, chunk_frames, detector_identity, detect, progress):
    """Yield complete chunks in frame order, with at most one chunk in memory."""
    if type(chunk_frames) is not int or chunk_frames <= 0:
        raise ValueError("Chunk size must be a positive integer")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    expected = dict(version=1, start=start, stop=stop, chunk_frames=chunk_frames,
                    detector=detector_identity)
    if not manifest.exists():
        publish(manifest, [json.dumps(expected, allow_nan=False, indent=2) + "\n"])
    with manifest.open(encoding="utf-8") as source:
        actual = json.load(source)
    if actual != expected:
        raise ValueError(f"Resume settings differ from {manifest}. Use the original settings or a new directory.")

    for first in range(start, stop, chunk_frames):
        end = min(first + chunk_frames, stop)
        path = directory / f"{first:09d}-{end - 1:09d}.jsonl"
        saved = path.is_file()
        if not saved:
            rows = detect(first, end)
            validate(rows, first, end, "Detector output")
            publish(path, (json.dumps(row, allow_nan=False) + "\n" for row in rows))
        chunk = load_chunk(path, first, end)
        progress(first, end, saved)
        yield chunk


def collect(directory, start, stop, chunk_frames, detector_identity, detect, progress):
    """Return the continuous timeline for association."""
    return [row for chunk in iter_chunks(directory, start, stop, chunk_frames,
                                        detector_identity, detect, progress) for row in chunk]
