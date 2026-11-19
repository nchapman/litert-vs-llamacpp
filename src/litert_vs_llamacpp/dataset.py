"""Reproducible CNN/DailyMail dataset sampling matching nchapman/lillm-summarizer."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

PILOT_COUNT = 50
PILOT_SEED = 20261028
CONFIRMATION_COUNT = 250
CONFIRMATION_SEED = 20261029


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records and validate the fields required by the benchmark."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each record must be a JSON object")
            for field in ("id", "document", "reference"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"{path}:{line_number}: field {field!r} must be a non-empty string")
            rows.append(row)
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_stream(
    records: Iterable[dict[str, Any]],
    count: int,
    seed: int,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select a reproducible reservoir sample, matching lillm-summarizer exactly."""
    excluded = exclude_ids or set()
    if count <= 0:
        raise ValueError("count must be greater than zero")
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    valid_seen = 0
    eligible_seen = 0
    for record in records:
        candidate = {
            "id": str(record.get("id") or record.get("article_id") or valid_seen),
            "document": record.get("article") or record.get("document"),
            "reference": record.get("highlights") or record.get("summary"),
        }
        if not candidate["document"] or not candidate["reference"]:
            continue
        valid_seen += 1
        if candidate["id"] in excluded:
            continue
        eligible_seen += 1
        if len(sample) < count:
            sample.append(candidate)
        else:
            slot = rng.randrange(eligible_seen)
            if slot < count:
                sample[slot] = candidate
    if len(sample) < count:
        raise ValueError(f"Requested {count} records but found only {len(sample)} valid records")
    return sample
