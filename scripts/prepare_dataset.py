#!/usr/bin/env python3
"""Prepare frozen CNN/DailyMail 3.0.0 test samples matching nchapman/lillm-summarizer."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from litert_vs_llamacpp.dataset import (
    CONFIRMATION_COUNT,
    CONFIRMATION_SEED,
    PILOT_COUNT,
    PILOT_SEED,
    load_jsonl,
    sample_stream,
    sha256_file,
    write_jsonl,
)


def prepare_split(
    output: Path,
    count: int,
    seed: int,
    *,
    exclude_input: Path | None = None,
    overwrite: bool = False,
) -> Path:
    if output.exists() and not overwrite:
        print(f"Already exists: {output} (sha256={sha256_file(output)[:12]}...); pass --overwrite to regenerate")
        return output

    excluded_ids: set[str] = set()
    excluded_sha256: str | None = None
    if exclude_input is not None:
        excluded_ids = {row["id"] for row in load_jsonl(exclude_input)}
        excluded_sha256 = sha256_file(exclude_input)

    stream = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test", streaming=True)
    rows = sample_stream(stream, count, seed, exclude_ids=excluded_ids)
    for row in rows:
        row["dataset"] = "abisee/cnn_dailymail"
        row["dataset_config"] = "3.0.0"
        row["split"] = "test"
        row["sampling_seed"] = seed
        if excluded_sha256:
            row["excluded_input_sha256"] = excluded_sha256

    write_jsonl(output, rows)
    print(f"Wrote {len(rows)} examples to {output} (sha256={sha256_file(output)[:12]}...)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--include-confirmation",
        action="store_true",
        help="Also generate the disjoint 250-example confirmation split (takes longer to stream).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pilot_path = args.data_dir / "cnn_dailymail_test_pilot.jsonl"
    prepare_split(pilot_path, PILOT_COUNT, PILOT_SEED, overwrite=args.overwrite)

    if args.include_confirmation:
        conf_path = args.data_dir / "cnn_dailymail_test_confirmation.jsonl"
        prepare_split(
            conf_path,
            CONFIRMATION_COUNT,
            CONFIRMATION_SEED,
            exclude_input=pilot_path,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
