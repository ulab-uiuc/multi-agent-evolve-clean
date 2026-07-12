#!/usr/bin/env python3
"""Create the placeholder parquet required by trainer init for general MAE runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_examples(seed_json: Path) -> list[dict]:
    with open(seed_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["examples"]


def build_placeholder_rows(examples: list[dict], min_rows: int) -> list[dict]:
    if not examples:
        raise ValueError("No examples found in seed JSON")

    rows = []
    for idx in range(min_rows):
        example = examples[idx % len(examples)]
        rows.append(
            {
                "data_source": example.get("data_source", "gen_general"),
                "prompt": example["prompt"],
                "ability": example.get("ability", "general"),
                "reward_model": example.get("reward_model", {}),
                "extra_info": example.get("extra_info", {}),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build data/code_reason/test_answer.parquet placeholder for general MAE training"
    )
    parser.add_argument(
        "--seed-json",
        type=str,
        default="data_16/fixed_datasets/fixed_fusionbench_1000.json",
        help="Fixed FusionBench seed JSON used for placeholder rows",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/code_reason/test_answer.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=16,
        help="Minimum number of rows (should be >= data.train_batch_size)",
    )
    args = parser.parse_args()

    seed_json = Path(args.seed_json)
    output_path = Path(args.output)
    examples = load_examples(seed_json)
    rows = build_placeholder_rows(examples, args.min_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path)

    print(f"Saved {len(rows)} placeholder rows to {output_path}")
    print(f"Source seed JSON: {seed_json}")


if __name__ == "__main__":
    main()
