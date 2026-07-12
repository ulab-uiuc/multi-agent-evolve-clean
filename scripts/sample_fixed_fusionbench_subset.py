#!/usr/bin/env python3
"""Randomly subsample examples from an existing fixed FusionBench JSON."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def sample_subset(
    input_path: Path,
    num_samples: int,
    seed: int,
    output_path: Path,
) -> Path:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = data["examples"]
    example_pairs = data["example_pairs"]
    if len(examples) != len(example_pairs):
        raise ValueError(
            f"examples ({len(examples)}) and example_pairs ({len(example_pairs)}) length mismatch"
        )
    if num_samples > len(examples):
        raise ValueError(f"Requested {num_samples} samples but only {len(examples)} available")

    rng = random.Random(seed)
    indices = rng.sample(range(len(examples)), k=num_samples)
    indices.sort()

    selected_examples = []
    selected_pairs = []
    for new_idx, old_idx in enumerate(indices):
        example = json.loads(json.dumps(examples[old_idx]))
        pair = json.loads(json.dumps(example_pairs[old_idx]))
        example.setdefault("extra_info", {})["index"] = new_idx
        pair.setdefault("extra_info", {})["index"] = new_idx
        example["extra_info"]["source_index"] = old_idx
        pair["extra_info"]["source_index"] = old_idx
        selected_examples.append(example)
        selected_pairs.append(pair)

    metadata = dict(data.get("metadata", {}))
    metadata.update(
        {
            "num_samples": num_samples,
            "seed": seed,
            "source_file": str(input_path),
            "source_indices": indices,
            "description": (
                f"Random subset of {num_samples} samples from {input_path.name} "
                f"(seed={seed}) for half-ref training"
            ),
        }
    )

    output = {
        "metadata": metadata,
        "examples": selected_examples,
        "example_pairs": selected_pairs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved {num_samples} samples to {output_path}")
    print(f"Source indices: {indices}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a fixed FusionBench subset")
    parser.add_argument(
        "--input",
        type=str,
        default="data/fixed_datasets/fixed_fusionbench_1000.json",
        help="Path to the full fixed FusionBench JSON",
    )
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to keep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output JSON path. Trainer expects the filename fixed_fusionbench_1000.json "
            "under trainer.default_data_dir/fixed_datasets/"
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output is None:
        # Trainer loads: {default_data_dir}/fixed_datasets/fixed_fusionbench_1000.json
        # So for num_samples=16 use: trainer.default_data_dir=data_16
        output_path = Path(f"data_{args.num_samples}/fixed_datasets/fixed_fusionbench_1000.json")
    else:
        output_path = Path(args.output)

    sample_subset(
        input_path=input_path,
        num_samples=args.num_samples,
        seed=args.seed,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
