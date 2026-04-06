#!/usr/bin/env python3
"""Randomly sample a subset from local parquet shards and save as multi-shard parquet.

Example:
  python scripts/sample_rl_train_subset.py \
    --source-dir /root/autodl-tmp/code/Vision-SR1-main/data/Vision-SR1-47K@train \
    --output-dir /root/autodl-tmp/code/Vision-SR1-main/data/Vision-SR1-10K@train \
    --num-samples 10000 \
    --seed 42 \
    --samples-per-shard 1000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a local parquet dataset into smaller parquet shards.")
    parser.add_argument("--source-dir", type=Path, default="/root/autodl-tmp/code/Vision-SR1-main/data/Vision-SR1-47K@train", help="Directory containing source parquet files.")
    parser.add_argument("--output-dir", type=Path, default="/root/autodl-tmp/code/Vision-SR1-main/data/Vision-SR1-10K@train", help="Directory to write sampled parquet shards.")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of examples to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=2000,
        help="Max number of records per output parquet shard.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {args.source_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_parquet = sorted(args.output_dir.glob("*.parquet"))
    if existing_parquet and not args.overwrite:
        raise RuntimeError(
            f"Output directory already contains parquet files: {args.output_dir}. "
            "Use --overwrite to overwrite."
        )

    for p in existing_parquet:
        p.unlink()

    # Load local parquet shards as one train split.
    dataset = load_dataset("parquet", data_dir=str(args.source_dir), split="train")
    total = len(dataset)
    if args.num_samples > total:
        raise ValueError(f"num-samples ({args.num_samples}) exceeds dataset size ({total}).")

    sampled = dataset.shuffle(seed=args.seed).select(range(args.num_samples))

    num_shards = math.ceil(args.num_samples / args.samples_per_shard)
    for shard_idx in range(num_shards):
        start = shard_idx * args.samples_per_shard
        end = min((shard_idx + 1) * args.samples_per_shard, args.num_samples)
        shard = sampled.select(range(start, end))
        shard_path = args.output_dir / f"train-{shard_idx}.parquet"
        shard.to_parquet(str(shard_path))
        print(f"Wrote {len(shard)} rows -> {shard_path}")

    metadata = {
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "source_size": total,
        "sample_size": args.num_samples,
        "seed": args.seed,
        "samples_per_shard": args.samples_per_shard,
        "num_shards": num_shards,
    }
    (args.output_dir / "subset_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Done. Metadata:")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
