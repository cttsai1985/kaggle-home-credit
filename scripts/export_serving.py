"""Package an existing Home Credit run for engineered-feature MLflow serving."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from home_credit.serving import _sha256, export_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.rows <= 64:
        parser.error("--rows must be between 1 and 64")
    features = json.loads((args.run_dir / "features.json").read_text())
    producer = json.loads((args.run_dir / "manifest.json").read_text())
    cache_manifest = json.loads((args.cache / "manifest.json").read_text())
    actual_hash = _sha256(args.cache / "selected.parquet")
    if (
        actual_hash != cache_manifest["cache_sha256"]
        or actual_hash != producer["features"]["cache_sha256"]
    ):
        raise ValueError("Export examples require the producer's unchanged feature cache")
    parquet = pq.ParquetFile(args.cache / "selected.parquet")
    example = next(parquet.iter_batches(batch_size=args.rows, columns=features)).to_pandas()
    receipt = export_model(args.run_dir, args.output, example)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
