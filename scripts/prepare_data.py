#!/usr/bin/env python
"""Download + clean + preprocess a curated COCO image-caption subset.

Example
-------
    python scripts/prepare_data.py --n 2000 --size 512 --seed 42

Outputs (in --out):
    images/*.jpg     square-cropped, resized RGB images
    metadata.jsonl   one row per image: caption, all cleaned captions, split
    stats.json       counts and caption-length statistics for the proposal/report
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2i.data import build_dataset, ensure_annotations, load_coco_records, select_subset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=2000, help="images to sample (1000-5000 for this project)")
    ap.add_argument("--size", type=int, default=512, help="output resolution (SD v1.5 native = 512)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--coco-split", choices=["val", "train"], default="val",
                    help="val = 5k images (enough for <=5k pairs); train = 118k")
    ap.add_argument("--annotations", type=Path, default=None,
                    help="path to a local captions_*.json (skips the 240 MB download)")
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    ann = args.annotations or ensure_annotations(args.raw, args.coco_split)
    records = load_coco_records(ann)
    print(f"{len(records):,} images with >=1 valid caption in annotations")
    subset = select_subset(records, n=args.n, seed=args.seed)
    print(f"Selected {len(subset):,} images; downloading + preprocessing -> {args.out}")
    build_dataset(subset, args.out, size=args.size, workers=args.workers, seed=args.seed)
    print(json.dumps(json.loads((args.out / "stats.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
