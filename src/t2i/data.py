"""Dataset selection, cleaning and preprocessing for COCO image-caption pairs.

Pipeline
--------
1. ``load_coco_records``  - read the official COCO captions JSON, clean every caption.
2. ``select_subset``      - filter by image size / caption count, seeded random sample.
3. ``build_dataset``      - download, square-crop + resize, dedupe, split, write
                            ``metadata.jsonl`` and ``stats.json``.

Only light dependencies (Pillow + stdlib) so it runs anywhere and is easy to test.
"""
from __future__ import annotations

import json
import random
import re
import time
import unicodedata
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image

COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
MIN_WORDS, MAX_WORDS = 4, 40  # CLIP truncates at 77 tokens; 40 words stays safely below
PRIMARY_MAX_WORDS = 25


# --------------------------------------------------------------------------- text
def clean_caption(text: str) -> Optional[str]:
    """Normalise one caption. Returns ``None`` if it should be dropped."""
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"\s+", " ", t).strip()
    t = t.strip(" \"'")
    t = re.sub(r"[.\s]+$", "", t)  # COCO captions often end with a stray period
    n_words = len(t.split())
    if n_words < MIN_WORDS or n_words > MAX_WORDS:
        return None
    return t


def primary_caption(captions: list[str]) -> str:
    """Pick the most descriptive caption (most words, capped) as the training caption."""
    short_enough = [c for c in captions if len(c.split()) <= PRIMARY_MAX_WORDS]
    pool = short_enough or captions
    return max(pool, key=lambda c: len(c.split()))  # max() keeps the first on ties


# ------------------------------------------------------------------------ records
def ensure_annotations(raw_dir: Path, coco_split: str = "val") -> Path:
    """Return path to ``captions_{split}2017.json``, downloading it if necessary."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"captions_{coco_split}2017.json"
    if target.exists():
        return target
    zip_path = raw_dir / "annotations_trainval2017.zip"
    if not zip_path.exists():
        print(f"Downloading COCO annotations (~240 MB) -> {zip_path}")
        urllib.request.urlretrieve(COCO_ANN_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        member = f"annotations/captions_{coco_split}2017.json"
        target.write_bytes(zf.read(member))
    return target


def load_coco_records(ann_path: str | Path) -> list[dict]:
    """One record per image: id, url, size and the list of *cleaned*, unique captions."""
    data = json.loads(Path(ann_path).read_text())
    images = {im["id"]: im for im in data["images"]}
    caps: dict[int, list[str]] = {}
    for ann in data["annotations"]:
        c = clean_caption(ann["caption"])
        if c is None:
            continue
        bucket = caps.setdefault(ann["image_id"], [])
        if c not in bucket:
            bucket.append(c)
    records = []
    for image_id, captions in caps.items():
        im = images[image_id]
        records.append(
            dict(
                image_id=image_id,
                file_name=im["file_name"],
                url=im["coco_url"],
                width=im["width"],
                height=im["height"],
                captions=captions,
            )
        )
    records.sort(key=lambda r: r["image_id"])
    return records


def select_subset(
    records: list[dict], n: int, seed: int = 42, min_side: int = 256, min_captions: int = 3
) -> list[dict]:
    """Filter low-quality candidates, then take a seeded random sample of ``n``."""
    eligible = [
        r
        for r in records
        if min(r["width"], r["height"]) >= min_side and len(r["captions"]) >= min_captions
    ]
    rng = random.Random(seed)
    return rng.sample(eligible, min(n, len(eligible)))


def assign_splits(
    ids: Iterable[int], seed: int = 42, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> dict[int, str]:
    """Deterministic train/val/test assignment by image id (no image in two splits)."""
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    out = {}
    for i, image_id in enumerate(ids):
        out[image_id] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return out


# ------------------------------------------------------------------------- images
def fetch_image(url: str, retries: int = 3, timeout: int = 30) -> Optional[Image.Image]:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return Image.open(BytesIO(resp.read())).convert("RGB")
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    """Square centre-crop then resize (SD v1.5 is trained at 512x512)."""
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------------- orchestration
def build_dataset(
    records: list[dict],
    out_dir: str | Path,
    size: int = 512,
    workers: int = 16,
    seed: int = 42,
) -> list[dict]:
    """Download + preprocess ``records`` into ``out_dir`` and write metadata.

    Resumable: images already on disk are not downloaded again.
    """
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    def work(rec: dict) -> Optional[dict]:
        rel = f"images/{rec['image_id']:012d}.jpg"
        dest = out_dir / rel
        if not dest.exists():
            img = fetch_image(rec["url"])
            if img is None:
                return None
            center_crop_resize(img, size).save(dest, quality=95)
        return dict(
            image_id=rec["image_id"],
            file=rel,
            caption=primary_caption(rec["captions"]),
            captions=rec["captions"],
            orig_size=[rec["width"], rec["height"]],
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(work, records))
    n_failed = sum(r is None for r in rows)
    rows = [r for r in rows if r is not None]

    # Drop images whose primary caption duplicates an earlier one (prevents leakage/memorisation)
    seen, unique_rows = set(), []
    for r in rows:
        key = r["caption"].lower()
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(r)
    n_dupes = len(rows) - len(unique_rows)

    splits = assign_splits([r["image_id"] for r in unique_rows], seed=seed)
    for r in unique_rows:
        r["split"] = splits[r["image_id"]]

    with open(out_dir / "metadata.jsonl", "w") as f:
        for r in unique_rows:
            f.write(json.dumps(r) + "\n")

    lengths = [len(r["caption"].split()) for r in unique_rows]
    stats = dict(
        requested=len(records),
        kept=len(unique_rows),
        download_failed=n_failed,
        duplicate_captions_dropped=n_dupes,
        resolution=size,
        seed=seed,
        splits={s: sum(r["split"] == s for r in unique_rows) for s in ("train", "val", "test")},
        caption_words=dict(
            min=min(lengths, default=0),
            mean=round(sum(lengths) / max(len(lengths), 1), 2),
            max=max(lengths, default=0),
        ),
    )
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return unique_rows


def load_metadata(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
