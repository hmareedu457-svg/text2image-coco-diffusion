import json
from pathlib import Path

from PIL import Image

from t2i.data import (
    assign_splits,
    build_dataset,
    center_crop_resize,
    clean_caption,
    load_coco_records,
    load_metadata,
    primary_caption,
    select_subset,
)


def test_clean_caption_normalises():
    assert clean_caption("  A man   riding a wave on a surfboard. ") == "A man riding a wave on a surfboard"
    assert clean_caption('"A cafe\u0301 table with two cups"') == "A cafe table with two cups"


def test_clean_caption_drops_too_short_or_long():
    assert clean_caption("A dog.") is None
    assert clean_caption(" ".join(["word"] * 60)) is None


def test_primary_caption_prefers_descriptive_but_capped():
    caps = ["a dog on grass", "a brown dog running across a green grassy field", " ".join(["x"] * 30)]
    assert primary_caption(caps) == "a brown dog running across a green grassy field"


def test_assign_splits_is_deterministic_and_disjoint():
    ids = list(range(100))
    a, b = assign_splits(ids, seed=1), assign_splits(ids, seed=1)
    assert a == b
    counts = {s: sum(v == s for v in a.values()) for s in ("train", "val", "test")}
    assert counts == {"train": 80, "val": 10, "test": 10}


def test_center_crop_resize_is_square():
    out = center_crop_resize(Image.new("RGB", (640, 480)), 256)
    assert out.size == (256, 256)


def _fake_coco(tmp: Path, n=30):
    images, anns = [], []
    for i in range(1, n + 1):
        p = tmp / f"src_{i}.jpg"
        Image.new("RGB", (400 + i, 300 + i), (i * 7 % 255, 90, 160)).save(p)
        images.append(dict(id=i, file_name=p.name, coco_url=p.as_uri(), width=400 + i, height=300 + i))
        for j in range(5):
            anns.append(dict(image_id=i, caption=f"A photo number {i} with object {j} on a table."))
    # image 1 and 2 share an identical best caption -> one must be deduped
    anns.append(dict(image_id=1, caption="the very same long caption used twice in the data set here"))
    anns.append(dict(image_id=2, caption="the very same long caption used twice in the data set here"))
    ann_path = tmp / "captions.json"
    ann_path.write_text(json.dumps(dict(images=images, annotations=anns)))
    return ann_path


def test_build_dataset_end_to_end(tmp_path):
    ann = _fake_coco(tmp_path)
    records = load_coco_records(ann)
    subset = select_subset(records, n=20, seed=3, min_side=256)
    assert len(subset) == 20
    out = tmp_path / "processed"
    rows = build_dataset(subset, out, size=64, workers=4, seed=3)
    meta = load_metadata(out / "metadata.jsonl")
    assert len(meta) == len(rows)
    assert {r["split"] for r in meta} <= {"train", "val", "test"}
    assert len({r["caption"].lower() for r in meta}) == len(meta)  # no duplicate captions
    for r in meta:
        assert Image.open(out / r["file"]).size == (64, 64)
    stats = json.loads((out / "stats.json").read_text())
    assert stats["kept"] == len(meta)
