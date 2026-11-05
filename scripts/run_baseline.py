#!/usr/bin/env python
"""Run the 5 baseline prompt->image tests (zero-shot Stable Diffusion v1.5).

    python scripts/run_baseline.py                      # -> results/baseline_v0/
    python scripts/run_baseline.py --out results/foo --steps 50

Needs a GPU for reasonable speed (~5 s/image on a Colab T4, minutes on CPU).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2i.evaluate import clip_scores, load_clip  # noqa: E402
from t2i.generate import SD_MODEL_ID, generate, load_pipeline, save_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, default=ROOT / "prompts" / "test_prompts.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "baseline_v0")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    args = ap.parse_args()

    items = json.loads(args.prompts.read_text())
    prompts = [i["prompt"] for i in items]
    seeds = [i["seed"] for i in items]

    pipe = load_pipeline()
    images = generate(pipe, prompts, seeds, steps=args.steps, guidance=args.guidance)
    clip_model, clip_proc = load_clip()
    scores = clip_scores(images, prompts, clip_model, clip_proc)

    config = dict(model=SD_MODEL_ID, scheduler="DPMSolverMultistep", steps=args.steps,
                  guidance=args.guidance, resolution=512, mean_clip_score=round(sum(scores) / len(scores), 3))
    save_run(args.out, prompts, seeds, images, scores, config)
    for p, s in zip(prompts, scores):
        print(f"{s:6.2f}  {p}")
    print(f"mean CLIPScore: {config['mean_clip_score']}  -> saved to {args.out}")


if __name__ == "__main__":
    main()
