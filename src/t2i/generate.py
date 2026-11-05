"""Stable Diffusion v1.5 text->image generation + run logging."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image

from .embed import get_device

SD_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def load_pipeline(model_id: str = SD_MODEL_ID, device: str | None = None):
    device = device or get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()  # lowers peak VRAM, negligible speed cost
    return pipe.to(device)


def generate(
    pipe,
    prompts: list[str],
    seeds: list[int],
    steps: int = 30,
    guidance: float = 7.5,
    height: int = 512,
    width: int = 512,
) -> list[Image.Image]:
    """One image per (prompt, seed). CPU generators make seeds reproducible across devices."""
    images = []
    for prompt, seed in zip(prompts, seeds):
        g = torch.Generator(device="cpu").manual_seed(seed)
        images.append(
            pipe(
                prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=height,
                width=width,
                generator=g,
            ).images[0]
        )
    return images


def make_grid(images: list[Image.Image], cols: int = 3) -> Image.Image:
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * w, rows * h), "white")
    for i, im in enumerate(images):
        grid.paste(im, ((i % cols) * w, (i // cols) * h))
    return grid


def save_run(
    out_dir: str | Path,
    prompts: list[str],
    seeds: list[int],
    images: list[Image.Image],
    scores: list[float] | None = None,
    config: dict | None = None,
) -> None:
    """Persist images, a grid and a JSON log so every result is reproducible."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for i, (p, s, im) in enumerate(zip(prompts, seeds, images), start=1):
        name = f"run_{i:02d}.png"
        im.save(out_dir / name)
        runs.append(
            dict(
                file=name,
                prompt=p,
                seed=s,
                clip_score=None if scores is None else round(scores[i - 1], 3),
            )
        )
    make_grid(images).save(out_dir / "grid.png")
    (out_dir / "runs.json").write_text(json.dumps(dict(config=config or {}, runs=runs), indent=2))
