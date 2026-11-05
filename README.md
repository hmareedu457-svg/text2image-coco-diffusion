# Text-to-Image Generation with CLIP + Stable Diffusion (COCO subset)

Generative-AI project: build a text→image pipeline on a curated **COCO** image–caption subset, establish a zero-shot **Stable Diffusion v1.5** baseline, then adapt it with **LoRA** fine-tuning.

| | |
|---|---|
| **Dataset** | COCO 2017 captions, 2,000 curated image–caption pairs (80/10/10 train/val/test) |
| **Text encoder** | CLIP ViT-L/14 (frozen), the encoder SD v1.x conditions on |
| **Generator** | Stable Diffusion v1.5 (latent diffusion, 512×512) via Hugging Face Diffusers |
| **Baseline** | Zero-shot SD v1.5, DPM-Solver++ 30 steps, CFG 7.5 |
| **Metrics** | CLIPScore, text→image retrieval recall, FID/KID, small human preference check |

## Repository layout

```
├── docs/proposal.docx          1-page proposal (dataset + model + roles)
├── notebooks/
│   └── 01_baseline_text_embedding.ipynb   embedding pipeline + 5 test runs
├── prompts/test_prompts.json   the 5 fixed prompts + seeds (reused for every model)
├── results/baseline_v0/        committed outputs of the 5 baseline test runs
├── scripts/
│   ├── prepare_data.py         download -> clean -> preprocess -> split
│   └── run_baseline.py         CLI version of the 5 test runs
├── src/t2i/                    data.py · embed.py · generate.py · evaluate.py
├── tests/                      unit + end-to-end tests for the data pipeline
└── data/                       raw/ (ignored) · processed/ (images ignored, metadata tracked)
```

## Quickstart

```bash
# 1. environment (Python 3.10+)
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. data: ~240 MB annotations download, then 2,000 images (~1-3 min)
python scripts/prepare_data.py --n 2000 --size 512 --seed 42

# 3. baseline: 5 prompt->image runs (GPU strongly recommended)
python scripts/run_baseline.py          # writes results/baseline_v0/{run_01..05.png, grid.png, runs.json}

# or run everything step by step
jupyter lab notebooks/01_baseline_text_embedding.ipynb

# tests
pytest -q
```

**Google Colab:** *Runtime → Change runtime type → T4 GPU*, clone the repo, `pip install -r requirements.txt`, run the notebook. A T4 (16 GB) is enough for baseline inference and for LoRA fine-tuning in fp16.

## Data pipeline details

1. Read official `captions_val2017.json`.
2. **Caption cleaning:** Unicode→ASCII, whitespace collapse, strip quotes and trailing periods, drop captions under 4 or over 40 words (CLIP truncates at 77 tokens), de-duplicate within an image.
3. **Image filtering:** shorter side ≥ 256 px and ≥ 3 valid captions.
4. Seeded random sample of *N* images, square centre-crop and resize to 512×512 (SD v1.5 native).
5. Primary training caption = most descriptive caption (≤ 25 words); all cleaned captions are kept in `captions` for evaluation.
6. Drop images whose primary caption duplicates another's, then split by image id (no image appears in two splits).

Everything is seeded: the same command produces the same subset.

## Reproducibility

- Seeds are fixed for data sampling, splits and generation (CPU generators → identical across devices).
- `results/*/runs.json` records model ID, scheduler, steps, guidance, per-image seed and CLIPScore.

## Licenses and responsible use

- COCO annotations: CC BY 4.0; images are subject to Flickr terms of use. Processed images are not committed to this repo.
- Stable Diffusion v1.5 weights: CreativeML Open RAIL-M. Read the use restrictions before publishing outputs.
- The default Diffusers safety checker stays enabled. It occasionally returns a black image on a false positive; re-run with a different seed if that happens.

## Team and workflow

| Role | Owner | Scope |
|---|---|---|
| Data lead | | `scripts/prepare_data.py`, `src/t2i/data.py`, dataset stats |
| Modeling lead | | `src/t2i/generate.py`, baseline, LoRA fine-tuning |
| Evaluation & integration lead | | `src/t2i/evaluate.py`, metrics, human study, repo/tests/README, final report |

Branching: `main` is protected; work on `feature/<name>` branches and open pull requests. Commit messages: imperative mood, e.g. `Add caption deduplication`.
