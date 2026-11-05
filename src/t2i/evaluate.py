"""CLIP-based evaluation: CLIPScore and text->image retrieval recall."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .embed import TEXT_ENCODER_ID, get_device


def load_clip(model_id: str = TEXT_ENCODER_ID, device: str | None = None):
    device = device or get_device()
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor


@torch.no_grad()
def embed_images(images: list[Image.Image], model, processor, batch_size: int = 32) -> torch.Tensor:
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(images), batch_size):
        inputs = processor(images=images[i : i + batch_size], return_tensors="pt").to(device)
        out.append(F.normalize(model.get_image_features(**inputs), dim=-1).cpu())
    return torch.cat(out)


@torch.no_grad()
def embed_texts(texts: list[str], model, processor, batch_size: int = 64) -> torch.Tensor:
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(texts), batch_size):
        inputs = processor(
            text=texts[i : i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)
        out.append(F.normalize(model.get_text_features(**inputs), dim=-1).cpu())
    return torch.cat(out)


def clip_scores(images: list[Image.Image], prompts: list[str], model, processor) -> list[float]:
    """CLIPScore (Hessel et al.) as implemented by torchmetrics: 100 * max(cos, 0)."""
    img = embed_images(images, model, processor)
    txt = embed_texts(prompts, model, processor)
    cos = (img * txt).sum(-1)
    return (100 * cos.clamp(min=0)).tolist()


def recall_at_k(text_emb: torch.Tensor, image_emb: torch.Tensor, ks=(1, 5, 10)) -> dict:
    """Text->image retrieval where caption i belongs to image i."""
    sims = text_emb @ image_emb.T  # [N_text, N_image]
    ranks = sims.argsort(dim=1, descending=True)
    target = torch.arange(sims.shape[0]).unsqueeze(1)
    hit_pos = (ranks == target).float().argmax(dim=1)  # position of the correct image
    return {f"R@{k}": float((hit_pos < k).float().mean()) for k in ks}
