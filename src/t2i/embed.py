"""Text -> embedding pipeline using the CLIP ViT-L/14 text encoder.

This is the *same* encoder Stable Diffusion v1.x conditions on, so the embeddings
produced here are exactly what the UNet cross-attends to.
"""
from __future__ import annotations

import torch
from transformers import CLIPTextModel, CLIPTokenizer

TEXT_ENCODER_ID = "openai/clip-vit-large-patch14"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_text_encoder(model_id: str = TEXT_ENCODER_ID, device: str | None = None):
    device = device or get_device()
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    encoder = CLIPTextModel.from_pretrained(model_id).to(device).eval()
    return tokenizer, encoder


@torch.no_grad()
def encode_text(prompts: list[str], tokenizer: CLIPTokenizer, encoder: CLIPTextModel) -> dict:
    """Encode prompts.

    Returns
    -------
    dict with
      input_ids    [B, 77]       token ids (padded with the EOS token, as SD does)
      token_embeds [B, 77, 768]  per-token hidden states (UNet cross-attention input)
      pooled       [B, 768]      hidden state at the EOS position (sentence-level vector)
      truncated    list[bool]    True where the prompt exceeded 77 tokens and was cut
    """
    device = next(encoder.parameters()).device
    max_len = tokenizer.model_max_length
    untruncated = tokenizer(prompts, truncation=False).input_ids
    batch = tokenizer(
        prompts, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt"
    )
    out = encoder(batch.input_ids.to(device))
    return dict(
        input_ids=batch.input_ids.cpu(),
        token_embeds=out.last_hidden_state.cpu(),
        pooled=out.pooler_output.cpu(),
        truncated=[len(ids) > max_len for ids in untruncated],
    )


@torch.no_grad()
def encode_corpus(
    prompts: list[str], tokenizer, encoder, batch_size: int = 64
) -> torch.Tensor:
    """Pooled embeddings for a long list of captions -> [N, 768]."""
    chunks = []
    for i in range(0, len(prompts), batch_size):
        chunks.append(encode_text(prompts[i : i + batch_size], tokenizer, encoder)["pooled"])
    return torch.cat(chunks)
