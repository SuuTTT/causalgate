"""Weak / strong views of a multimodal event sample for consistency regularisation and masked reconstruction.

Weak view   : small jitter + per-channel scaling (label-preserving).
Strong view : larger jitter/scaling, random contiguous time mask (also the reconstruction target mask), optional
              segment permutation (TS-TCC style), and random drop-out of text documents / graph neighbours.
All operations are batched tensor ops so they run on the device inside the training step.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch


def _gen_like(gen: Optional[torch.Generator], device: torch.device) -> Optional[torch.Generator]:
    return gen if (gen is None or gen.device == device) else None


def weak_augment(x: torch.Tensor, jitter: float = 0.02, scale: float = 0.05, gen: Optional[torch.Generator] = None) -> torch.Tensor:
    g = _gen_like(gen, x.device)
    B, L, F = x.shape
    s = 1.0 + scale * torch.randn(B, 1, F, generator=g, device=x.device)
    j = jitter * torch.randn(B, L, F, generator=g, device=x.device)
    return x * s + j


def random_time_mask(x: torch.Tensor, frac: float, gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """One contiguous masked span per sample. Returns a bool mask (B, L); True = masked."""
    g = _gen_like(gen, x.device)
    B, L, _ = x.shape
    span = max(1, int(round(frac * L)))
    start = torch.randint(0, L - span + 1, (B, 1), generator=g, device=x.device)
    ar = torch.arange(L, device=x.device).unsqueeze(0)
    return (ar >= start) & (ar < start + span)


def segment_permute(x: torch.Tensor, n_segments: int, gen: Optional[torch.Generator] = None) -> torch.Tensor:
    if n_segments <= 1:
        return x
    g = _gen_like(gen, x.device)
    B, L, F = x.shape
    seg = L // n_segments
    if seg == 0:
        return x
    core = x[:, : seg * n_segments].reshape(B, n_segments, seg, F)
    perm = torch.argsort(torch.rand(B, n_segments, generator=g, device=x.device), dim=1)
    core = torch.gather(core, 1, perm.view(B, n_segments, 1, 1).expand(-1, -1, seg, F))
    return torch.cat([core.reshape(B, seg * n_segments, F), x[:, seg * n_segments :]], 1)


def strong_augment(x: torch.Tensor, jitter: float = 0.05, scale: float = 0.2, time_mask_frac: float = 0.3,
                   n_segments: int = 0, gen: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    xa = weak_augment(x, jitter, scale, gen)
    xa = segment_permute(xa, n_segments, gen)
    mask = random_time_mask(x, time_mask_frac, gen)
    xa = xa.masked_fill(mask.unsqueeze(-1), 0.0)
    return xa, mask


def augment_batch(batch: Dict[str, torch.Tensor], mode: str, cfg: Optional[Dict] = None,
                  gen: Optional[torch.Generator] = None) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """Shallow-copy the batch and replace the model-visible modalities with an augmented view.

    Returns (augmented_batch, time_mask or None). Targets (y, impact, fut_*) are untouched.
    """
    cfg = cfg or {}
    nb = dict(batch)
    if mode == "weak":
        nb["price"] = weak_augment(batch["price"], cfg.get("weak_jitter", 0.02), cfg.get("weak_scale", 0.05), gen)
        if "lob" in nb:
            nb["lob"] = weak_augment(batch["lob"], cfg.get("weak_jitter", 0.02), cfg.get("weak_scale", 0.05), gen)
        return nb, None
    if mode != "strong":
        raise ValueError(mode)
    nb["price"], mask = strong_augment(batch["price"], cfg.get("strong_jitter", 0.05), cfg.get("strong_scale", 0.2),
                                       cfg.get("time_mask_frac", 0.3), cfg.get("n_segments", 0), gen)
    if "lob" in nb:
        nb["lob"] = batch["lob"].masked_fill(mask.unsqueeze(-1), 0.0)
    g = _gen_like(gen, batch["price"].device)
    p_drop = float(cfg.get("drop_doc_p", 0.3))
    if "text_mask" in nb and p_drop > 0:
        keep = torch.rand(batch["text_mask"].shape, generator=g, device=batch["text_mask"].device) >= p_drop
        nb["text_mask"] = batch["text_mask"] & keep
    if "graph_mask" in nb and p_drop > 0:
        keep = torch.rand(batch["graph_mask"].shape, generator=g, device=batch["graph_mask"].device) >= p_drop
        nb["graph_mask"] = batch["graph_mask"] & keep
    return nb, mask
