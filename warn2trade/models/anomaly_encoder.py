"""Stage 1 - Multimodal Anomaly Encoder (MAE): modality tokens -> cross-modal fusion -> (anomaly logit, latent z)."""
from __future__ import annotations
from typing import Any, Dict
import torch
import torch.nn as nn

from .encoders import PatchTSEncoder, LOBConvEncoder, TimeDecayTextEncoder, NeighborAttentionEncoder, liq_transform


class MultimodalAnomalyEncoder(nn.Module):
    def __init__(self, dims: Dict[str, int], cfg: Dict[str, Any]):
        super().__init__()
        d = int(cfg.get("d_model", 128))
        self.d_model = d
        self.use_lob = bool(cfg.get("use_lob", True)) and dims.get("lob", 0) > 0
        self.use_text = bool(cfg.get("use_text", True)) and dims.get("text", 0) > 0
        self.use_graph = bool(cfg.get("use_graph", True)) and dims.get("graph", 0) > 0
        self.price_enc = PatchTSEncoder(dims["price"], d, cfg.get("patch_len", 8), cfg.get("stride", 4),
                                        cfg.get("price_layers", 2), cfg.get("n_heads", 4), cfg.get("dropout", 0.1))
        self.liq_enc = nn.Sequential(nn.Linear(dims["liq"], d), nn.GELU())
        if self.use_lob:
            self.lob_enc = LOBConvEncoder(dims["lob"], d, dropout=cfg.get("dropout", 0.1))
        if self.use_text:
            self.text_enc = TimeDecayTextEncoder(dims["text"], d, cfg.get("text_decay", 0.1), cfg.get("dropout", 0.1))
        if self.use_graph:
            self.graph_enc = NeighborAttentionEncoder(dims["graph"], d, cfg.get("n_heads", 4), cfg.get("dropout", 0.1))
        n_tokens = 1 + 2 + int(self.use_lob) + int(self.use_text) + int(self.use_graph)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.mod_emb = nn.Parameter(torch.zeros(1, n_tokens, d))
        nn.init.trunc_normal_(self.mod_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.get("n_heads", 4), 4 * d, cfg.get("dropout", 0.1), batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, cfg.get("fusion_layers", 2))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.dir_head = nn.Linear(d, 1)                                  # sign of the peak-horizon impact (aux target)
        self.register_buffer("center", torch.zeros(d))                   # one-class (Deep SVDD) centre, set after warm-up

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = batch["price"].shape[0]
        price_tok = self.price_enc(batch["price"])
        toks = [self.cls.expand(B, -1, -1).squeeze(1), price_tok, self.liq_enc(liq_transform(batch["liq"]))]
        if self.use_lob:
            toks.append(self.lob_enc(batch["lob"]))
        if self.use_text:
            toks.append(self.text_enc(batch["text"], batch["text_mask"], batch["text_age"]))
        if self.use_graph:
            toks.append(self.graph_enc(price_tok, batch["graph"], batch["graph_mask"]))
        h = torch.stack(toks, 1) + self.mod_emb
        h = self.fusion(h)
        z = self.norm(h[:, 0])
        return {"logit": self.head(z).squeeze(-1), "dir_logit": self.dir_head(z).squeeze(-1), "z": z, "tokens": h}

    @torch.no_grad()
    def set_center(self, z_normal: torch.Tensor, eps: float = 0.1) -> None:
        c = z_normal.mean(0)
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c >= 0)] = eps
        self.center.copy_(c)
