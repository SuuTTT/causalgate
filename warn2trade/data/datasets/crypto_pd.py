"""D5 Crypto pump-and-dump. STATUS: PUBLIC (Telegram P&D) / PLACEHOLDER ("DEX-Anomalies" is not a known dataset).
Primary: La Morgia, Mei, Sassi, Stefa (ACM TOIT 2023; arXiv:2105.00733) Telegram pump dataset + Binance trades,
         https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset . Hu et al. SIGMOD 2023 (arXiv:2204.12929) target-coin
         sequences. Rug-pull substitutes for the DEX slot: Yaremus et al. 2025 (TON, arXiv:2509.01168), Cao et al. 2026 (BSC).

Recipe
    bars = 5 s or 1 min trades around each announced pump (+-6 h); mid = last trade; labels = 1 from
    (announcement - lead) to pump peak, bars_to_peak from the curated peak timestamp (exact, can be negative)
    text = Telegram message embeddings (M=8); graph = coins co-pumped by the same channel (K=8)
    liq: half-spread from best quotes if available else Roll estimator; ADV from 24 h volume
    tradable = False during exchange halts / zero-volume bars
"""
from __future__ import annotations
from ..base import BaseDatasetBuilder
from ..schema import Panel
from ...utils.registry import DATASETS
from ._common import require_files


@DATASETS.register("crypto_pd")
class CryptoPumpDumpBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "crypto_pd", "1min", 365 * 24 * 60, (1, 2, 5, 10, 30, 60)

    def build_panel(self) -> Panel:
        require_files(self.root, ["crypto_pd"], "git clone https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset data_raw/crypto_pd")
        raise NotImplementedError("Implement: pump table + trades -> 1min bars around events -> Panel.")
