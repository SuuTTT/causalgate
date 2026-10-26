"""D2 LOBSTER level-10 LOB (Huang & Polak). STATUS: LICENSED (academic subscription, https://lobsterdata.com).
Free substitute with the same loader: FI-2010 (Ntakaris et al. 2018, 5 Finnish stocks, 10 days, normalised).
Spoofing labels: no public ground truth -> use (a) the synthetic spoof-injection protocol of Lin & Yang 2025
(arXiv:2508.17086, cascaded contrastive LOB) or (b) rule labels: large far-from-touch orders cancelled within
delta ms followed by an opposite-side trade. Report both.

Recipe
    sample LOB at 100 ms -> bars; mid = (bid1 + ask1)/2 ; lob = 10 levels x [bid_px rel, bid_sz, ask_px rel, ask_sz]
    price_feat = [OFI (Kolm et al. 2023), mid ret, trade imbalance, queue imbalance, cancel ratio]
    liq = [half-spread/mid, depth at touch / median depth, bar volume / daily volume]
    tradable = False when spread > 20 ticks or no quotes (open/close auctions)
    bars_per_year = 252 * 6.5 * 3600 * 10 (100 ms bars)
"""
from __future__ import annotations
from ..base import BaseDatasetBuilder
from ..schema import Panel
from ...utils.registry import DATASETS
from ._common import require_files


@DATASETS.register("lobster")
class LOBSTERBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "lobster", "lob", 252 * 6 * 3600 * 10, (10, 20, 50, 100, 300, 600)

    def build_panel(self) -> Panel:
        require_files(self.root, ["lobster"], "Place LOBSTER *_message_10.csv / *_orderbook_10.csv under data_raw/lobster/ "
                      "(or FI-2010 under data_raw/fi2010/ and set dataset.variant=fi2010).")
        raise NotImplementedError("Implement: parse message/orderbook csv -> 100ms bars -> Panel.")
