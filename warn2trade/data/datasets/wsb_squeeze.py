"""D9 WallStreetBets / StockTwits coordinated short-squeeze. STATUS: PUBLIC (Reddit) / API (StockTwits).
Reddit: Kaggle 'Reddit WallStreetBets Posts' (G. Preda) and Pushshift dumps 2020-2021; StockTwits via API (rate-limited).
Short-interest: FINRA bi-monthly (public) ; prices: Yahoo daily / Polygon minute.
Related evidence: Buz & de Melo 2021 (arXiv:2105.02728), Warkulat & Pelster 2024, Neela 2025 AIMM (arXiv:2512.16103).

Recipe
    labels = squeeze events: short interest > 40% float AND 5-day return > 50% (GME, AMC, BB, KOSS, EXPR, ...);
             bars_to_peak from the realised peak close; lead window 5 days
    text = post embeddings (M=16, age in days); graph = co-mention graph of tickers in the same posts
    tradable = False on broker-restricted days (2021-01-28 .. 02-04 for the restricted list)
"""
from __future__ import annotations
from ..base import BaseDatasetBuilder
from ..schema import Panel
from ...utils.registry import DATASETS
from ._common import require_files


@DATASETS.register("wsb_squeeze")
class WSBSqueezeBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "wsb_squeeze", "daily", 252, (1, 2, 3, 5, 10, 20)

    def build_panel(self) -> Panel:
        require_files(self.root, ["wsb"], "Download the Kaggle WSB posts csv to data_raw/wsb/ and FINRA short interest to data_raw/wsb/short_interest/")
        raise NotImplementedError("Implement: posts + prices + short interest -> Panel.")
