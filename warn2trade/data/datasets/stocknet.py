"""D1 StockNet (Xu & Cohen, ACL 2018). STATUS: PUBLIC. https://github.com/yumoxu/stocknet-dataset
88 US stocks, 2014-01-01 .. 2016-01-01, daily OHLCV + tweets per stock-day.

Recipe
    price/   -> mid = adj close; price_feat = [ret/sigma, vol z, hl range, gap]
    tweet/   -> embed each tweet with FinBERT (or a frozen LLM) OFFLINE into text/{ticker}/{date}.npy; the builder
                pads M=16 most recent tweets per day (age = days since tweet) into Panel.text
    labels   -> abnormal_move_labels(horizon=5, k_ret=2.5, k_vol=2.0); no curated manipulation table exists.
    graph    -> sector co-membership from the dataset's industry file (K=8 neighbours, Dg = price_feat dim)
    liq      -> half-spread proxy = 0.5 * (high-low)/close * 0.1 ; ADV fraction from dollar volume
"""
from __future__ import annotations
from ..base import BaseDatasetBuilder
from ..schema import Panel
from ...utils.registry import DATASETS
from ._common import require_files


@DATASETS.register("stocknet")
class StockNetBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "stocknet", "daily", 252, (1, 2, 3, 5, 10, 20)

    def build_panel(self) -> Panel:
        require_files(self.root, ["stocknet/price/raw", "stocknet/tweet/preprocessed"],
                      "git clone https://github.com/yumoxu/stocknet-dataset data_raw/stocknet && "
                      "python scripts/prepare_data.py --dataset stocknet  (embeds tweets, writes cache/stocknet.npz)")
        raise NotImplementedError("Implement: load cache/stocknet.npz -> Panel (see module docstring).")
