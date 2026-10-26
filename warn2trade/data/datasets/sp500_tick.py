"""D7 S&P 500 intraday crash / mini-flash-crash events. STATUS: LICENSED (NYSE TAQ) or CONSTRUCTED (Polygon/IEX).
Event labels: mini flash crashes per Johnson et al. (2013) definition (>=10 ticks in one direction within 1.5 s,
reverted) or the Liu, Xu, Zheng (SSRN 4992077) protocol; macro events: 2010-05-06, 2015-08-24, 2020-03 circuit breakers.

Recipe
    bars = 1 s or 1 min; mid from NBBO; price_feat = [ret, OFI, trade intensity, quote-update rate, spread z]
    tradable = False during LULD halts and market-wide circuit breakers; liq from NBBO depth
"""
from __future__ import annotations
from ..base import BaseDatasetBuilder
from ..schema import Panel
from ...utils.registry import DATASETS
from ._common import require_files


@DATASETS.register("sp500_tick")
class SP500TickBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "sp500_tick", "1min", 252 * 390, (1, 2, 5, 10, 30, 60)

    def build_panel(self) -> Panel:
        require_files(self.root, ["sp500_tick"], "Export TAQ/Polygon 1-min bars to data_raw/sp500_tick/{ticker}.parquet")
        raise NotImplementedError("Implement: minute bars + halt calendar -> Panel.")
