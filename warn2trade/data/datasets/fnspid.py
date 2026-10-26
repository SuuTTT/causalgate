"""D3 FNSPID (Dong, Fan et al., KDD 2024). STATUS: PUBLIC. https://huggingface.co/datasets/Zihan1004/FNSPID
~29.7M news articles + 29.7M price records, 4,775 S&P 500 tickers, 1999-2023 (daily prices; news timestamps).

Recipe
    news -> embed title+summary with FinBERT offline, align to the next tradable bar (strict t+1 alignment,
            no same-day leakage for after-close news) ; M = 8 most recent articles, age in bars
    labels -> abnormal_move_labels(horizon=3) OR event-table labels from earnings-surprise / guidance cuts
    graph  -> co-mention graph: two tickers are neighbours if co-mentioned in >= k articles in the trailing 60 days
"""
from __future__ import annotations
from ..base import BaseDatasetBuilder
from ..schema import Panel
from ...utils.registry import DATASETS
from ._common import require_files


@DATASETS.register("fnspid")
class FNSPIDBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "fnspid", "daily", 252, (1, 2, 3, 5, 10, 20)

    def build_panel(self) -> Panel:
        require_files(self.root, ["fnspid"], "huggingface-cli download Zihan1004/FNSPID --repo-type dataset --local-dir data_raw/fnspid")
        raise NotImplementedError("Implement: FNSPID parquet -> embeddings cache -> Panel.")
