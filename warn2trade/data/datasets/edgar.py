"""D4 SEC EDGAR 10-K / 10-Q MD&A + restatements. STATUS: PUBLIC text, LICENSED labels.
Text: EDGAR full-text (https://www.sec.gov/edgar), Loughran-McDonald 10-X parsed files, or EDGAR-CORPUS (Loukas et al. 2021).
Labels: (a) AAER accounting-fraud labels from Bao, Ke, Yu, Zhang & Zhang (JAR 2020, public GitHub), or
        (b) Audit Analytics restatements (LICENSED). Price impact: CRSP (LICENSED) or Yahoo daily (public).

Recipe
    one decision point per filing date (t = first bar after acceptance datetime); text = MD&A chunk embeddings (M=32)
    impact horizons in trading days after filing; labels = fraud/restated fiscal year flag
    graph = auditor / industry (SIC-2) / supply-chain neighbours
"""
from __future__ import annotations
from ..base import BaseDatasetBuilder
from ..schema import Panel
from ...utils.registry import DATASETS
from ._common import require_files


@DATASETS.register("edgar")
class EdgarBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "edgar", "daily", 252, (1, 3, 5, 10, 20, 60)

    def build_panel(self) -> Panel:
        require_files(self.root, ["edgar"], "See docs/benchmark_audit.md#d4 for the download recipe (EDGAR-CORPUS + AAER labels).")
        raise NotImplementedError("Implement: filings -> MD&A embeddings -> filing-date panel -> Panel.")
