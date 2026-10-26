"""D10 Multi-relational financial entity graph + disclosures. STATUS: PLACEHOLDER.
"FinWeb-Multimodal Fraud/Scam Dataset" could not be verified as a public dataset. Substitutes with the same
(graph + text + price) structure:
    * DGraph-Fin (Huang et al., NeurIPS 2022 D&B): 3.7M nodes, financial-fraud labels, no prices -> detection-only slot
    * Elliptic / Elliptic++ (Weber et al. 2019; Elmougy & Liu KDD 2023): Bitcoin transaction graph with illicit labels
    * CSMAR / Wind related-party graphs + CNINFO disclosures for A-shares (LICENSED)
We wire Elliptic++ (public) with BTC 1-h prices so that the trade loop is still closed, and report DGraph-Fin
in the detection-only table.
"""
from __future__ import annotations
from ..base import BaseDatasetBuilder
from ..schema import Panel
from ...utils.registry import DATASETS
from ._common import require_files


@DATASETS.register("fin_graph")
class FinGraphBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "fin_graph", "1h", 365 * 24, (1, 2, 4, 8, 24, 48)

    def build_panel(self) -> Panel:
        require_files(self.root, ["ellipticpp"], "Download Elliptic++ (https://github.com/git-disl/EllipticPlusPlus) to data_raw/ellipticpp/")
        raise NotImplementedError("Implement: transaction graph snapshots + BTC hourly bars -> Panel.")
