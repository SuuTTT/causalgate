"""Registered names for external baselines that must be wrapped from their upstream code (not re-implemented).

Verified references (check again before citing):
    time_llm   Jin et al., "Time-LLM: Time Series Forecasting by Reprogramming LLMs", ICLR 2024. github.com/KimMeen/Time-LLM
    fpt        Zhou et al., "One Fits All: Power General Time Series Analysis by Pretrained LM", NeurIPS 2023. github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All
    units      Gao et al., "UniTS: A Unified Multi-Task Time Series Model", NeurIPS 2024. github.com/mims-harvard/UniTS
    macrohft   Zong et al., "MacroHFT", KDD 2024 (arXiv:2406.14537). github.com/ZONG0004/MacroHFT
    lob_ccrl   Lin & Yang, cascaded contrastive LOB manipulation detection, arXiv:2508.17086 (code availability to confirm)
    deep_sad   Ruff et al., ICLR 2020 -> provided here as a trainer-flag baseline (configs/baselines/deep_sad.yaml)
    devnet     Pang et al., KDD 2019 -> trainer-flag baseline (lam_dev only)
    fixmatch   Sohn et al., NeurIPS 2020 -> trainer-flag baseline (configs/baselines/fixmatch_fin.yaml)
Names used in the project brief that do NOT correspond to a verifiable publication as named and must be replaced or
confirmed before submission: "DeLise-SparseNet" (DeLise 2023 is Deep SAD on TMX futures), "TradeMaster-WeakRL",
"FinAgent-Weak", "Causal-SemiRL", "SemiDyn-2026", "LabelFree-Alpha". See docs/RESEARCH_PLAN.md section 6.
"""
from __future__ import annotations
from .base import ScoreBaseline, SEMI_BASELINES

_UPSTREAM = {
    "time_llm": "https://github.com/KimMeen/Time-LLM",
    "fpt": "https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All",
    "units": "https://github.com/mims-harvard/UniTS",
    "macrohft": "https://github.com/ZONG0004/MacroHFT",
    "lob_ccrl": "arXiv:2508.17086",
}


def _make(name: str, url: str):
    class _Placeholder(ScoreBaseline):
        def fit(self, ds):
            raise NotImplementedError(f"Baseline {name!r}: wrap the upstream implementation ({url}) behind ScoreBaseline.fit/score. "
                                      "Few-shot adaptation protocol: freeze the backbone, fine-tune the head on the K anchors + confident normals.")

        def score(self, ds):
            raise NotImplementedError(name)
    _Placeholder.__name__ = f"{name}_placeholder"
    _Placeholder.name = name
    return _Placeholder


for _n, _u in _UPSTREAM.items():
    SEMI_BASELINES.register(_n)(_make(_n, _u))
