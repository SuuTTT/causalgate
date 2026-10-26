"""Baseline slots that wrap external code. Each records the paper, the official repo and its verification status.
`status` = 'wrapper-needed' means the paper and code exist and only the adapter to the EventDataset is missing;
'placeholder-unverified' means we could not verify a paper with that name (see docs/benchmark_audit.md) and the
slot should be filled with the suggested verified substitute before any number is reported.
"""
from __future__ import annotations
from .event_base import BaseBaseline, register_event_baseline


def _stub(name, paper, repo, status, substitute=""):
    @register_event_baseline(name)
    class _Stub(BaseBaseline):
        pass
    _Stub.__name__ = name
    _Stub.name, _Stub.paper, _Stub.official_repo, _Stub.status = name, paper, repo, status
    _Stub.substitute = substitute

    def fit(self, train, val=None):
        raise NotImplementedError(f"[{name}] status={status}. paper: {paper}. repo: {repo}. "
                                  f"{'Suggested substitute: ' + substitute if substitute else ''} "
                                  f"Implement fit/predict against EventDataset batches (see base.py).")
    _Stub.fit = fit
    _Stub.predict = fit
    return _Stub


# --- verified papers, adapter needed ------------------------------------------------------------------
_stub("master", "Li et al. MASTER: Market-Guided Stock Transformer for Stock Price Forecasting. AAAI 2024", "https://github.com/SJTU-DMTai/MASTER", "wrapper-needed")
_stub("man_sf", "Sawhney et al. Deep Attentive Learning for Stock Movement Prediction from Social Media Text and Company Correlations (MAN-SF). EMNLP 2020", "https://github.com/midas-research/man-sf-emnlp", "wrapper-needed")
_stub("patchtst", "Nie et al. A Time Series is Worth 64 Words (PatchTST). ICLR 2023", "https://github.com/yuqinie98/PatchTST", "wrapper-needed")
_stub("trademaster_ppo", "Sun et al. TradeMaster: A Holistic Quantitative Trading Platform Empowered by RL. NeurIPS 2023 D&B", "https://github.com/TradeMaster-NTU/TradeMaster", "wrapper-needed")
_stub("fingpt_sentiment", "Yang, Liu, Wang. FinGPT: Open-Source Financial LLMs. 2023 (arXiv:2306.06031) - sentiment -> rule", "https://github.com/AI4Finance-Foundation/FinGPT", "wrapper-needed")
_stub("stockformer", "Gao et al. StockFormer: Learning Hybrid Trading Machines with Predictive Coding. IJCAI 2023", "https://github.com/gsyyysg/StockFormer", "wrapper-needed")
_stub("time_llm", "Jin et al. Time-LLM: Time Series Forecasting by Reprogramming LLMs. ICLR 2024", "https://github.com/KimMeen/Time-LLM", "wrapper-needed")
_stub("finagent", "Zhang et al. A Multimodal Foundation Agent for Financial Trading (FinAgent). KDD 2024", "https://github.com/DVampire/FinAgent", "wrapper-needed")
_stub("units", "Gao et al. UniTS: A Unified Multi-Task Time Series Model. NeurIPS 2024", "https://github.com/mims-harvard/UniTS", "wrapper-needed")
_stub("macrohft", "Zong et al. MacroHFT: Memory Augmented Context-aware RL on High Frequency Trading. KDD 2024", "https://github.com/ZONG0004/MacroHFT", "wrapper-needed")
_stub("cpd_dmn", "Wood, Roberts, Zohren. Slow Momentum with Fast Reversion (CPD + Deep Momentum Network). JFDS 2022", "https://github.com/kieranjwood/slow-momentum-fast-reversion", "wrapper-needed")
_stub("deep_ofi", "Kolm, Turiel, Westray. Deep Order Flow Imbalance. Mathematical Finance 2023", "", "wrapper-needed")
_stub("lob_cascade_contrastive", "Lin, Yang. Detecting Multilevel Manipulation from LOB via Cascaded Contrastive Representation Learning. 2025 (arXiv:2508.17086)", "", "wrapper-needed")
# --- names from the task brief that could not be verified ------------------------------------------------
_stub("findiffusion", "'FinDiffusion' (2025) - no paper with this title verified", "", "placeholder-unverified", "LOB diffusion simulators: Hultin et al. 2023 'A generative model of a limit order book using recurrent neural networks' or Nagy et al. 2023 'Generative AI for end-to-end LOB modelling' (arXiv:2309.00638)")
_stub("causal_finrl", "'Causal-FinRL' (2025) - no paper with this title verified", "", "placeholder-unverified", "MacroHFT (KDD 2024) or Asadi & Safabakhsh 2026 regime-aware RL (arXiv:2608.28252)")
_stub("finkgc", "'FinKGC / Graph-LLM Financial Auditor' (2025) - no paper with this title verified", "", "placeholder-unverified", "Xiang et al. WWW 2025 dynamic-graph spoofing (arXiv:2510.05562) or PERSEUS (arXiv:2503.01686)")
_stub("dynoalpha_2026", "'DynoAlpha-2026' - not found; fill with a verified 2026 paper", "", "placeholder-unverified", "PumpSense (IEEE ICBC 2026, arXiv:2605.09431) for the manipulation track; Al Ridhawi et al. 2026 AE-gated dual transformer + SAC (arXiv:2603.19136) for the regime track")
_stub("omnifin_llm_2026", "'OmniFin-LLM (2026)' - not found; fill with a verified 2026 paper", "", "placeholder-unverified", "Yang 2026 'When Valid Signals Fail: LLM features -> RL policies' (arXiv:2604.10996); Asadi & Safabakhsh 2026 (arXiv:2608.28252)")
