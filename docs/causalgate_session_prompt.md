# Paste-ready prompt — CausalGate track (look-ahead and leakage risks)

Copy everything below the line into a fresh Claude Code session opened in
`/Users/zyy/package/anomalydetection/TradingSignal` (or on `c225:/date/zyy/TradingSignal`).

---

You are a senior financial-ML researcher and PyTorch engineer. We are writing a CCF-A paper (KDD / NeurIPS / ICLR / AAAI) in
financial trading anomaly detection, where trading signals are derived from anomaly signals. The research gap is **look-ahead
and leakage risk**: published anomaly-to-trade results standardise inputs with full-history statistics, read alarm thresholds
off the test-set score distribution, point-adjust labels, score text with language models trained after the test window, and use
today's universe. Competitions (DRW, Mitsui, Hull Tactical, 2025) leaked; the 2026 ADIA Structural Break Challenge: Real-Time
Edition made strictly online detection (one observation at a time, TS-AUC) a benchmark requirement.

**Blueprint.** Read `docs/causalgate_research_plan.md` first and follow it; it fixes the taxonomy (leakage channels L1–L8), the
three protocols (P-O as published, P-W purged walk-forward with offline normalisation/thresholds, P-C strictly causal), the
quantities (leakage premium LP, deployability gap, honest ranking), the method (Filtration Guard + streaming normalisation + causal
scorer + delayed-feedback adaptive-conformal alarm gate + Leakage Audit Card with the future-perturbation invariance test,
date-shift probe, martingale-difference placebo and DSR/placebo Sharpe plausibility), ten datasets, ten baselines (2021–2026, two
from 2026) and experiments E1–E8. Also read `docs/benchmark_audit.md`, `README.md`, `warn2trade/data/{schema,splits,labels}.py`,
`warn2trade/baselines/event_base.py`, `warn2trade/backtest/{engine,metrics}.py`.

**Repository rules (other paper tracks edit this checkout concurrently, no git).** Put all new code in `causalgate/`, configs in
`configs/causalgate/`, scripts as `scripts/cg_*.py`, tests in `tests/test_causalgate.py` (plain-script mode; system Python 3.9
has no pytest; `timeout` is not available in zsh), results in `results/causalgate/`. Shared modules under `warn2trade/` are
append-only or new-file-only; never rewrite `configs/default.yaml`, `warn2trade/baselines/base.py`, `schema.py`, `engine.py`.
Register baselines through `warn2trade.baselines.event_base.register_event_baseline` (first registration wins) and import
warn2trade pieces explicitly. Every method must emit the shared `out = {"a_prob","w","hold_p","kappa_p"[,"abstain"]}` dict so the
event backtester and the metric matrix stay identical across methods.

**Deliverables, in order.**
1. `causalgate/data/planted.py` (synthetic regime-switching panel with injected events and switchable leaks: global
   normalisation, test-quantile threshold, same-bar fill, label-time join), `causalgate/data/pit.py` (PITView + availability
   arrays appended as optional Panel fields), `causalgate/audit/fpit.py`. Smoke: FPIT flags each planted leak and passes on the
   causal pipeline. Run `python3 tests/test_causalgate.py --device cpu` and paste the output.
2. `causalgate/data/stream.py` (Welford/EWMA and P²-quantile normalisers with as-of stamps), `causalgate/models/scorer.py`
   (causal TCN, prequential AE, left matrix profile), `causalgate/models/gate.py` (ACI and delayed-ACI with τ = horizon + latency;
   α derived from the cost budget), `causalgate/models/causalgate.py` wired to the Warn2Trade policy head and
   `backtest/engine.py`. Smoke on the synthetic panel: realised alarm rate within ±0.02 of α under a variance regime switch.
3. Real loaders D1–D10 in `causalgate/data/loaders/` following the recipes in `warn2trade/data/datasets/*.py` docstrings, each
   emitting availability arrays (tweet/news/post timestamps, AAER release dates, as-of constituents, halt calendars, label lags).
   Public sets first (StockNet, FNSPID, FI-2010, Qlib Alpha158, La Morgia pump-and-dump). Download recipes go in
   `scripts/cg_prepare_data.py`; never fabricate data.
4. Baselines B1–B10 in `causalgate/baselines/`, each with `mode ∈ {published, causal}` and a convention sheet in
   `causalgate/baselines/CONVENTIONS.md` stating, with file/line pointers to the official code or paper equations, which of L1–L8
   the original uses and how the causal mode removes it. Use official code where it exists (Anomaly Transformer, DAMP,
   MemStream, DCdetector, MOMENT, CATCH); re-implement CPD-DMN, STAGE, Al Ridhawi et al. 2026 (arXiv:2603.19136) and
   Kończal & Połoczański 2026 (arXiv:2608.09576) from the papers and mark every unspecified choice.
5. Protocol runners `causalgate/eval/protocols.py` (P-O, P-W, P-C, P-C/ADIA), metrics `causalgate/eval/metrics.py` (LP per
   channel, deployability gap, TS-AUC, VUS-PR, Kendall τ, placebo Sharpe, alarm-rate error), audit modules, `scripts/cg_run_all.sh`
   for datasets × baselines × protocols × seeds {42, 2024, 2025, 2026}, `scripts/cg_make_tables.py`, `scripts/cg_audit.py`
   producing one Leakage Audit Card per (method, dataset). Freeze and hash the protocol before any test-fold run.
6. `docs/causalgate_results.md` with the E1–E8 tables (placeholders until runs exist), figure scripts, and a paper outline that
   positions against Zhang et al. 2026 (arXiv:2605.23959), Fonseca 2026 (arXiv:2607.04958), Nikolopoulos 2026
   (arXiv:2604.15531), El Halabi & Brandt 2026 (arXiv:2609.07251), Parrino et al. KDD 2026 (arXiv:2609.39215) and the
   LLM look-ahead line (Glasserman & Lin 2023; Sarkar & Vafa 2024; He et al. 2025; Kelly et al. 2026; Gao et al. 2025).

**Hard constraints.** Cite only papers whose DOI or arXiv id you have resolved; if a name in any brief cannot be verified, say so
and propose a verified substitute, never invent one. Report numbers only from runs that exist; use placeholders otherwise. Do not
claim STAGE or any baseline "leaks" until our re-implementation shows a positive leakage premium; call it protocol-ambiguous.
Check data licences (LOBSTER, TAQ, Kaggle competition files, ADIA data) and use the public substitutes from
`docs/benchmark_audit.md` where needed. Keep costs realistic (5 bps training; 1/5/10/15/30 bps evaluation; half-spread; sqrt
impact; latency ≥ 1), purge ≥ max(horizon) + latency + schedule, embargo on, seeds 42/2024/2025/2026.

**Environment.** Local Mac: system Python 3.9, torch 2.8, numpy 2.0, no CUDA (use `--device cpu` and `--max-folds 1` for smoke
tests). Cluster: `c225:/date/zyy/TradingSignal`; sync with
`rsync -av --exclude runs --exclude data_raw --exclude results . c225:/date/zyy/TradingSignal/`; export
`CUBLAS_WORKSPACE_CONFIG=:4096:8` on CUDA.

**Reporting.** After each deliverable: what was built, the exact smoke command and its output, what is still a stub, and any
place where you had to guess a baseline's protocol. Ask only when a decision changes the paper's claims; otherwise choose, state
the assumption, and continue.
