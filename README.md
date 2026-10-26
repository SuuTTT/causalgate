# Warn2Trade — closing the loop from anomaly warnings to executable trades

Code skeleton for the paper plan in [`docs/paper_plan.md`](docs/paper_plan.md). Dataset and baseline verification status is
in [`docs/benchmark_audit.md`](docs/benchmark_audit.md); the literature survey is [`paper_catalogue.md`](paper_catalogue.md).

```
warn2trade/
  data/        Panel schema, EventDataset (lazy windows), weak labels + impact curves, purged walk-forward splits,
               synthetic generator, datasets/ (10 benchmark builders, documented stubs)
  models/      encoders (price / LOB / text / graph), MultimodalAnomalyEncoder, TemporalDynamicsExtrapolator,
               RiskAwarePolicy + BayesPlugInPolicy, losses, Warn2Trade (with ablation switches)
  backtest/    CostModel (fees + half-spread + sqrt impact + delay), differentiable event backtest, portfolio backtest,
               metrics (AUC-PR, F1, P@K, WLT, FDP, EAR, CR, ARR, SR, SoR, MDD, Calmar, IR, win rate, turnover, PSR, DSR)
  baselines/   shared interface; zscore rules, DLinear, open-loop detect_then_rule implemented; 13 verified papers as
               wrapper stubs; 5 unverifiable names from the brief as explicit placeholders
  train/       stage-wise trainer (detect -> extrap -> policy -> e2e) with validation-based model selection
configs/       warn2trade/ (this track), ablations/*.yaml (11), datasets/*.yaml (11), baselines/*.yaml
scripts/       train.py, run_baseline.py, run_all.sh, make_tables.py, sweep_cost.py, backtest.py, prepare_data.py
tests/         metrics, backtest gradient/freeze semantics, model forward shapes
```

## Quick start (local CPU smoke test, ~2 min per run)
```bash
pip install -r requirements.txt
python3 scripts/train.py --dataset synthetic --seed 42 --max-folds 1 --device cpu
```
```bash
python3 scripts/run_baseline.py --baseline detect_then_rule --dataset synthetic --seed 42 --max-folds 1 --device cpu
```
```bash
python3 scripts/make_tables.py --runs runs --bps 5 --fmt markdown
```
Each run writes `runs/<dataset>/<method>/seed<k>/fold<j>.json` (metrics at every cost level in `eval.cost_grid_bps`) and
`trades_fold<j>.csv` (one row per decision point: score, weight, holding horizon, schedule, net PnL, frozen flag, bars to peak).

## Full grid on the cluster
```bash
rsync -av --exclude runs --exclude data_raw . c225:/date/zyy/TradingSignal/
```
```bash
bash scripts/run_all.sh stocknet lobster fnspid edgar crypto_pd csi300_limit sp500_tick qlib_alpha158 wsb_squeeze fin_graph
```
`run_all.sh` loops datasets × seeds {42, 2024, 2025, 2026} × {warn2trade, 9 ablations, 4 baselines}. Deterministic flags are set in
`warn2trade/utils/seed.py`; export `CUBLAS_WORKSPACE_CONFIG=:4096:8` on CUDA.

## Adding a dataset
Implement `build_panel()` in `warn2trade/data/datasets/<name>.py` returning a validated `Panel` (see `data/schema.py`), register
it, add `configs/datasets/<name>.yaml`. Everything downstream (windows, labels, splits, model, backtest, metrics) is shared.
Document embeddings are computed offline by `scripts/prepare_data.py` so no LLM runs during training.

## Adding a baseline
Subclass `warn2trade.baselines.base.BaseBaseline`, implement `fit(train_loader, val_loader)` and
`predict(batch) -> {"a_prob", "w", "hold_p", "kappa_p"[, "abstain"]}`. The event backtester and the metric matrix are then identical
to the main model's.

## Shared checkout notice
Other experiment tracks (SparseWarn2Trade, StressGuard) are developed in this same directory. Everything specific to this
track lives under `configs/warn2trade/` and `warn2trade/baselines/event_base.py`; `configs/default.yaml` and
`warn2trade/baselines/base.py` belong to the other tracks and are not read by `scripts/train.py` / `scripts/run_baseline.py`.

## Conventions that matter for the paper
- Decision at bar `t`, fill at close of `t + latency` (latency ≥ 1). Untradable entry ⇒ no trade (counted against EAR); untradable exit ⇒ delayed.
- Training charges 5 bps + half-spread + square-root impact (strategy notional = 5 % of ADV); evaluation re-prices every trade at 1/5/10/15/30 bps.
- Each asset owns a 1/N capital slot; trade PnL is booked on the exit bar; equal-weight universe is the IR benchmark.
- Purge ≥ max(horizon) + latency + max schedule, plus an embargo, between train/val/test windows.
- Negative bars are sub-sampled in the training fold only (`data.neg_subsample`); validation and test are never sub-sampled.

## CausalGate track (look-ahead / leakage audit; append-only note, 2026-10-03)
Lives in `causalgate/`, `configs/causalgate/`, `scripts/cg_*.py`, `tests/test_causalgate.py`, `results/causalgate/`; plan in
[`docs/causalgate_research_plan.md`](docs/causalgate_research_plan.md), results ledger in [`docs/causalgate_results.md`](docs/causalgate_results.md).
```bash
python3 tests/test_causalgate.py --device cpu
```
```bash
python3 scripts/cg_run.py --dataset cg_planted --baseline causalgate --protocol causal --seed 42 --device cpu --max-folds 1 --smoke
```
Protocol hashes are frozen once in `configs/causalgate/PROTOCOL_HASH` (committed) and every run verifies them. Do not
re-freeze as part of a run: `--freeze` is a no-op when nothing changed and refuses to overwrite a differing freeze;
a deliberate re-freeze after changing a protocol is `python3 scripts/cg_run.py --freeze --force-refreeze` (the old
hashes go to `PROTOCOL_HASH.history`; say so in the paper and re-run every affected test fold). L8 is an audit-card
flag, not a one-switch run. P-O ('published') runs use each baseline's own `published_norm` / `published_threshold`
(class `published_protocol` > baseline yaml > class default), recorded per fold as `published_conventions`.

## CrossPump track (cross-platform pump precursors; append-only note, 2026-10-03)
Lives in `crosspump/`, `configs/crosspump/`, `scripts/xp_*.py`, `tests/test_crosspump.py`, `results/crosspump/`; plan in
[`docs/crosspump_research_plan.md`](docs/crosspump_research_plan.md), ledger in [`docs/crosspump_results.md`](docs/crosspump_results.md).
```bash
python3 tests/test_crosspump.py
```
```bash
python3 scripts/xp_run.py --dataset synth --baseline crosspump --seed 42 --device cpu --smoke --max-folds 1
```

## HazardHedge track (DeFi liquidation / rug-pull risk as time-to-event; append-only note, 2026-10-03)
Lives in `hazardhedge/`, `configs/hazardhedge/`, `scripts/hh_*.py`, `tests/test_hazardhedge.py`, `results/hazardhedge/`; plan in
[`docs/hazardhedge_research_plan.md`](docs/hazardhedge_research_plan.md), results ledger in [`docs/hazardhedge_results.md`](docs/hazardhedge_results.md).
```bash
python3 tests/test_hazardhedge.py --device cpu
```
```bash
python3 scripts/hh_run.py --dataset synthetic_lending --method hazardhedge --seed 42 --device cpu --smoke
```

## ManipArena track (adversarial market simulation as a manipulation benchmark; append-only note, 2026-10-04)
Lives in `maniparena/`, `configs/maniparena/`, `scripts/ma_*`, `tests/test_maniparena.py`, `tests/maniparena_parts/`,
`results/maniparena/`; plan in [`docs/maniparena_research_plan.md`](docs/maniparena_research_plan.md), ledger in
[`docs/maniparena_results.md`](docs/maniparena_results.md), prompt in [`docs/maniparena_session_prompt.md`](docs/maniparena_session_prompt.md).
```bash
python3 tests/test_maniparena.py --parts
```
```bash
python3 scripts/ma_cotrain.py --market default --seed 42 --device cpu --smoke --workers 2
```

## EndoExo track (mechanism-aware early warning: endogenous build-up vs exogenous shock, response routing; append-only note, 2026-10-04)
Lives in `endoexo/`, `configs/endoexo/`, `scripts/ee_*`, `tests/test_endoexo.py`, `tests/endoexo_parts/`, `results/endoexo/`;
contract in [`endoexo/CONTRACT.md`](endoexo/CONTRACT.md), plan in [`docs/endoexo_research_plan.md`](docs/endoexo_research_plan.md),
status + results ledger in [`docs/endoexo_results.md`](docs/endoexo_results.md), baseline conventions in
[`endoexo/baselines/CONVENTIONS.md`](endoexo/baselines/CONVENTIONS.md).
```bash
python3 tests/test_endoexo.py --device cpu            # -k <part> for one component
```
```bash
python3 scripts/ee_run.py --dataset ee_synthetic --baseline endoexo --seed 42 --device cpu --max-folds 1 --smoke
python3 scripts/ee_make_tables.py
```
Protocol hashes live in `configs/endoexo/PROTOCOL_HASH` (`python3 scripts/ee_run.py --freeze` rewrites them; every run
verifies them). Lead times use the per-bar alarm series (trade rows OR the method's own `alarm` key); synthetic smoke runs
are plumbing checks, not findings. Real datasets need `scripts/ee_prepare_data.py --dataset <name> --download` first.
