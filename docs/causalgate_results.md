# CausalGate — results ledger, figure scripts and paper outline

Companion of `docs/causalgate_research_plan.md`. **Every number in this file comes from a run that exists on disk**
(path given); everything else is a placeholder `[·]`. Protocol definitions are frozen in
`configs/causalgate/PROTOCOL_HASH` (re-frozen 2026-10-04 after the review removed `causal+L8` from the run set; 12 protocols, hash = sha256[:16] of the canonical JSON; the
other 12 hashes are unchanged and the previous freeze is kept in `PROTOCOL_HASH.history`). L8 is an audit-card flag, not a run.

Reproduce the smoke results on CPU:
```bash
python3 tests/test_causalgate.py --device cpu
```
```bash
python3 scripts/cg_planted_leaks.py --seeds 42,2024,2025,2026 --n-probes 4,8,16
```
```bash
python3 scripts/cg_e7_gate_sweep.py --seeds 42,2024,2025,2026
```
Full grid (cluster): `bash scripts/cg_run_all.sh --protocols published,walkforward,causal --seeds 42,2024,2025,2026 --datasets all --baselines all --one-switch --ablations`,
then `python3 scripts/cg_make_tables.py` and `python3 scripts/cg_figures.py`.

---

## 0. Implementation status (what exists, what is a stub)

| Component | File | Status |
|---|---|---|
| PIT availability + PITView + Filtration Guard + future perturbation | `causalgate/data/pit.py` | implemented, tested |
| Planted-leak testbed (A2) + reference pipeline with 4 switches | `causalgate/data/planted.py` | implemented, tested |
| FPIT (4 families + L7 fill-gap probe, localisation, stratified probes) | `causalgate/audit/fpit.py` | implemented, tested |
| Streaming normalisers (Welford / EWMA / P²-MAD) with as-of stamps | `causalgate/data/stream.py` | implemented, tested |
| Causal scorers: causal TCN, prequential AE, left matrix profile | `causalgate/models/scorer.py` | implemented, tested |
| Gates: fixed conformal, POT, ACI, delayed-ACI; α from cost budget | `causalgate/models/gate.py` | implemented, tested |
| CausalGate pipeline (event baseline `causalgate`) → shared `out` dict | `causalgate/models/causalgate.py` | implemented, tested (momentum head; learned head = `RiskAwarePolicy` trained with `event_pnl`) |
| Baseline scaffold (mode ∈ {published, causal}; L1/L2/L3 switches) | `causalgate/baselines/common.py` | implemented, tested |
| B1–B10 wrappers + convention sheets | `causalgate/baselines/b*.py`, `CONVENTIONS.md` (merged from `conventions_b2_b7.md` and `conventions_b1_b8_b9_b10.md`), `THIRD_PARTY.md` | all ten registered (`cg_cpd_dmn`, `cg_anomaly_transformer`, `cg_damp`, `cg_memstream`, `cg_dcdetector`, `cg_moment`, `cg_catch`, `cg_stage`, `cg_alridhawi`, `cg_konczal_evt`), each with `mode ∈ {published, causal}`, `conventions` L1–L8 with file/line or equation pointers, and an `unspecified` list. Status: **official** only when the vendored repo imports (none vendored here); B3 DAMP, B4 MemStream, B1 CPD-DMN, B8 STAGE (three Eq. 20 readings), B10 Kończal & Połoczański = `reimplemented`; B2 Anomaly Transformer, B5 DCdetector, B7 CATCH, B9 Al Ridhawi (SAC replaced by a bounded reward-driven update) = `reimplemented-fallback`; B6 MOMENT = `wrapper-needed` (needs `momentfm`; `cutoff_date` 2024-02-01 declared for the L8 check). Every guessed protocol detail is listed per class and in CONVENTIONS.md. |
| Loaders D1–D10 with availability arrays | `causalgate/data/loaders/*.py` | all ten written against the real file layouts and registered as `cg_<name>`; **not yet run on real files**: `data_raw/` holds other tracks' downloads (e.g. `lamorgia_pd/`, `lobster_mirror_localtest/`, `binance_trades/`, `lobench_ashare/`) but nothing in the per-dataset layouts these loaders read, so no real-data panel has been built; parse / PIT logic covered by 12 format-fixture tests (`tests/causalgate_parts/test_loaders_{public,constructed}.py`). Public set: D1 stocknet (tweets at content row, `avail_text` = next 16:00 ET close), D2 fi2010 (DecPre variant; LOBSTER variant shares the code), D3 fnspid, D5 crypto_pd (stacked ±window per pump; peak derived as argmax high in 60 min because the public table has no peak column), D8 qlib_alpha158 (`.bin` reader, Alpha158-lite computed causally, universe as-of from instrument date ranges). Constructed / licensed: D4 edgar (`avail_label` = AAER release bar, which the public JAR-2020 file does not carry → user-joined column), D6 csi300_limit (Qlib CN or Tushare layout; limit-hit tradability; as-of membership), D7 sp500_tick (minute-bar adaptation of the Johnson et al. 2013 mini-flash-crash rule; LULD halts; constituents as-of), D9 wsb_squeeze (FINRA short interest joined on its publication bar; restricted-broker days; Kaggle licence to check), D10 fin_graph (Elliptic++ needs a user-supplied step calendar or tx timestamps; `label_lag_bars` axis). Text embeddings come from `scripts/cg_prepare_data.py --embed` (PIT-safe `hash_tfidf` default; FinBERT / ChronoBERT / PIT-LM hooks need `transformers`). Graph modalities are built only for D4 (SIC-2) and D10. |
| Protocols P-O / P-W / P-C / P-C-ADIA / one-switch, hashing, freezing | `causalgate/eval/protocols.py` | implemented, tested |
| Metrics: LP, DG, TS-AUC, VUS-PR, PA/no-PA F1, event F1, Kendall τ, alarm-rate error, placebo rank | `causalgate/eval/metrics.py` | implemented, tested |
| Audit: date-shift, placebo, plausibility (PSR/DSR/haircut), card | `causalgate/audit/{date_shift,placebo,plausibility,card}.py` | implemented; card smoke-tested end-to-end |
| Scripts | `scripts/cg_run.py`, `cg_run_all.sh`, `cg_make_tables.py`, `cg_audit.py`, `cg_figures.py`, `cg_planted_leaks.py`, `cg_e7_gate_sweep.py`, `cg_prepare_data.py` | implemented (prepare_data: see §6) |

---

## 1. Smoke results that exist (CPU, this machine; regenerated 2026-10-04 after the code review fixes)

Unit-level checks below are printed by `python3 tests/test_causalgate.py --device cpu` (pass/fail log in `results/causalgate/smoke_full.txt`;
the printed values are reproduced by re-running the test file):

| Check | Result |
|---|---|
| FPIT on the causal reference pipeline | violation rate 0.00, fill-gap 0.00 (pass) |
| FPIT, planted global normalisation (L1) | violation rate 0.75, first operator `normaliser` |
| FPIT, planted test-quantile threshold (L2) | violation rate 1.00, first operator `threshold` |
| FPIT, planted same-bar fill (L7) | decision-time violations 0.00, fill-gap probe 1.00 |
| FPIT, planted label-time join (L5) | violation rate 0.50 with event-adjacent probes, 0.15 with uniform probes; first operator `text_join` |
| Gate under a variance switch (σ 1→3), α = 0.05 | fixed quantile 0.340 overall / 0.538 post-switch; POT 0.068 / 0.080; ACI 0.050 / 0.051; delayed-ACI (τ = 6) 0.050 / 0.051 |
| CausalGate end-to-end on the planted panel (fold 0, TCN scorer, delayed-ACI) | realised alarm rate 0.047 vs α 0.05; prequential replay (refit model, bar-by-bar on the truncated panel) gives DG = 0; 0 guard violations |
| Protocol runner, trivial |z| baseline (fold 0) | causal SR −2.22 / alarm 0.048; walk-forward (L1+L2) SR −2.59; offline_all SR −1.04; causal+L7 (same-bar fill) SR −1.24 → LP^{(L7)} = +0.98 |
| FPIT through the fitted-baseline adapter | causal: pass; causal+L1: violation 0.50, localised to `normaliser`; causal+L2: violation 1.00, localised to `threshold` |
| B2–B7 and B1/B8–B10 wrappers, both modes (fold 0, smoke sizes) | all emit the shared out dict, realised alarm rate within 0.03 of α in causal mode, 0 guard violations in causal mode; STAGE Eq. 20 thresholds differ by reading (full_sample 2.47, test_fold 2.08, train_fold 2.61) |

Full file: `results/causalgate/smoke_full.txt` — 73 tests passed in 172 s (after the review fixes).

P-C/ADIA detection-only replay (truth = an event onset has occurred by step s), fold 0, from `summary.json` under
`results/causalgate/cg_planted/<baseline>/<protocol>/seed42/`: CausalGate 0.712, DAMP 0.745 (causal) / 0.548 (published),
CATCH 0.796 / 0.510, STAGE 0.649 / 0.715. Planted panel, not a benchmark result.

### Planted mini-grid through the CLI (seed 42, fold 0, smoke sizes) — `results/causalgate/tables/tables.md`
`scripts/cg_run.py` × {causalgate, cg_damp, cg_catch, cg_stage} × {published, causal}, then `scripts/cg_make_tables.py --bps 5`.
A plumbing check of the full path, **not a finding** (one seed, one fold, smoke-sized models):

| baseline | protocol | SR | AUC-PR | F1 (headline) | TS-AUC | VUS-PR | alarm rate |
|---|---|---|---|---|---|---|---|
| causalgate | published | −3.44 | 0.171 | 0.395 | 0.746 | 0.193 | 0.050 |
| causalgate | causal | −3.44 | 0.171 | 0.179 | 0.746 | 0.193 | 0.050 |
| cg_damp | published | −2.82 | 0.053 | 0.404 | 0.547 | 0.110 | 0.037 |
| cg_damp | causal | −3.60 | 0.078 | 0.129 | 0.601 | 0.155 | 0.050 |
| cg_catch | published | −3.56 | 0.114 | 0.594 | 0.712 | 0.137 | 0.043 |
| cg_catch | causal | −3.04 | 0.185 | 0.197 | 0.715 | 0.211 | 0.049 |
| cg_stage | published | −3.11 | 0.041 | 0.067 | 0.550 | 0.070 | 0.056 |
| cg_stage | causal | −2.61 | 0.042 | 0.052 | 0.527 | 0.079 | 0.047 |

The headline F1 is now the F1 of each pipeline's own alarms, with point-adjust (per asset, along time) under the published protocol.
The large published-minus-causal F1 gaps (DAMP +0.28, CATCH +0.40, even CausalGate +0.22 with identical alarms) come from point-adjust
alone, which is the L4 inflation the literature reports; the Sharpe and AUC-PR gaps are mixed in sign on this synthetic panel.

`results/causalgate/cards/causalgate__cg_planted.{json,md}` (from `scripts/cg_audit.py`, 6 probes × 2 families, 5 placebos):
verdict CAUSAL; FPIT violation rate 0.00, L7 fill-gap probe run with no violation; placebo mean Sharpe −0.26 (real −0.90, the momentum
head is not tuned); realised alarm rate 0.050 vs α 0.05; no pre-trained component.

### E4 (planted testbed, 4 seeds × 3 probe budgets) — `results/causalgate/planted/fpit_power_table.csv`

FPIT flag rate = power for planted leaks, false-positive rate for the causal pipeline ("none"):

| planted leak | probing | n = 4 | n = 8 | n = 16 |
|---|---|---|---|---|
| none (causal) | uniform / stratified | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| global normalisation (L1) | uniform / stratified | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| test-quantile threshold (L2) | uniform / stratified | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| same-bar fill (L7, fill-gap probe) | uniform / stratified | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| label-time join (L5) | uniform / stratified | 0.75 / 1.00 | 0.75 / 1.00 | 1.00 / 1.00 |

The flag rates are unchanged after the review fixes (FPIT now also perturbs LOB, document age, universe and tradability; a pipeline
counts as flagged only when `passed is False`). Per-probe violation rates are in `fpit_power.csv`;
Prop. 5's bound 1 − (1 − p)^n explains why the sparse-in-time join leak needs stratified probes or n ≥ 16.
Date-shift collapse and placebo flags are reported in `fpit_power.csv` but are uninformative for the reference pipeline,
whose momentum head has negative Sharpe on the planted panel (the collapse ratio is only defined for SR₀ > 0).

### E7 (gate sweep, 4 seeds) — `results/causalgate/e7/gate_summary.csv`
`scripts/cg_e7_gate_sweep.py`: α = 0.05, calibration on the first 500 bars, then streamed. Stream A = variance-switch null
(|N(0, σ_t)|, σ 1 → 3 → 1, T = 6000); stream B = planted panel scored by the causal reference pipeline, pooled over 8 assets (events included).
Realised long-run alarm rate / |ᾱ − α| / worst 500-bar window after a switch (stream A):

| gate | A: realised | A: abs err | A: post-switch max err | B: realised | B: abs err |
|---|---|---|---|---|---|
| fixed conformal quantile | 0.211 | 0.161 | 0.444 | 0.072 | 0.026 |
| POT (GPD, refit every 20; label-free calibration) | 0.060 | 0.010 | 0.197 | 0.052 | 0.004 |
| ACI γ = 0.002 | 0.050 | 0.000 | 0.040 | 0.048 | 0.003 |
| ACI γ = 0.01 | 0.050 | 0.000 | 0.017 | 0.049 | 0.001 |
| ACI γ = 0.05 | 0.050 | 0.000 | 0.002 | 0.049 | 0.001 |
| delayed-ACI γ = 0.01, τ = 1 | 0.050 | 0.000 | 0.017 | 0.049 | 0.001 |
| delayed-ACI γ = 0.01, τ = 6 | 0.050 | 0.000 | 0.018 | 0.049 | 0.001 |
| delayed-ACI γ = 0.01, τ = 21 | 0.050 | 0.000 | 0.019 | 0.049 | 0.001 |
| delayed-ACI γ = 0.002, τ = 21 | 0.050 | 0.000 | 0.044 | 0.050 | 0.004 |
| delayed-ACI γ = 0.05, τ = 21 | 0.050 | 0.000 | 0.004 | 0.051 | 0.002 |

Reading: the long-run rate is controlled by every (delayed-)ACI setting; the transient after a switch is governed by γ (step size), and the
delay τ costs little at these γ (Prop. 4's C(τ, γ, T) term). Stream B worst 500-bar window after a planted regime switch: fixed 0.062,
POT 0.035 (was 0.053 before the gate stopped reading same-bar labels), ACI γ = 0.01 0.015, delayed-ACI γ = 0.01 τ = 6 0.017, τ = 21 0.025, γ = 0.05 τ = 21 0.023. Full τ × γ grid in `gate_sweep.csv`.

---

## 2. E1 — leakage premium (10 datasets × 10 baselines × {P-O, P-W, P-C} × 4 seeds)
Table template (`results/causalgate/tables/E1_SR.csv`, produced by `cg_make_tables.py`): rows = dataset, columns = baseline,
cell = LP_SR = SR_{P-O} − SR_{P-C} (mean ± sd over seeds × folds), plus LP_AUC-PR, LP_F1 (PA vs no-PA), LP_EAR.
Rank agreement: Kendall τ (P-O vs P-C) per dataset (`E1_kendall_tau_SR.csv`). **All cells `[·]` until the grid has run.**

## 3. E2 — channel decomposition (one-switch from P-C)
`E2_SR.csv`: LP^{(l)} = M_{causal+L_l} − M_{causal}, l = L1..L7 (L8 is reported on the audit card as a cutoff check, not as a run), per baseline and dataset family (daily / intraday / text / graph). `[·]`.
Caveats baked into the runner: L5 toggles training-label availability and document visibility (needs loaders that emit
`avail_text` / `avail_label`); L6 needs `universe`; L8 only applies to methods with a pre-trained component (B6 MOMENT, text embedders).

## 4. E3 — honest ranking under P-C, cost grid 1/5/10/15/30 bps
`E3_causal.csv`: SR, AUC-PR, EAR, F1 (no PA), TS-AUC, VUS-PR, alarm rate, trades per (dataset, baseline). `[·]`.

## 5. E5 — pre-trained leakage (L8)
Text datasets D1/D3/D9: FinBERT vs ChronoBERT (arXiv:2502.21206) vs PIT-LM monthly checkpoints (arXiv:2607.11889) as the
embedder, cutoff declared per fold in the embedding cache (`scripts/cg_prepare_data.py --embedder ... --cutoff`). MOMENT (B6)
cutoff check via `cutoff_date` on the audit card. `[·]` (transformers is not installed locally; the default `hash_tfidf`
embedder is PIT-safe by construction and is the control arm).

## 6. E6 — causal ≠ streaming (refit policy)
Ablation tags `abl_refit_streaming`, `abl_refit_periodic` vs the static default under P-C (`ablations.csv`). `[·]`.

## 7. E8 — case studies and competition-leak replication
2020-03, 2021-01 (GME, D9), 2022-05 (LUNA, D5), 2024-08-05: P-O vs P-C equity curves from `trades_fold*.csv` (`cg_figures.py` Fig 1).
Competition replication only on public data (D8) unless Kaggle rules permit; `[·]`.

## 8. Ablations (CausalGate parts)
`configs/causalgate/ablations/`: no_fg (run under causal+L1), gate_fixed / gate_aci / gate_pot, no_warmup_abstain, scorer_ae / scorer_lmp,
refit_streaming / refit_periodic, policy_learned. `[·]`.

---

## 9. Figures (`scripts/cg_figures.py`)
1 motivation equity curves · 2 LP heatmap · 3 channel bars · 4 rank bump chart · 5 FPIT power curves (**exists**: `results/causalgate/figures/fig5_fpit_power.pdf`
after running the script) · 6 alarm-rate tracking · 7 date-shift profiles · 8 audit card example (**exists** for causalgate on cg_planted).

---

## 10. Paper outline (positioning)

**Title.** Peeking Costs Alpha: Strictly Causal Anomaly-to-Trade Pipelines and a Leakage Audit for Financial Anomaly Detection.

1. **Introduction.** Anomaly detectors are used as trading signals; published results rest on offline conventions (full-history
   normalisation, test-set thresholds, point-adjust, post-cutoff language models, today's universe). Competition evidence (DRW, Mitsui,
   Hull Tactical 2025; grey literature, footnoted) and the 2026 ADIA Structural Break Challenge: Real-Time Edition (one observation at a
   time, TS-AUC) make strict causality a benchmark requirement. Contributions: taxonomy + leakage premium; CausalGate; black-box audit
   with power analysis; 10 × 10 × 3 benchmark with PIT vintages and protocol hashes.
2. **Related work.** (a) Evaluation leakage: Kapoor & Narayanan 2023; López de Prado 2018 (purging/embargo); Bailey & López de Prado 2014
   (DSR); Kim et al. AAAI 2022 (point-adjust); Wu & Keogh TKDE 2023; Liu & Paparrizos NeurIPS 2024 (TSB-AD); Sarfraz et al. ICML 2024.
   (b) **Zhang et al. 2026 (arXiv:2605.23959)**: one-switch benchmark for decision-time leakage, 2 daily equity datasets, 6 forecasters,
   no anomaly detection, no audit tooling, no method — we adopt one-switch and extend to the anomaly-to-trade loop with a per-channel
   premium over 10 datasets × 10 methods. (c) **Fonseca 2026 (arXiv:2607.04958)**: look-ahead-freedom as temporal non-interference, a
   static type-and-effect checker; undecidable in general and blind to opaque models — FPIT is the black-box statistical complement, with
   soundness (Prop. 5) and measured power (E4). (d) **Nikolopoulos 2026 (arXiv:2604.15531)**: martingale-difference placebos for
   forecasting — we reuse the placebo as one audit instrument for anomaly-to-trade pipelines. (e) **El Halabi & Brandt 2026
   (arXiv:2609.07251)**: ACI under delayed feedback, synthetic only — we apply it to alarm gating with τ = horizon + latency and a
   cost-derived α, first finance / AD evaluation. (f) **Parrino et al., KDD 2026 (arXiv:2609.39215)**: static TSAD beats streaming updates
   by ~64 % AUC-PR — we separate causality from streaming (E6). (g) LLM look-ahead line: Glasserman & Lin 2023 (arXiv:2309.17322);
   Sarkar & Vafa 2024 (SSRN); He et al. 2025 ChronoBERT (arXiv:2502.21206); Kelly et al. 2026 PIT-LMs (arXiv:2607.11889); Gao et al. 2025
   Lookahead Propensity (arXiv:2512.23847) — none touches anomaly detection or the trade loop; they are the L8 switch (E5).
   Conformal background: Gibbs & Candès 2021; Angelopoulos et al. 2023.
3. **Problem.** Filtration, availability map, operator composition, the eight channels, three protocols, LP / DG / honest ranking (§2 of the plan).
4. **CausalGate.** Filtration Guard (as-of stamps, PITView), streaming normalisation with warm-up abstention, causal scorer with an explicit
   refit-policy axis, delayed-ACI gate (Prop. 4), policy head via the shared engine; Prop. 1 (DG = 0 by construction, verified by replay).
5. **Leakage Audit Card.** FPIT (families, fill-gap probe, localisation, stratified probing; Prop. 5), date-shift probe, placebo, DSR/haircut
   plausibility, pre-trained cutoff check.
6. **Benchmark.** Datasets D1–D10 with PIT fields and licences; baselines B1–B10 as-published and causalised with convention sheets
   (official code for B2–B7 where importable, re-implementations for B1, B8, B9, B10 with every unspecified choice listed). STAGE is reported
   under three readings as the canonical protocol-ambiguous method; Kończal & Połoczański (verified 2026-10-03: GPD fitted on the first 80 %
   training sample, F1-optimal thresholds chosen with test labels for precision/recall) and Al Ridhawi et al. (verified: expanding-window
   train-fitted z-scores, 95th-percentile training reconstruction-error threshold, SAC reward from the previous day's realised close) are
   mostly causal by design; their open channels are L3 (B10), L5/L6 and online adaptation on test (B9).
7. **Experiments E1–E8** (tables above) and ablations.
8. **Limitations.** Re-implementation fidelity; licensed data substitutes (LOBSTER → FI-2010, TAQ → Polygon/IEX); Kaggle competition files
   not used; FPIT cost for deep models (refit per probe) and its power on sparse leaks; grey-literature competition evidence.
9. **Reproducibility.** Frozen protocol hashes, seeds 42/2024/2025/2026, PIT vintages, embedding cutoffs, audit cards for every (method, dataset).

Reference ledger: ids resolved on 2026-10-03 — see plan §12; additionally today: arXiv:2603.19136 (v2, 2 Apr 2026) and arXiv:2608.09576
(10 Aug 2026) fetched and their methods extracted for the convention sheets.
