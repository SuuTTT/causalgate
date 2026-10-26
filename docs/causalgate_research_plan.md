# CausalGate — research plan and paper blueprint
## Strictly causal anomaly-to-trade pipelines and a leakage audit for financial anomaly detection

Working codename **CausalGate** (rename freely). Sibling of Warn2Trade / SparseWarn2Trade / StressGuard in this checkout;
lives in its own package `causalgate/`, configs `configs/causalgate/`, scripts `scripts/cg_*.py`, results `results/causalgate/`.
Written 2026-10-03. Target venues: KDD 2027 research track, NeurIPS 2027 (main or D&B), ICLR 2027, AAAI 2027.
Experiment node: `c225:/date/zyy/TradingSignal` (rsync from this checkout, see §9).

---

## 0. Honesty audit of the brief (read first)

| Claim in the brief | Status | What we can say in the paper |
|---|---|---|
| "STAGE is a likely case" of full-history thresholds | **Under-specified, not proven.** The PLOS ONE paper (Shi, Rasli, Wang 2025, doi:10.1371/journal.pone.0318939) sets the anomaly threshold from the mean and std of the anomaly scores (their Eq. 20) but never states which sample the statistics come from, nor whether the split is chronological, nor how inputs are normalised. It reports Sharpe 1.78 / MDD −16.5 %. | Treat STAGE as the canonical *protocol-ambiguous* method: re-implement it under both readings and report the gap. Never write "STAGE leaks" before our re-implementation shows a positive leakage premium. |
| "DRW, Mitsui and Hull leaderboards were all affected" | **Grey literature only.** DRW: host invalidated all submissions after probable future-peeking ~3 weeks in and re-released data; ~0.95 public scores were later removed (fast.ai forum thread; Substack review of 30+ quant competitions). Mitsui: public LB was probe-able from historical data, "effectively meaningless" (community posts). Hull: public LB rows overlapped the training set and labels were public (participant write-ups). | Cite as footnoted URLs in the motivation, not as papers. Re-verify each on the Kaggle discussion pages before submission. Kaggle competition data is usually restricted to competition use: **do not publish results on these files without checking the rules page** (needs login); replicate the same leak patterns on our public datasets instead (E8). |
| "2026 ADIA real-time challenge" | **Verified.** ADIA Lab Structural Break Challenge: Real-Time Edition (CrunchDAO), 6 May – 1 Oct 2026, 10k train / 20k test univariate series, history segment 1 000–5 000 break-free points, online segment 10–1 000 points fed one at a time, output a cumulative break probability after every point, metric Time-Stratified AUC (per-step cross-sectional AUC weighted by n_pos·n_neg). | Use its protocol (one observation at a time, no revisiting, TS-AUC) as the template for our detection-only causal protocol and as an external sanity benchmark (A1 in §5). Data licence to check. |
| Ten datasets, ten baselines (≥ 2 from 2026) | Feasible; all ten baselines below resolve to real papers. | See §5–§6. |
| Repo state | The ten dataset builders in `warn2trade/data/datasets/` are **documented stubs** (`raise NotImplementedError`); only `synthetic` runs. | Real loaders are the first implementation milestone (§9). |

Adjacent 2026 work that **must** be cited and differentiated (all ids resolved on 2026-10-03):

| Work | What it does | What it does not do (our room) |
|---|---|---|
| Zhang, Li, Peng, Chen, *When Alpha Disappears: A One-Switch Benchmark for Decision-Time Leakage in Financial Backtests*, arXiv:2605.23959 (May 2026) | Toggles one evaluation convention at a time (centred features, execution timing, global normalisation) for 6 model families on 2 daily OHLCV equity datasets 2016–2024; finds leakage effects "highly selective". | No anomaly detection, no online/causal detectors, 2 datasets, no audit tooling, no method. We adopt the one-switch idea and extend it to the anomaly-to-trade loop, 10 datasets × 10 methods, and a per-channel leakage premium. |
| Fonseca, *Look-Ahead-Freedom as Temporal Non-Interference*, arXiv:2607.04958 (Jul 2026) | Formal property + linear-time type-and-effect checker for value-independent pipelines (windowing, joins, PIT reads, agentic retrieval); notes the general problem is undecidable and that "differential and tiling detectors" miss some planted leaks. | Static, needs the pipeline in its calculus; cannot check an opaque PyTorch/LLM model. Our Future-Perturbation Invariance Test is the black-box, statistical complement; we cite his negative result and quantify FPIT power on planted leaks. |
| Nikolopoulos, *Spurious Predictability in Financial Machine Learning*, arXiv:2604.15531 (Apr 2026) | Falsification audit: run the unchanged workflow on martingale-difference nulls and microstructure placebos; quantifies in-sample vs walk-forward inflation. | Forecasting only, no anomaly detection, no multi-dataset benchmark. We reuse the martingale-difference placebo as one audit instrument (placebo Sharpe > 0 ⇒ leak). |
| El Halabi & Brandt, *Adaptive Conformal Inference Under Delayed Feedback*, arXiv:2609.07251 (Sep 2026) | ACI with feedback delay τ; finite-sample coverage bounds; delay-to-memory ratio r = τ/L. Synthetic only. | We apply delayed ACI to anomaly alarms with τ = horizon + latency and tie α to a cost/turnover budget; first finance / AD evaluation. |
| Parrino, Ajenjo, Remy, Stephan, Senellart, Boniol, *In a Streaming World, Should You Stand Still?*, KDD 2026, arXiv:2609.39215 | 10 streaming vs 19 static TSAD methods on TSB-AD-M + new TSB-drift; static methods trained once beat streaming updates by ~64 % AUC-PR on average. | Not finance, no trading loop, no leakage. Key conceptual import: **causality ≠ streaming updates**. A static model applied causally is leakage-free; we separate the two axes (E6). |
| LLM look-ahead line: Glasserman & Lin 2023 (arXiv:2309.17322); Sarkar & Vafa 2024 (SSRN); He, Lv, Manela, Wu, *Chronologically Consistent LLMs*, arXiv:2502.21206; Kelly, Malamud, Schwab, Xu, *Scaling Point-in-Time Language Models*, arXiv:2607.11889 (monthly checkpoints 2013–2024, up to 4B params, released); Gao, Jiang, Yan, *Detecting Lookahead Bias in LLM Forecasts*, arXiv:2512.23847 (Lookahead Propensity); Jia, *HindsightBench*, arXiv:2607.18867; Zhang, *Temporal Leakage in LLM Backtesting*, arXiv:2608.02985; arXiv:2602.17234; TEMPO arXiv:2605.18843 | Text-side leakage from pretraining cutoffs; PIT models as the fix. | None touches anomaly detection or the trade loop. We use PIT-LM / ChronoBERT vs FinBERT as the L8 switch on the text datasets (E5). |

Classic evaluation-leakage references to anchor the taxonomy: Kapoor & Narayanan, *Patterns* 2023 (leakage taxonomy); López de Prado 2018 (purging, embargo, CPCV); Bailey & López de Prado 2014 (DSR); Bailey, Borwein, López de Prado, Zhu 2017 (PBO); Harvey & Liu 2015 (haircut Sharpe); Kim et al. AAAI 2022 (point-adjust inflation); Wu & Keogh TKDE 2023 (flawed TSAD benchmarks); Liu & Paparrizos NeurIPS 2024 D&B (TSB-AD); Sarfraz et al. ICML 2024 (position: quo vadis TSAD); Gibbs & Candès NeurIPS 2021 (ACI); Angelopoulos, Candès, Tibshirani NeurIPS 2023 (conformal PID).

---

## 1. Title candidates and structured abstract

1. **Peeking Costs Alpha: Strictly Causal Anomaly-to-Trade Pipelines and a Leakage Audit for Financial Anomaly Detection**
2. **CausalGate: Filtration-Consistent Anomaly Detection for Executable Trading Signals**
3. **How Much of the Alpha Was Look-Ahead? Measuring and Removing Leakage in Anomaly-Driven Trading**

**Background.** Anomaly detectors are increasingly used as trading signals, yet most reported results are produced offline:
inputs are standardised with full-sample statistics, alarm thresholds are read off the test-set score distribution, labels are
point-adjusted, text is scored by language models trained after the test period, and universes are defined as of today.
Public competitions (DRW, Mitsui, Hull Tactical, 2025) show that even curated leaderboards leak, and the 2026 ADIA real-time
challenge has made strictly online detection a benchmark requirement.

**Gap.** Nobody has measured how much of the reported anomaly-to-trade performance survives when every quantity used at decision
time t is restricted to the information filtration F_t, nor provided tooling that certifies a learned pipeline is causal.

**Method.** We (i) formalise anomaly-to-trade pipelines as operator compositions over a time-indexed availability map and define
the **leakage premium** LP = performance(offline convention) − performance(causal protocol), decomposed over eight leakage
channels; (ii) build **CausalGate**, a strictly causal detector-to-trade stack: streaming robust normalisation, a causal scorer
trained on purged folds, and a **delayed-feedback adaptive-conformal alarm gate** whose long-run alarm rate is controlled at a
cost-derived budget α under arbitrary distribution shift; (iii) release a **Leakage Audit Card** combining a black-box
future-perturbation invariance test, date-shift probes, martingale-difference placebos and multiple-testing-aware Sharpe
plausibility.

**Results (placeholders until runs exist).** Across 10 datasets × 10 baselines (2021–2026, two from 2026) × 3 protocols × 4 seeds,
[x] % of published-convention Sharpe is leakage premium; method rankings under the causal protocol have Kendall τ = [·] with the
offline rankings; the audit flags [·] % of leaky configurations with [·] % false positives on planted leaks; CausalGate is
[competitive / best] under the causal protocol at 1–30 bps while holding the realised alarm rate within [·] of α.

**Contributions.** (1) A leakage taxonomy and a measurable leakage premium for anomaly-to-trade systems. (2) The first
strictly causal, alarm-rate-controlled anomaly-to-trade method with delay-aware conformal guarantees. (3) A black-box leakage
audit with power analysis on planted leaks. (4) A 10 × 10 benchmark with every baseline run as-published and causalised,
released with point-in-time vintages and protocol hashes.

---

## 2. Problem formulation

### 2.1 Data, filtration, availability
Panel as in `warn2trade/data/schema.py`: times t = 1..T, assets i = 1..N, fields x ∈ {mid, price_feat, liq, tradable, lob, text,
graph, labels, bars_to_peak}. Every datum carries an **availability time** a(x_{t,i}) ≥ t (when it became knowable): a bar's close
is available at t; a tweet at its publish time; an AAER label at its release date (years later); an index constituent list at its
announcement; a restated fundamental at the restatement date. The decision filtration is
F_t = σ({x : a(x) ≤ t}). A pipeline is a composition
score s_t = (S ∘ N)(x_{≤·}), alarm z_t = 1[s_t > q_t], trade w_t = π(s_t, z_t, ·),
with normaliser N, scorer S, threshold q, policy π. The pipeline is **causal** iff every one of N, S, q, π (including their fitted
parameters and any model-selection decision) is F_t-measurable at the time it is used.

### 2.2 Leakage channels (the taxonomy the whole paper hangs on)
| Code | Channel | Typical offline convention | Causal replacement |
|---|---|---|---|
| L1 | Normalisation | global z-score / min-max / PCA / scaler fit on the whole series | streaming mean-var (Welford / EWMA) or median-MAD; fit on train fold only |
| L2 | Threshold | test-set quantile at the true anomaly ratio; POT fitted on test scores; "best F1" threshold | threshold calibrated on train/val scores; online conformal quantile with delayed feedback |
| L3 | Model selection | hyper-parameters / epochs / checkpoint chosen on test; many trials, best reported | validation fold only; trial count declared; DSR |
| L4 | Label construction & scoring | point-adjust (PA); labels extended backwards then leaking into features | event-wise metrics without PA; labels are targets only (as in `labels.py`) |
| L5 | Availability lag (point-in-time) | news/tweets joined on calendar date; filings on period-end; labels on event date | join on a(x); vintage tables; label release dates |
| L6 | Universe / survivorship | today's constituents; delisted names dropped | as-of constituents; keep delisted paths until exit |
| L7 | Execution timing | decide at close t, fill at close t; signal from bar t traded at bar t open | decision at t, fill at t + latency (latency ≥ 1), tradability mask |
| L8 | Pre-trained components | LLM / TS-foundation model trained past the test window | PIT-LM / ChronoBERT checkpoints with cutoff < test start; declare cutoffs |

### 2.3 Protocols
- **P-O (as published):** each baseline with the conventions in its paper/official code (documented per baseline in §6).
- **P-W (purged walk-forward, offline inside the fold):** `WalkForwardSplitter` with purge ≥ max(h) + latency + schedule and
  embargo, but normalisation and thresholds fitted on the *test fold* as most TSAD code does. The common "we did walk-forward" compromise.
- **P-C (strictly causal):** all operators F_t-measurable, enforced by the Filtration Guard (§3.1), fills at t + latency, no PA.
- Detection-only causal protocol **P-C/ADIA**: one observation at a time, cumulative score, TS-AUC.

### 2.4 Quantities
- **Leakage premium** LP_M(m, D) = M_{P-O}(m, D) − M_{P-C}(m, D) for metric M ∈ {Sharpe, AUC-PR, EAR, F1}; channel-wise
  LP_M^{(ℓ)} by toggling only channel ℓ from P-C (one-switch design, after Zhang et al. 2026).
- **Deployability gap** DG = M_{backtest} − M_{prequential replay} (replay = the frozen pipeline run bar by bar through the
  same test window). Zero in expectation for a causal pipeline (Prop. 1).
- **Leakage-adjusted Sharpe** = Sharpe under P-C; **honest ranking** = method ranking under P-C.

---

## 3. Method: CausalGate

### 3.1 Filtration Guard (FG) — causality by construction
- `PITView(panel, t)`: every Panel field is paired with an availability array (`avail_time`, (T,N) or (T,N,M) for text) or a
  scalar lag; a view at t exposes only entries with avail ≤ t. Builders must emit availability arrays (new optional Panel fields,
  **appended** to the schema, never breaking existing tracks).
- Every fitted operator (normaliser, scaler, threshold, PCA, embedding index, model checkpoint) implements
  `fit_causal(stream_until=t)` and is stamped with its as-of time; using an operator stamped after t raises.
- **Future-Perturbation Invariance Test (FPIT):** for random probe times t and perturbation families
  𝒫 ∈ {block-shuffle x_{>t}, Gaussian noise on x_{>t}, splice a different fold after t, delete x_{>t}}, recompute s_t, q_t, w_t.
  Any change ⇒ violation. Report violation rate, mean |Δs_t|, and the first operator whose output changed (localisation by
  operator-wise diff). FPIT is the black-box complement to Fonseca's static checker; we report its power on planted leaks (E4).

### 3.2 Causal scorer and streaming normalisation
- Streaming robust normaliser: EWMA mean/var (Welford) and P²-quantile median/MAD; warm-up bars are **abstained**, not filled.
- Scorer options (ablation): (a) causal TCN / left-only attention one-step predictor, score = standardised prediction error;
  (b) prequential autoencoder (reconstruction of the current window only, no future context); (c) left matrix profile
  (DAMP-style) as a non-parametric reference. Trained on the purged train fold; **refit policy** ∈ {static, per-fold, periodic,
  streaming} is an explicit axis (E6), because Parrino et al. show static often wins — causality does not require streaming updates.
- Multimodal inputs (text, graph) only through PIT embeddings computed with a model whose cutoff < fold test start (L8).

### 3.3 Delayed-feedback adaptive-conformal alarm gate
- Alarm z_t = 1[s_t > Q̂_{1−α_t}(calibration window of past scores)].
- The alarm budget α is derived from the cost model: with per-trade cost c and acceptable annual cost burn B, α = B / (c · bars/yr).
- α_t is updated by ACI with delay τ = h + latency: when the outcome of alarm t − τ becomes known (event realised or not; trade
  PnL sign), α_{t+1} = α_t + γ(α − err_{t−τ}). Guarantee (Prop. 4, from El Halabi & Brandt 2026): the long-run realised alarm
  rate converges to α with a finite-sample deviation explicit in τ and γ, under arbitrary non-stationarity.
- The alarm feeds the Warn2Trade policy head (size w, hold distribution, schedule) through the shared event backtester; trades
  only on alarms, abstains otherwise. Cost-aware policy training reuses `backtest/engine.py` unchanged.

### 3.4 Leakage Audit Card (LAC) — one per (method, dataset)
1. LP per channel (L1–L8 one-switch) and total LP.
2. FPIT violation rate and localisation.
3. Date-shift probe: shift the signal by k ∈ {−3..+3} bars before execution. Honest edge decays smoothly with delay and
   *improves* when allowed to peek; a leak collapses as soon as the signal leaves the executed bar.
4. Martingale-difference placebo: run the unchanged pipeline on surrogates with no predictability (block-bootstrapped or
   sign-randomised returns that preserve volatility clustering). A significantly positive placebo Sharpe ⇒ leak (Nikolopoulos 2026).
5. Sharpe plausibility: PSR, DSR with the declared trial count; flag if DSR p < 0.05 **and** SR exceeds the 99th percentile of
   the placebo distribution — the "implausibly high Sharpe" alarm from the brief, made quantitative.
6. Pre-trained component check: cutoff date of every LLM/TSFM vs fold test start; Lookahead-Propensity-style date probe for LLM
   parts (Gao et al. 2025; Jia 2026).

---

## 4. Theory obligations (appendix)
- **Prop. 1 (replay unbiasedness).** Under FG with fills at t + latency, the backtest PnL process equals the prequential-replay
  PnL process pathwise; DG = 0. Proof by construction (every operator uses the same F_t-measurable inputs in both runs).
- **Prop. 2 (threshold leakage premium).** For a test-quantile threshold at the true anomaly ratio ρ versus a threshold
  calibrated on a shifted distribution, the F1 inflation is bounded below by a function of ρ and the Wasserstein (or quantile)
  shift δ between calibration and test score distributions; LP^{(L2)} ≥ 0 with equality iff δ = 0.
- **Prop. 3 (normalisation leakage as information).** For a two-regime variance process, the full-sample z-score at t has
  strictly positive mutual information with the future regime indicator; hence LP^{(L1)} > 0 for volatility-driven anomalies.
- **Prop. 4 (alarm-rate control under delay).** Specialise El Halabi & Brandt's delayed-ACI bound to the alarm gate:
  |T^{-1} Σ_t z_t-error − α| ≤ C(τ, γ, T).
- **Prop. 5 (FPIT soundness and power).** Zero violations for any causal pipeline; for a pipeline with non-zero leakage
  sensitivity on a perturbation family, the detection probability after n probes is ≥ 1 − (1 − p)^n.

---

## 5. Datasets (ten slots, each tagged with the leakage channels it exercises)
Builders and verification status are in `docs/benchmark_audit.md`; the ten builders are stubs today. PIT availability arrays
are the new requirement for every builder.

| # | Dataset (builder) | Status | Freq | Channels exercised | PIT requirement |
|---|---|---|---|---|---|
| D1 | StockNet (`stocknet`) | PUBLIC | daily | L5 (tweet UTC time vs ET close), L1, L2, L8 | tweet timestamps → avail; PIT-LM vs FinBERT embeddings |
| D2 | FI-2010 LOB (`lobster`, variant fi2010; LOBSTER licensed) | PUBLIC | event / 100 ms | L1 (dataset ships pre-normalised; z-score uses prior-day stats — a built-in PIT example), L7 | raw vs normalised versions; same-event fills forbidden |
| D3 | FNSPID (`fnspid`) | PUBLIC | daily | L5 (article time after close ⇒ next day), L8 | article timestamps; PIT-LM cutoff per fold |
| D4 | EDGAR 10-K/10-Q + AAER (`edgar`) | PUBLIC | filing-day | L5 (AAER release lag of years), L4 | acceptance datetime; AAER release date as label avail |
| D5 | Crypto pump-and-dump, La Morgia et al. 2023 (`crypto_pd`) | PUBLIC | 1 min | L2/L7 (pump timestamps are peaks; detection must use pre-peak only), L5 | Telegram message time; peak time as label avail |
| D6 | CSI 300 limit-hit (`csi300_limit`) | CONSTRUCTED | daily | L6 (as-of constituents), L7 (limit tradability), L1 | constituent change announcements |
| D7 | S&P 500 minute bars (`sp500_tick`; Polygon/IEX, TAQ licensed) | CONSTRUCTED | 1 min | L7 (LULD halts), L6, L4 (mini-flash-crash rule labels are future-defined) | halt calendar; constituents |
| D8 | Qlib Alpha158 US + CN (`qlib_alpha158`) | PUBLIC | daily | L1 (Qlib processors' fit range), L3 | processor fit window = train only |
| D9 | WSB / StockTwits squeeze (`wsb_squeeze`) | PUBLIC/API | daily | L5 (post time; Pushshift retro-collection), L6 (broker restrictions) | post timestamps; restriction dates |
| D10 | Elliptic++ + BTC (`fin_graph`) | PUBLIC | 1 h | L4/L5 (illicit labels assigned retrospectively) | label avail = block time + declared lag (sensitivity axis) |

Protocol anchors (not counted in the ten):
- **A1 ADIA Structural Break Real-Time** (CrunchDAO 2026): detection-only, TS-AUC, one point at a time — external sanity check
  for the causal scorer + gate. Licence to check before use in the paper.
- **A2 Synthetic planted-leak testbed** (`causalgate/data/planted.py`): regime-switching returns with injected events and
  switchable leaks (global norm, test-quantile threshold, same-bar fill, label-time join). Ground truth for E4.
- **A3 Competition replication** (Hull / Mitsui / DRW public train files) only if the Kaggle rules permit publication; otherwise
  replicate the documented leak patterns on D8 and state so.

---

## 6. Baselines (ten, 2021–2026, two from 2026) — every one run as published (P-O) and causalised (P-C)
| # | Method | Venue / id | Family | Offline conventions in the original (to document, then switch) |
|---|---|---|---|---|
| B1 | CPD-DMN, Wood, Roberts, Zohren | JFDS 2022, arXiv:2105.13727 | anomaly → trade | GP change-point score is online; volatility scaling and feature standardisation to check |
| B2 | Anomaly Transformer, Xu et al. | ICLR 2022 | deep TSAD | L2 test-ratio threshold, L4 point-adjust, L1 |
| B3 | DAMP, Lu et al. | KDD 2022 | streaming (left matrix profile) | strictly causal reference; only L7 to control |
| B4 | MemStream, Bhatia et al. | WWW 2022 | streaming memory AD | causal reference (in Parrino et al.'s streaming set); initial-memory fit on train only |
| B5 | DCdetector, Yang et al. | KDD 2023 | contrastive TSAD | L2, L4, L1 |
| B6 | MOMENT zero-shot AD, Goswami et al. | ICML 2024 | TS foundation model | L8 (Time Series Pile contents vs test windows), L2 |
| B7 | CATCH, Wu et al. | ICLR 2025 | frequency-patching MTSAD | L2, L4, L1 |
| B8 | STAGE, Shi, Rasli, Wang | PLOS ONE 2025 | GAT + VAE anomaly → trend → trade | protocol unspecified (threshold from score mean/std); run both readings |
| B9 | Al Ridhawi, Haj Ali, Al Osman | arXiv:2603.19136 (2026) | AE-gated dual transformers + SAC-tuned threshold | threshold tuned by RL on prediction performance — check which window; normalisation unspecified |
| B10 | Kończal & Połoczański | arXiv:2608.09576 (2026) | GPD/EVT threshold + GBM one-bar-ahead | GPD fitted on which sample; out-of-sample claimed, split unspecified |

Reference rows (reported, not counted): Warn2Trade (previous track), `detect_then_rule`, `zscore_*` rules, Isolation Forest,
TranAD (VLDB 2022), Carlei et al. 2026 (Computational Economics, IF + CatBoost), Letteri 2025 (arXiv:2507.14960),
FinBERT-sentiment vs ChronoBERT / PIT-LM (the L8 switch on D1/D3/D9).
Every baseline subclasses `warn2trade/baselines/event_base.BaseBaseline` (tolerant registration) and takes `mode ∈ {published, causal}`;
its convention sheet (which of L1–L8 the official code uses, with file/line pointers) goes in `causalgate/baselines/CONVENTIONS.md`.

---

## 7. Metrics
- Detection: AUC-PR and VUS-PR (no point-adjust), event-wise F1, TS-AUC (ADIA style), WLT (repo).
- Trading: SR, PSR, DSR with declared trials, MDD, Calmar, turnover, EAR, FDP (all in `backtest/metrics.py`), at 1/5/10/15/30 bps.
- New (in `causalgate/eval/metrics.py`): LP per channel and total; deployability gap; FPIT violation rate; date-shift asymmetry
  index; realised-alarm-rate error |ᾱ − α|; placebo Sharpe and placebo rank; Kendall τ between P-O and P-C rankings.

---

## 8. Experiment matrix
| Exp | Question | Design |
|---|---|---|
| E1 | How much reported performance is leakage? | 10 datasets × 10 baselines × {P-O, P-W, P-C} × seeds {42, 2024, 2025, 2026}; LP heatmaps; rank correlation P-O vs P-C |
| E2 | Which channel leaks most, where? | One-switch toggles L1–L8 from P-C; channel-wise LP per dataset family (daily / intraday / text / graph) |
| E3 | Is a causal method competitive? | CausalGate vs all baselines under P-C, cost grid 1–30 bps, dual metric view |
| E4 | Does the audit work? | Planted leaks on A2 and on real data (inject each channel): FPIT power/FPR, date-shift, placebo Sharpe, DSR flags; ROC of the audit |
| E5 | Pre-trained leakage | FinBERT / FinGPT vs ChronoBERT vs PIT-LM monthly checkpoints on D1/D3/D9; MOMENT with cutoff check; LAP-style date probes |
| E6 | Causal ≠ streaming | refit policy {static, per-fold, periodic, streaming} under P-C; replicate Parrino et al.'s finding in finance |
| E7 | Alarm-rate control | realised alarm rate vs α across regimes; τ sweep; fixed quantile vs POT vs ACI vs delayed-ACI |
| E8 | Case studies | 2020-03 crash, 2021-01 GME, 2022-05 LUNA, 2024-08-05 vol spike: P-O vs P-C equity curves; competition-leak replication (A3 or D8) |
| Abl. | CausalGate parts | no FG (allow global norm), gate ∈ {fixed, ACI, delayed-ACI}, no warm-up abstention, scorer (a)/(b)/(c), refit frequency |

Figures: (1) motivation — same detector, offline vs causal equity curve on D8; (2) LP heatmap 10 × 10; (3) channel decomposition
bars; (4) rank-shift bump chart P-O → P-C; (5) FPIT power curves; (6) alarm-rate tracking under regime shifts; (7) date-shift
probe profiles (honest vs leaky); (8) audit card example.

---

## 9. Implementation blueprint (`c225:/date/zyy/TradingSignal`)
```
causalgate/
  data/pit.py          PITView, availability arrays, vintage tables (append optional fields to Panel; never rewrite schema.py)
  data/stream.py       Welford / EWMA / P²-quantile normalisers with as-of stamps
  data/planted.py      synthetic planted-leak generator (A2)
  data/loaders/        real loaders for D1–D10 (fill the stubs' recipes; emit avail arrays); start with D1, D3, D2, D8, D5 (all public)
  models/scorer.py     causal TCN / prequential AE / left-MP scorers
  models/gate.py       ACI and delayed-ACI alarm gate; α from the cost budget
  models/causalgate.py scorer + gate + reused Warn2Trade policy head → shared `out` dict
  baselines/           B1–B10 wrappers with mode={published,causal}; CONVENTIONS.md; register via event_base tolerant decorator
  audit/fpit.py, audit/date_shift.py, audit/placebo.py, audit/plausibility.py, audit/card.py
  eval/protocols.py    P-O / P-W / P-C / P-C-ADIA runners; eval/metrics.py (LP, DG, TS-AUC, VUS-PR, Kendall τ)
  train/trainer.py
configs/causalgate/{default,causalgate}.yaml, baselines/*.yaml, datasets/*.yaml (PIT fields), ablations/*.yaml, protocols/*.yaml
scripts/cg_train.py cg_baseline.py cg_audit.py cg_protocol_sweep.py cg_make_tables.py cg_run_all.sh cg_planted_leaks.py
tests/test_causalgate.py   plain-script mode (system Python 3.9 has no pytest; `timeout` is not in this zsh)
results/causalgate/
```
Rules inherited from the shared checkout (see memory notes): new files only; shared modules (`warn2trade/models/*`,
`backtest/{engine,metrics,costs}.py`, `train/trainer.py`, `data/{base,schema,labels,splits}.py`) are append-only; configs stay
under `configs/causalgate/`; import warn2trade pieces explicitly, never via package `__init__` side effects.

Milestones (8 weeks):
1. **Week 1** — planted-leak testbed + FG + FPIT; smoke: FPIT must flag global z-score, test-quantile threshold and same-bar fill
   on A2 and pass on the causal pipeline. `python3 tests/test_causalgate.py` on CPU.
2. **Week 2** — streaming normalisers, scorer (a)(c), delayed-ACI gate; E7 on synthetic; CPU.
3. **Weeks 3–4** — real loaders D1, D3, D2, D8, D5 with avail arrays; baselines B3, B4, B2, B5, B7 (public code) in both modes.
4. **Week 5** — D4, D6, D7, D9, D10; baselines B1, B6, B8, B9, B10 (re-implementations from the papers; document every guess).
5. **Week 6** — full E1/E2 grid on c225 (`cg_run_all.sh`), 4 seeds; audit cards.
6. **Week 7** — E3–E8, ablations, figures, tables (`cg_make_tables.py`).
7. **Week 8** — paper draft, appendix proofs, reproducibility package, licence checks.

Compute estimate: ~1 200 baseline runs (10 × 10 × 3 × 4) × ~5 folds; deep baselines 5–30 GPU-min per fold → ≈ 300–500 GPU-hours
worst case; streaming/rule baselines negligible. Deterministic flags as in `warn2trade/utils/seed.py`; `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Sync and run:
```bash
rsync -av --exclude runs --exclude data_raw --exclude results . c225:/date/zyy/TradingSignal/
```
```bash
bash scripts/cg_run_all.sh --protocols published,walkforward,causal --seeds 42,2024,2025,2026 --datasets all --baselines all
```

---

## 10. Risks and reviewer objections
- *"Leakage is a known problem."* Known qualitatively; never quantified for anomaly-to-trade systems across 10 datasets × 10
  methods, and no prior work ships a black-box audit with a power analysis or a causal method with alarm-rate guarantees.
  The 2026 adjacent papers (§0) are narrow (2 datasets / 6 forecasters; static checker; synthetic ACI; LLM-only).
- *"Causal methods will simply look worse."* That is the finding, and we add the honest ranking plus a causal method that stays
  competitive; the paper's value is the measurement, not a higher Sharpe.
- *"Re-implemented baselines may be unfaithful."* Convention sheets with file/line pointers; official code where it exists
  (B2–B7); both protocol readings for under-specified papers (B8–B10).
- *"Multiple comparisons."* Trial counts declared; DSR; protocol frozen and hashed (`configs/causalgate/PROTOCOL_HASH`) before
  any test-fold run.
- *Licences:* LOBSTER, TAQ, Kaggle competition data, ADIA data — substitutes stated; nothing unlicensed in the paper.
- *Data builders are stubs:* the first four weeks are data work; budget them.

## 11. Reproducibility checklist
- Frozen protocol file with hash; config and seed manifests per run; PIT vintages archived with the release; cutoff dates of
  every pre-trained checkpoint; audit cards for every (method, dataset) in the supplement; planted-leak generator and FPIT
  released; all arXiv ids re-resolved at camera-ready.

## 12. Reference verification ledger (2026-10-03)
Resolved by fetching the abstract page or found directly in arXiv listings today: 2605.23959, 2607.04958, 2604.15531,
2607.18867, 2512.23847, 2609.07251, 2609.39215, 2609.39232, 2607.11889, 2502.21206, 2603.19136, 2608.09576, 2608.02985,
2602.17234, 2605.18843, 2604.02549, 2608.05373, 2309.17322, 2105.13727, 2507.14960; PLOS ONE 10.1371/journal.pone.0318939;
Springer 10.1007/s10614-025-11274-8. From memory with high confidence but to be re-checked before citing: Anomaly Transformer
ICLR 2022; DAMP KDD 2022; MemStream WWW 2022; DCdetector KDD 2023; MOMENT ICML 2024; CATCH ICLR 2025; TranAD VLDB 2022;
Kim et al. AAAI 2022; Wu & Keogh TKDE 2023; Liu & Paparrizos NeurIPS 2024; Sarfraz et al. ICML 2024; Kapoor & Narayanan Patterns
2023; Gibbs & Candès NeurIPS 2021; Angelopoulos et al. NeurIPS 2023; Bailey & López de Prado 2014; Bailey et al. 2017; Harvey & Liu 2015.
Grey literature (URLs only): DRW/Mitsui/Hull leak reports; ADIA real-time docs page.
