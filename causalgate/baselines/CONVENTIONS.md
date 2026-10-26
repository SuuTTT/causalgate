# Baseline convention sheets (B1–B10): which leakage channels the original uses, and how the causal mode removes them

Channel codes (docs/causalgate_research_plan.md §2.2): L1 normalisation · L2 threshold · L3 model selection · L4 label construction /
point-adjust · L5 availability lag (PIT join) · L6 universe / survivorship · L7 execution timing · L8 pre-trained components.

How to read a sheet: *original* = what the paper or the official code does (pointer = file:line of the official repository or the
equation / section number of the paper, fetched on 2026-10-03); *causal mode* = what `mode='causal'` of our wrapper does instead.
Channels L4, L5, L6, L7 are not properties of the model: they are switched by the protocol runner (`causalgate/eval/protocols.py`)
at the metric, data and pricing level; L8 is a declared `cutoff_date` checked on the audit card.

Status legend: **official** = the official implementation is imported from `third_party/` (see THIRD_PARTY.md); **reimplemented** =
re-implemented from the paper with the algorithm reproduced; **reimplemented-fallback** = compact re-implementation with listed
simplifications (used when the official code is not vendored); **wrapper-needed** = needs an external package / weights.

Rule of the paper: no baseline is called "leaky" from its sheet alone. A channel in the *original* column is an offline convention;
whether it inflates performance is measured by the leakage premium (E1/E2). Under-specified papers (B8 STAGE in particular) are run
under every admissible reading and reported as protocol-ambiguous.

The per-baseline sections below are merged from `conventions_b2_b7.md` and `conventions_b1_b8_b9_b10.md` (kept as the editable sources).


---

# Part A — B1, B8, B9, B10 (re-implemented from the papers)


Re-implementations for the CausalGate leakage audit (`causalgate/baselines/b1_cpd_dmn.py`, `b8_stage.py`,
`b9_alridhawi.py`, `b10_konczal_evt.py`; configs in `configs/causalgate/baselines/{cpd_dmn,stage,alridhawi,konczal_evt}.yaml`;
smoke test `tests/causalgate_parts/test_baselines_reimpl.py`, run with
`python3 tests/test_causalgate.py --device cpu -k baselines_reimpl`). Channel codes L1–L8 follow
`docs/causalgate_research_plan.md` §2.2. Every wrapper subclasses `ScoreToTradeBaseline` (`common.py`) and runs in
`mode ∈ {published, causal}`; the scaffold owns the L1 switch (OfflineScaler vs streaming normaliser), the L2/L3
switch (published threshold vs delayed-ACI gate) and the out dict. L4 (point-adjust), L5 (availability join),
L7 (fill at t vs t + latency) and L8 (pre-trained cutoffs) are properties of the protocol runner, not of the wrapper;
the rows below record what the *original* does so the runner can mirror it.

Shared implementation note. The scaffold hands `_score` only the window tensor X. A small `RowAwareMixin`
(defined in `b1_cpd_dmn.py`, imported by the other three) stashes the rows behind those windows — bar `t`, asset,
label, the raw feature window and the in-fold future returns — at the single point where every path sees the batch
(`_windows`), and stamps/`check`s the fitted model operators so that the Filtration Guard log covers the model and
each published threshold, not only the gate. "Inert" below means the scaffold's switch does not change the baseline's
computation because the baseline consumes raw returns (its own normalisation is what the paper does).

Reference verification (2026-10-03): B1 arXiv:2105.13727 (text extracted from the arXiv PDF), official code
`kieranjwood/slow-momentum-fast-reversion` (`src/changepoint_detection.py`, `src/classical_strategies.py`) and
`kieranjwood/trading-momentum-transformer` (`mom_trans/model_inputs.py`, `mom_trans/deep_momentum_network.py`,
`mom_trans/changepoint_detection.py`); B8 PLOS ONE 10.1371/journal.pone.0318939 (HTML full text); B9
arxiv.org/html/2603.19136v2; B10 arxiv.org/html/2608.09576. Equation numbers are those printed in each source.

---

## B1 — CPD-DMN (Wood, Roberts, Zohren, JFDS 2022, arXiv:2105.13727) — status: `reimplemented`

Pipeline: GP change-point detection on a look-back window (LBW) → severity ν and location γ as extra inputs of a
Deep Momentum Network (LSTM + tanh position head) trained with the Sharpe loss. Score for the gate = ν (online by
construction); direction = DMN position in (−1, 1) (|direction| ≤ 1, the scaffold multiplies by `w_max`).

| Channel | Original (pointer) | published mode | causal mode |
|---|---|---|---|
| L1 normalisation | **Causal.** CPD input standardised within the LBW window (Eq. 2; `changepoint_detection.py`: `StandardScaler().fit(Y_data)` on the window). DMN inputs = returns / 60-day EWM std lagged one day (`classical_strategies.py`: `VOL_LOOKBACK = 60`, `annualised_vol.shift(1)`), MACD normalised by rolling 63 / 252-day std, winsorised at 5 EWM std with 252-day half-life (App. B), then `StandardScaler` **fitted on the training split only** (`mom_trans/model_inputs.py`, `ModelFeatures.set_scalers`). No full-sample statistic. | Same (train-fitted feature scaler stamped at the train end); scaffold switch inert | Same |
| L2 threshold | **None.** A position X_t ∈ (−1, 1) is taken every day (Eq. 13, Sec. IV-A). | threshold = −∞ (`always_in_market`) | delayed-ACI gate on ν (our anomaly-to-trade reading; the P-O/P-C gap for B1 therefore also contains the gating itself) |
| L3 selection | Validation = last 10 % of each training window per asset; 50 random-search trials over Exhibit 7; LBW ∈ {10, 21, 63, 126, 252} picked by validation loss; early-stopping patience 25 (Sec. IV-A, App. B); expanding 5-year windows 1990–2020 with re-tuning each window (Sec. IV-B). No test-set selection. | fixed smoke hyper-parameters, no search | same |
| L4 labels | No labels; no point-adjust. Planted labels are used for AUC-PR reporting only. | — | — |
| L5 availability | Prices only (Pinnacle CLC continuous futures). | — | — |
| L6 universe | Fixed 50 liquid continuous futures 1990–2020 with < 10 % missing data (Sec. IV-B, App. A): mild ex-post selection. | — | — |
| L7 execution | X_t formed at close t earns r_{t+1} = p_{t+1}/p_t − 1 (Eq. 11): **decision and fill at the same close**. Costs not in the loss; App. C Eq. 15 charges C·σ_tgt·\|X_t/σ_t − X_{t−1}/σ_{t−1}\|, C ∈ [0, 5] bps (strategy survives ≤ 2 bps). | runner prices at t | runner prices at t + latency |
| L8 pre-trained | None. | — | — |

Equations implemented: Matérn-3/2 kernel (Eq. 4), nlml (Eq. 7), change-point kernel k_C = k_1 σ(x)σ(x′) + k_2 (1−σ(x))(1−σ(x′)),
σ(x) = 1/(1+e^{−s(x−c)}) (Eq. 9), ν = 1 − 1/(1 + e^{−(nlml_C − nlml_M)}), γ = (c − (t − l))/l (Eq. 10), captured return
R_{t+1} = X_t σ_tgt/σ_t r_{t+1} with σ_tgt = 15 % (Eq. 11), Sharpe loss −√252·E[R]/√Var[R] over all (asset, time) pairs (Eq. 14).

Unspecified / our choices (`CPDDMNBaseline.unspecified`):
1. Kernel hyper-parameters are maximised over a fixed grid with the signal variance profiled out in closed form (one batched quadratic form per window) instead of gpflow L-BFGS-B per window; steepness s is on the grid. Eq. 7 remains the objective; no window subsampling is needed (6 Matérn × 684 change-point grid points in the smoke config).
2. ν, γ are computed once at the window end and broadcast to every LSTM step (the paper precomputes them per day); LSTM sequence length = scaffold look-back (48; paper 63); LBW = 21 (paper tunes it).
3. Return scales > look-back − 1 are dropped (48-bar windows keep scales 1 and 21); the MACD normalisers (rolling 63/252-day std) and the EWM volatility (span 60) are truncated to the window (expanding statistics inside it).
4. Location follows Eq. 10; the official code exports the complement (X_last − c)/(X_last − X_first).
5. Winsorisation at 5 EWM std replaced by clipping the train-standardised features at ±5.
6. Sharpe loss over all (asset, bar) pairs of the minibatch (keras batch loss); the next-bar target at the last window step is the in-fold future return.
7. The σ_tgt/σ_t leverage enters the loss only; the traded weight is the tanh position × `w_max`.
8. No random search / early stopping in the smoke config (2 epochs, hidden 8).

---

## B8 — STAGE (Shi, Rasli, Wang, PLOS ONE 2025, doi:10.1371/journal.pone.0318939) — status: `reimplemented`

The canonical **protocol-ambiguous** baseline. The paper defines a stock graph G = (V, E) with adjacency A, a GAT layer
(Eq. 5 h′_i = σ(Σ_j α_ij W x_j), Eq. 6 α_ij = softmax_j(LeakyReLU(aᵀ[W x_i ‖ W x_j]))), a VAE on the node features
(Eqs. 7–9, loss Eq. 10 = reconstruction + KL), an anomaly score per node (Eq. 19) and the threshold of Eq. 20,
τ = μ_S + δ·σ_S, "the mean and standard deviation of the anomaly scores S". It does **not** state which sample
μ_S, σ_S are computed over, nor the split, nor any normalisation. All three readings are implemented
(`cfg['stage_reading']`) and the gap is reported; nothing in the code or here calls the method leaky.

| Channel | Original (pointer) | published mode | causal mode |
|---|---|---|---|
| L1 normalisation | **Not stated** (no normalisation / standardisation / scaling sentence). | scaffold OfflineScaler, `published_norm: global` (a reading) | streaming z-score (a reading) |
| L2 threshold | Eq. 20, τ = μ_S + δ σ_S; sample not stated; δ "adjustment parameter", value not stated. | reading ∈ {`full_sample`, `test_fold` (default), `train_fold`}, δ = 2 | delayed-ACI gate |
| L3 selection | Split not stated; results over "Window 1/2/3" (Fig. 5, Fig. 8) without dates; 20 epochs stated; other hyper-parameters in "Table 2" (not in the text); no validation procedure. | fold train/val/test | same |
| L4 labels | No point-adjust; prediction accuracy 85 % → 95 % with anomaly detection; Sharpe 1.78, MDD −16.5 % (Table 5). | — | — |
| L5 availability | Combined_News_DJIA.csv (25 daily headlines) described but not used in the reported experiments. | — | — |
| L6 universe | DJIA constituents, vintage not stated. | — | — |
| L7 execution | Not stated; no transaction costs. | runner prices at t | runner prices at t + latency |
| L8 pre-trained | None. | — | — |

The three readings of Eq. 20 on the planted panel (fold 0, δ = 2, same fitted model):
`full_sample` (train ∪ val ∪ test scores) τ = 2.473, `test_fold` τ = 2.078, `train_fold` τ = 2.610 — alarm rates 0.073
(test_fold) vs 0.036 (train_fold) for the same scores; only `train_fold` leaves the Filtration Guard log empty (the
threshold operator is stamped at the last bar of the sample it was computed on and checked at every predict batch).

Unspecified / our choices (`STAGEBaseline.unspecified`):
1. Graph construction: kNN (k = 2) on the rolling correlation of the past 60 bars of returns + self-loops, rebuilt causally at every bar in both modes.
2. Node features: the asset's last `node_bars` (= 48, capped at the run's lookback) bars of scaffold-normalised features, flattened; bars before the panel start are zero-filled (never wrapped to the panel end).
3. GAT: one layer, one head, ELU; the extra regularisation terms printed in Eqs. 5/10/19 are dropped.
4. VAE: hidden 16, latent 4, KL weight 0.1, Adam, 2 epochs (paper: 20).
5. Anomaly score (Eq. 19) read as the node's reconstruction error + kl_weight·KL at the posterior mean.
6. Eq. 20 sample: three readings, default `test_fold` (the literal "scores being thresholded"); δ = 2.
7. Trend module and trading rule not described: trend = sign(MA₅ − MA₂₀) of the window's cumulative return; trade = scaffold alarm → position in the trend direction, hold h bars.
8. Split and normalisation not stated: fold split + scaffold L1 switch.

---

## B9 — Al Ridhawi, Haj Ali, Al Osman (arXiv:2603.19136, 2026) — status: `reimplemented-fallback`

AE-gated dual node transformers with an RL (SAC) controller of the anomaly threshold τ and the blending weight α.
Score for the gate = e_t = ‖x_t − x̂_t‖² (Sec. III-C2); in published mode the score is e_t − (τ_t − τ_0) so that the
scalar threshold τ_0 reproduces the controller's time-varying rule e_t > τ_t; direction = sign of the blended one-step
forecast. Fallback: the SAC controller is replaced by a deterministic bounded gradient-free update (see below).

| Channel | Original (pointer) | published mode | causal mode |
|---|---|---|---|
| L1 normalisation | **Train-fitted, not global**: expanding-window z-scores x̃_{i,t} = (x_{i,t} − μ_{1:t})/σ_{1:t} inside the training period; validation / test use the frozen statistics of the full training period (Sec. III-B, Eq. 3). | `published_norm: train` (OfflineScaler on the train-fold rows, stamped at the train end — no violation) | streaming z-score = the paper's own training-time rule |
| L2 threshold | τ_0 = 95th percentile of the **training-set** reconstruction errors (Sec. III-C); then Δτ ∈ [−0.1, 0.1] from the controller whose reward uses the previous day's realised close (one-day delay, Sec. III-E / IV-H). | τ_0 (`train_quantile`), controller offset folded into the score | delayed-ACI gate on e_t |
| L3 selection | Chronological split 1982–2010 / 2011–2016 / 2017–2025 (Sec. IV); AE early stopping on validation loss; SAC policy frozen on test (state-dependent, parameter-static inference). | — | — |
| L4 labels | No anomaly labels: e_t > τ_t routes the regime; metrics MAPE / RMSE / DA / Theil's U / CTR on prices. | — | — |
| L5 availability | **Unspecified** for the event-pathway context c_t: sentiment spike from a BERT model over X posts (2007–2025; zero before 2007), earnings features (source and release time not stated), VIX regime terciles; the availability time relative to the forecast is not stated. | context replaced by a realised-vol proxy channel | same |
| L6 universe | 20 named S&P 500 stocks 1982–2025 (16 from 1982, 4 from IPO): **today's constituents** (survivorship not discussed). | — | — |
| L7 execution | Close-at-t+h forecast from data to close t; no trading rule, timing or costs (no Sharpe/PnL reported). | runner prices our trades at t | at t + latency |
| L8 pre-trained | BERT fine-tuned on the Market Sentiment Evaluation dataset (cutoff not stated); not used here (`cutoff_date = None`). | — | — |

Unspecified / our choices (`AlRidhawiBaseline.unspecified`):
1. **Controller fallback**: SAC replaced by a deterministic bounded update driven by the same delayed reward r_t = −RMSE_t − 0.5(1 − DA_t) − 0.1|Δτ|: α hill-climbs on the reward with |Δα| ≤ 0.1; τ tracks the recent (60-day) 95th percentile of e_t with |Δτ| ≤ 0.1 τ_0.
2. Online adaptation on test: `controller_online: true` (default) keeps moving (τ, α) from the state during test (the paper's frozen policy still produces state-dependent actions); `false` freezes both at their end-of-validation values. Both readings are causal.
3. VIX unavailable: "stable day" = 20-bar realised volatility of the asset's own returns below the training 75th percentile.
4. Dual 6-layer, d = 512 node transformers replaced by two causal GRUs (hidden 8) on the window; the event pathway gets the realised-vol proxy as its context channel.
5. AE reconstructs the last 5 bars of the scaffold-normalised features (paper: the 17-dim daily feature vector); hidden (16, 8), latent 4.
6. Forecast target = next-bar log return / window return std (paper: normalised closes at h ∈ {1, 5, 20}).
7. τ_0 over all training rows (paper does not say all vs stable days only); α_0 = 0.5 (not stated).
8. No early stopping / validation selection in the smoke config (2 epochs).

---

## B10 — Kończal & Połoczański (arXiv:2608.09576, 2026) — status: `reimplemented`

EVT benchmark + one-bar-ahead prediction of anomaly indicators on 1-minute ETP returns. Score for the gate =
predicted probability that the next bar is an anomaly (D^any by default); direction = −1 (the paper studies left-tail
drops; the short-on-predicted-crash trade, the fill and the hold are **ours** — the paper reports no trading rule,
PnL or Sharpe).

| Channel | Original (pointer) | published mode | causal mode |
|---|---|---|---|
| L1 normalisation | Not stated (tree models; EVT on raw returns). | raw feature window (switch inert); logistic variant uses a train-fitted standardiser | same |
| L2 threshold | EVT: u = 95th percentile of **training** losses, GPD by ML on **training** exceedances, p = 0.01 tail quantile q_p = u + β/ξ[(p/ζ_u)^{−ξ} − 1] (Eq. 3); classifier: F1-optimal probability threshold computed **on the test set** (stated as descriptive, "not genuine out-of-sample"). | GPD train-fitted (`evt_sample: train`; `full_sample` = leaky one-switch variant, published only); `best_f1_test` against the indicator labels of the test rows | GPD train-fitted; delayed-ACI gate on the probability |
| L3 selection | Chronological 80/20 split, fixed hyper-parameters across instruments and targets, trained once, no retraining; the F1-optimal threshold uses test labels. | test-label threshold (logged by the Filtration Guard) | none |
| L4 labels | D^NR (no-recovery) uses the **next K = 10 bars**: a label only, never a feature. No point-adjust. | — | — |
| L5 availability | 1-minute OHLC, volume, bid/ask volumes, effective spread; features from bars strictly before the predicted bar. | — | — |
| L6 universe | Four ETPs chosen ex post (VanEck BTC/ETH on Xetra, Virtune BTC/ETH on Nasdaq Stockholm), inactive bars removed. | — | — |
| L7 execution | No trading rule / timing / costs. | runner prices our trade at t | at t + latency |
| L8 pre-trained | None. | — | — |

Indicators implemented (single venue): D^POT r_{t+1} < −q_p; D^MR r_{t+1} < rolling 1 % quantile ∧ Σ_{j=1..5} r_{t+1−j} > 0;
D^NR r_{t+1} < rolling 1 % quantile ∧ Σ_{k=2..11} r_{t+k} < 0.3|r_{t+1}| (label only); D^any = OR. D^CV (cross-venue rolling
OLS residual, n = 1440, p = 0.01) is not implemented (single-venue panels), so D^any = D^POT ∨ D^MR ∨ D^NR here is a subset of the paper's D^ANY.

Unspecified / our choices (`KonczalEVTBaseline.unspecified`):
1. XGBoost / LightGBM not installed: `HistGradientBoostingClassifier(class_weight='balanced', max_iter 60, depth 3)` is the GBM stand-in; `clf: logistic` available; random forest not wired.
2. Rolling 1 % quantile over n = 1440 bars, the 120-bar z-score and std_120 are clipped to the look-back window (48 bars).
3. 30 of the 34 features are computable from [log_ret, log_vol, hl_range, log_spread]: Kyle λ and order imbalance (and their 5-bar means) need signed order flow / bid-ask volumes and are omitted; the cross-venue indicator needs a second venue.
4. ATR = mean high-low range over the window (no previous-close term); Amihud = |r|/volume relative to its window mean.
5. GPD fitted per asset on the training rows (`evt_pooled: true` pools them), pooled-median fallback for unseen assets.
6. Published threshold chosen against the paper's indicator labels of the test rows (`f1_labels: indicator`); `planted` uses the dataset labels; train-quantile fallback if the test fold has no positive.
7. Classifier target D^any; class weights balanced (the paper's weights are not given).
8. Direction −1 on a predicted crash and the hold / fill rules are ours.

---

## Smoke results (planted panel seed 1, T = 1800, N = 6, fold 0: train 900 / val 200 / test 500, purge 16, embargo 10, 5 bps)

| Baseline | mode | AUC-PR | realised alarm rate | trades | FG violations | fit + predict |
|---|---|---|---|---|---|---|
| B1 CPD-DMN | published (always in market) | 0.065 | 1.000 | 3000 | 0 | 2.4 s |
| B1 CPD-DMN | causal (delayed ACI α = 0.05) | 0.056 | 0.050 | 151 | 0 | 1.4 s |
| B8 STAGE | published, `test_fold` reading | 0.062 | 0.073 | 220 | 6 | 0.8 s |
| B8 STAGE | published, `train_fold` reading | 0.062 | 0.036 | 107 | 0 | 0.8 s |
| B8 STAGE | causal | 0.052 | 0.046 | 139 | 0 | 0.9 s |
| B9 Al Ridhawi | published (τ_0 train quantile) | 0.083 | 0.051 | 152 | 0 | 2.2 s |
| B9 Al Ridhawi | causal (controller online / frozen: same alarms, SR −2.06 / −2.08) | 0.079 | 0.046 | 139 | 0 | 1.9 s |
| B10 Kończal-EVT | published (best-F1 on test indicators) | 0.059 | 0.031 | 92 | 6 | 1.1 s |
| B10 Kończal-EVT | published, `evt_sample: full_sample` | 0.052 | 0.042 | 126 | 12 | 1.1 s |
| B10 Kończal-EVT | causal | 0.057 | 0.048 | 143 | 0 | 1.2 s |

Test loader batch = 510 rows (a multiple of N = 6) so that no bar straddles two batches: `common.py::predict` resets the
gate stream only when a batch starts at a bar strictly earlier (`<`) than the previous batch's last bar (a new pass), so a
bar that straddles two batches no longer resets the delayed-ACI state; the table was produced with batch 510.

Threshold switches (B1, B4, B9, B10): the baseline-specific rule (B1 −∞ `always_in_market`, B4 beta, B9 adaptive τ, B10 best-F1 on indicator labels) applies only when the protocol asks for the as-published threshold (runner flag `threshold_source`/`as_published` when present, else the yaml `published_threshold` kind passed through unchanged); an explicit kind from an L2 / L3 switch, P-W or offline_all (test_quantile, combined_quantile, best_f1_test) is honoured. Operators trained on future-return targets are stamped at the last bar those targets read: B1 / B9 forecasters at t_end + 1, B10 classifier at t_end + 1 (+K for `nr` / `any`).

FG violations count the (operator, decision bar) uses logged by the Filtration Guard: in published mode they come
from threshold operators stamped at the test end (B8 `test_fold` / `full_sample`, B10 best-F1) or from the
`full_sample` GPD (B10), one per predict batch. The planted panel carries no exploitable change-point / tail structure
for these smoke sizes (AUC-PR ≈ base rate ≈ 0.05), so these numbers validate the plumbing, not the methods.

---

# Part B — B2–B7 (official-code families)


Companion of `causalgate/baselines/b2_anomaly_transformer.py`, `b3_damp.py`, `b4_memstream.py`, `b5_dcdetector.py`,
`b6_moment.py`, `b7_catch.py`. Every wrapper subclasses `causalgate.baselines.common.ScoreToTradeBaseline`, which owns the
leakage channels that are *not* model properties: L1 normalisation (`OfflineScaler` on `published_norm` rows vs
`StreamingNormaliser`), L2/L3 threshold (`published_threshold` read off the test scores vs the delayed-ACI gate calibrated on
the validation fold), the out dict and the trading rule. The wrapper owns the scoring function `s(window)` and this sheet.
L4 (point-adjust), L5 (availability join), L6 (universe) and L7 (fill timing) are applied by the protocol runner
(`causalgate.eval`, `docs/causalgate_research_plan.md` section 2.2); the tables below record what the *original* code does so
that the P-O protocol can replicate it.

**How the pointers were obtained.** The official repositories could not be cloned in the build session; the files were read
verbatim through the GitHub raw endpoints on 2026-10-03 (default branches: `main` for thuml/Anomaly-Transformer,
sihohan/DAMP, Stream-AD/MemStream, DAMO-DI-ML/KDD2023-DCdetector, moment-timeseries-foundation-model/moment; `master` for
decisionintelligence/CATCH) and the line numbers were counted on those copies. They are therefore marked `~L` (approximate):
grep the quoted statement in a clone to resolve the exact line. CATCH's `CATCH.py` was fetched method by method, so it is
cited by method name only.

Status vocabulary: `official` = the official model module imported from `third_party/` (see `THIRD_PARTY.md`);
`reimplemented` = the published algorithm reproduced statement for statement; `reimplemented-fallback` = a compact model
with the simplifications listed under *Unspecified / chosen here*; `wrapper-needed` = the pre-trained artefact is absent.

Published-mode defaults (the scaffold's `published_threshold`): B2, B5, B7 `combined_quantile` with `anomaly_ratio`
(the official `np.percentile(train + test energy, 100 - anomaly_ratio)`), B3 `train_quantile`, B4 `train_quantile`
(and the algorithm's own admission rule, see B4), B6 `best_f1_test`.

---

## B2 Anomaly Transformer (Xu, Wu, Wang, Long, ICLR 2022)

Registered `cg_anomaly_transformer`; official repo https://github.com/thuml/Anomaly-Transformer; status
`reimplemented-fallback` (`official` when `third_party/Anomaly-Transformer` imports **and** CUDA is available: `model/attn.py`
calls `.cuda()` unconditionally at ~L29 `self.distances = torch.zeros((window_size, window_size)).cuda()` and ~L51).

| Ch. | Official code / paper | Causal mode here |
|---|---|---|
| L1 | `data_factory/data_loader.py` `PSMSegLoader.__init__` ~L21-34: `self.scaler = StandardScaler()`, `self.scaler.fit(data)` on `train.csv`, `self.test = self.scaler.transform(test_data)`; same for MSL/SMAP/SMD loaders. ~L37 `self.val = self.test` (the validation split *is* the test set). | `StreamingNormaliser` (Welford, warm-up abstains); published mode: `OfflineScaler` on the train fold (`published_norm: train`, pass `norm_rows`). |
| L2 | `solver.py` `test()` ~L287-288: `combined_energy = np.concatenate([train_energy, test_energy], axis=0)`; `thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)`; `main.py` `--anormly_ratio` default 4.00 (%), the dataset scripts use 0.5-1 (the known test ratios). | Delayed-ACI gate calibrated on validation scores. Published: `combined_quantile`, `anomaly_ratio: 0.01`. |
| L3 | `solver.py` `train()` ~L196 `vali_loss1, vali_loss2 = self.vali(self.test_loader)` feeds `EarlyStopping` (patience 3) -> the checkpoint is selected on the test set; anomaly ratio = test ratio. | Fixed epochs, no test access; `anomaly_ratio` is a config constant. |
| L4 | `solver.py` `test()` ~L338-358: point-adjust loop (`anomaly_state`, forward and backward fill inside each labelled segment; comment cites issue #14) before `precision_recall_fscore_support`. | None (event-wise metrics, `causalgate.eval`). |
| L5 | n/a (no documents / late labels). | Protocol runner. |
| L6 | n/a (one multivariate series per dataset). | Protocol runner. |
| L7 | Detection only. `thre` loader scores every point of non-overlapping windows (`data_loader.py` `__getitem__` else-branch `index // self.step * self.win_size`). | Window ends at decision bar t; score = energy at the last position (`_energy(...)[:, -1]`); fill at t + latency. |
| L8 | None (trained per dataset). `cutoff_date = None`. | - |

Model and criterion (replicated): `model/attn.py` `AnomalyAttention` ~L21-56: `sigma = torch.sigmoid(sigma * 5) + 1e-5`,
`sigma = torch.pow(3, sigma) - 1`, `prior = 1/(sqrt(2 pi) sigma) * exp(-|i-j|^2 / 2 sigma^2)`, `series = softmax(QK^T / sqrt(E))`;
`mask_flag=False` (`model/AnomalyTransformer.py` ~L68). `solver.py` ~L12 `my_kl_loss`; ~L178-179 `loss1 = rec_loss - k * series_loss`,
`loss2 = rec_loss + k * prior_loss`; ~L188-190 minimax (`loss1.backward(retain_graph=True); loss2.backward()`); ~L212
`temperature = 50`; ~L244-245 `metric = torch.softmax((-series_loss - prior_loss), dim=-1)`, `cri = metric * loss`.

Unspecified / chosen here: Linear token embedding (official circular Conv1d k=3, `model/embed.py`); sizes
`d_model/n_heads/e_layers/d_ff` from `cfg['model']` (official 512/8/3/512, `win_size` 100, batch 1024, lr 1e-4, 10 epochs,
k=3); `(loss1 + loss2).backward()` (same gradients as the two backward calls); score at the last position of a sliding
window, reported as `(official energy)^(1/50)` computed in log space (`exp((log softmax(-dis * 50)[last] + log rec_err[last]) / 50)`)
- a positive monotone transform (identical ranking and quantile thresholds) that avoids float32 underflow: the one-sided prior makes the discrepancy
systematically larger at the window edges (measured on the planted panel: mean weight of the last position ~3e-14, exactly
0 in float32 for ~59 % of windows, i.e. the raw official energy ties at 0 and a quantile threshold becomes 5e-27);
no early stopping; per-epoch lr halving kept. Smoke: 16/2/1/32, 1 epoch, lookback 48.

---

## B3 DAMP (Lu, Wu, Keogh, KDD 2022)

Registered `cg_damp`; authors' code is MATLAB (`DAMP_2_0.m`, project site); cited Python port https://github.com/sihohan/DAMP
(`damp.py`, `utils.py`, numpy only); status `reimplemented` (`official` when `third_party/DAMP/damp.py` imports and
`DAMP_2_0(..., lookahead=0)` succeeds on every (asset, feature) stream; the port raises on constant regions and asserts m > 10).

| Ch. | Official code / paper | Causal mode here |
|---|---|---|
| L1 | None. Every subsequence is z-normalised inside `utils.py` `MASS_V2` (mean / std of the query and of every window), so the profile is invariant to per-feature affine scaling; `damp.py` `__main__` loads the raw series. | Scaffold L1 only decides where the warm-up NaNs are; profile values are identical under both modes. |
| L2 | None. `damp.py` ~L229-230 `discord_score, position = np.amax(left_mp), np.argmax(left_mp)` - the output is a discord ranking. | Delayed-ACI gate on the profile. Published: `train_quantile` (`anomaly_ratio: 0.05`). |
| L3 | `subsequence_length` and `location_to_start_processing` are user inputs (`damp.py` argparse defaults 100 / 400; docstring: start of the test set). | `m` from config; `t_start` = first training bar. |
| L4 | None (UCR-archive style: a hit if the top discord lies in the labelled region). | None. |
| L5 | n/a | - |
| L6 | n/a | - |
| L7 | `lookahead = 0` -> "a pure online algorithm with no pruning" (`damp.py` ~L113 and ~L203 comments). With `lookahead > 0` (default `2^nextpow2(16 m)`, ~L44) the forward MASS ~L115-133 / ~L205-223 reads future bars to prune, and pruned points get `left_mp[i-1] - 1e-05` (~L100, ~L144). | `lookahead = 0`; profile indexed by the subsequence END bar so that `lmp[t]` is F_t-measurable; fill at t + latency. |
| L8 | None. | - |

Algorithm 1 as implemented (`DampStream.extend`): prefix of `16 m` queries after `t_start` brute force over the whole past
(`damp.py` ~L92-111: `left_mp[i] = np.amin(MASS_V2(time_series[:i], query))`, `best_so_far = np.amax(left_mp)`); afterwards
`X = 2^nextpow2(8 m)` (~L152), `while approximate_distance >= best_so_far` (~L166): nearest X bars first (Case 2, ~L181-185),
then farther segments (Case 3, ~L187-192) with `X *= 2` (~L200); stop with an approximate value when a distance below
best-so-far is found (~L194-197), exact value and best-so-far update when the beginning is reached (Case 1, ~L167-179).
Verified in the build session: `exact=True` equals `LeftMatrixProfileScorer.score_series` to 0.0; the DAMP values are upper
bounds with identical running maximum and top discord.

Unspecified / chosen here: multivariate input = per-feature profiles averaged (`features: all`); candidates must end
<= t - m (non-overlapping, as MASS over `time_series[:i]`); the port's first backward segment `time_series[i-X+1:i+1]`
contains overlapping subsequences - not replicated; prefix 16 m and initial window 2^nextpow2(8 m) from the port;
`exact: true` disables the early stop; the window-local fallback (`LeftMatrixProfileScorer`) divides by the training median.
Smoke: m = 8, lookback 48.

---

## B4 MemStream (Bhatia, Jain, Srivastava, Kawaguchi, Hooi, WWW 2022)

Registered `cg_memstream`; official repo https://github.com/Stream-AD/MemStream (`code/memstream.py`, autoencoder features);
status `reimplemented` (the official file is an argparse script, not importable; `THIRD_PARTY.md` says how the vendored copy
is detected).

| Ch. | Official code / paper | Causal mode here |
|---|---|---|
| L1 | `memstream.py` ~L74-77 `train_autoencoder`: `self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)`; `new = (data - self.mean) / self.std`; `new[:, self.std == 0] = 0`; ~L92 the statistics are recomputed after every memory update; ~L106-107 the same standardisation in `forward`. Statistics of the records currently in memory (initially the first N normal records). | Scaffold `StreamingNormaliser` first, then the same memory statistics (replicated). Published: `OfflineScaler` on the train fold, then memory statistics. |
| L2 | No alarm threshold: ~L141 `auc = metrics.roc_auc_score(labels, scores)` over the whole stream. The only decision threshold is beta, the memory-admission rule ~L88 `if output_loss <= self.max_thres` (`--beta` default 0.1, ~L16; per-dataset values in the README). | Delayed-ACI gate on the prequential scores. Published: `published_rule: beta` -> alarm iff the record is not admitted (score > beta); `published_rule: quantile` -> `train_quantile` of the streamed train scores. |
| L3 | ~L130 `init_data = numeric[labels == 0][:N]`: the initial memory is the first N records *with label 0*; beta / memlen chosen per dataset with the test labels in view (README). | Causal: first N training records regardless of label. Published: the official label filter on the train fold (`init_normal_only: null` -> on only for the as-published row or an L3 `best_f1_test` threshold, never for an L1- or L2-only switch). beta: config value or the `beta_quantile` (0.95) quantile of the leave-one-out nearest-neighbour L1 distances inside the initial memory. |
| L4 | None (ROC-AUC). | None. |
| L5 | n/a (records in file order). | - |
| L6 | n/a | - |
| L7 | Prequential: ~L105-110 `forward` scores the record, then `update_memory` (FIFO slot `self.count % self.memory_len`, ~L89). | Same; rows streamed in (t, asset) order; a replay restores the post-calibration snapshot; fill at t + latency. |
| L8 | None. | - |

Model (replicated): ~L60-66 encoder `Linear(in, 2 in) + Tanh`, decoder `Linear(2 in, in)`; ~L78-84 full-batch Adam
(`--lr` 1e-2, `--epochs` 5000) on `new + 0.001 * randn`; ~L97-103 `initialize_memory`; ~L109 score
`torch.norm(self.memory - encoder_output, dim=1, p=1).min()` (k = 1 nearest neighbour, L1).

Unspecified / chosen here: record = last `record_bars` (1) bars of the normalised window flattened; beta rule above;
memory size / epochs reduced in the smoke config (256 / 200); fewer than N records -> tiling; batch rows streamed in
(t, asset) order; the scaffold's train, validation and published-threshold passes are streamed in time order exactly as
the official single pass over the record stream.

---

## B5 DCdetector (Yang, Zhang, Zhou, Wen, Sun, KDD 2023)

Registered `cg_dcdetector`; official repo https://github.com/DAMO-DI-ML/KDD2023-DCdetector (a fork of the
Anomaly-Transformer code base: same loader and solver skeleton); status `reimplemented-fallback` (`official` when
`third_party/KDD2023-DCdetector` imports; needs `einops` and `tkinter`).

| Ch. | Official code / paper | Causal mode here |
|---|---|---|
| L1 | `data_factory/data_loader.py` `PSMSegLoader` ~L21-34 `StandardScaler` fitted on the train file, applied to train and test, `self.val = self.test` (identical code to B2; eleven dataset loaders, all fit on the train array). Plus per-window instance normalisation inside the model: `model/DCdetector.py` ~L62-65 `revin_layer = RevIN(num_features=M)`, `x = revin_layer(x, 'norm')`. | `StreamingNormaliser` + the same window-internal RevIN (F_t-measurable). Published: `OfflineScaler` on the train fold. |
| L2 | `solver.py` `test()` ~L261-262 `combined_energy = np.concatenate([train_energy, test_energy])`, `thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)`; `main.py` default 4.00 %, scripts 0.5-1 %. | Delayed-ACI gate. Published: `combined_quantile`, `anomaly_ratio: 0.01`. |
| L3 | `solver.py` `train()` ~L180 `self.vali(self.test_loader)` -> early stopping on the test set (patience 5); `main.py` picks the batch size from the data length. | Fixed epochs. |
| L4 | `solver.py` ~L309-329 point-adjust before P/R/F1, and ~L304 `combine_all_evaluation_scores(pred, gt, test_energy)` (`metrics/metrics.py`: PA-F1, MCC, affiliation P/R, R_AUC, VUS) - called with `(pred, gt)` although the signature is `(y_test, pred_labels, anomaly_scores)`. | None. |
| L5 | n/a | - |
| L6 | n/a | - |
| L7 | Detection only, non-overlapping test windows scored point by point. | Window ends at t, score = softmax energy at the last position; fill at t + latency. |
| L8 | None. | - |

Model (compact): `model/attn.py` `DAC_structure` ~L24-36 patch-wise and in-patch softmax attention, ~L38-42 upsampling
(`repeat` for the patch-wise map, `.repeat` tiling for the in-patch map) and `reduce(..., 'mean', reduce_b=self.channel)`;
`model/DCdetector.py` ~L74 `'b m (n p) -> (b m) n p'` and ~L76 `'b m (p n) -> (b m) p n'` (the in-patch view is a block split,
not a transpose - replicated as is). Loss `solver.py` ~L168 `loss = prior_loss - series_loss` (series = patch-wise, prior =
in-patch, stop-gradient KL pairs ~L15 `my_kl_loss`); score ~L196 `temperature = 50`, ~L290-291
`metric = torch.softmax((-series_loss - prior_loss), dim=-1)`, `cri = metric` (no reconstruction term).

Unspecified / chosen here: Linear + sinusoidal embedding (official circular Conv1d); `patch_sizes` must divide the lookback
(official scripts: patch_size 357 = [3,5,7] with win_size 105 for SMAP/SWaT, 57 @ 105 SMD, 35 @ 90 MSL, 135 @ 60 PSM; `main.py` defaults win_size 100, patch_size [5]); the `e_layers` attention layers read the same embeddings (as the
official Encoder); loss sign kept as in the official code (`variant: official_code`; `paper` = additive loss and raw
discrepancy as score); d_model/n_heads/e_layers from config (official 256/1/3, batch 128, lr 1e-4; `main.py` default 10 epochs, the official scripts train 2-3); score at the
last position, reported as `softmax(-dis * 50)[last] ^ (1/50)` computed in log space (positive monotone transform of the
official metric, no float32 underflow).
Capacity matters for the official-code sign on the planted panel: d_model 16 / 1 layer / 1 epoch ranked the anomalies
*below* normal bars (AUC_ROC 0.33) while d_model 32 / 2 layers / 3 epochs gave 0.63. Smoke: patch sizes [4, 8], d_model 32,
2 layers, 2 epochs, lookback 48.

---

## B6 MOMENT zero-shot anomaly detection (Goswami, Szafer, Choudhry, Cai, Li, Dubrawski, ICML 2024)

Registered `cg_moment`; official repo https://github.com/moment-timeseries-foundation-model/moment, pip `momentfm`,
checkpoints HF `AutonLab/MOMENT-1-{large,base,small}`; status `official` when `momentfm` imports, else `wrapper-needed`
(`fit` raises `ImportError` with the install line). `cutoff_date = '2024-02-01'`.

| Ch. | Official code / paper | Causal mode here |
|---|---|---|
| L1 | `momentfm/models/moment.py` `reconstruction()`: `x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")` (per-window RevIN with the input mask, de-normalised output). The tutorial's `AnomalyDetectionDataset` (`momentfm/data/anomaly_detection_dataset.py`) applies its own scaler - statistics not verified here. | `StreamingNormaliser` + the model's window-internal RevIN. |
| L2 | No operating threshold. `tutorials/anomaly_detection.ipynb` last cells: `anomaly_scores = (trues - preds)**2`; `adjbestf1(y_true=labels, y_scores=anomaly_scores)`; `momentfm/utils/anomaly_detection_metrics.py` ~L61-77 `adjbestf1`: `thresholds = np.linspace(y_scores.min(), y_scores.max(), n_splits)` (100), best point-adjusted F1 over the TEST scores. | Delayed-ACI gate. Published: `best_f1_test` (scaffold best-F1 threshold on the test labels, no PA). |
| L3 | Best-over-thresholds on the test labels (above); otherwise zero-shot. | None. |
| L4 | Point-adjust inside `adjbestf1` (`adjust_predicts` ~L15-58). | None. |
| L5 | n/a | - |
| L6 | n/a | - |
| L7 | Detection only: 512-step windows, every point scored, overlapping windows de-duplicated in the notebook. | Window ending at t left-padded to 512 with `input_mask = 0`; score = mean over channels of the squared error at the last bar; fill at t + latency. |
| L8 | Pre-trained on the Time Series Pile (README: "We compile a large and diverse collection of public time-series, called the Time-series Pile", HF `AutonLab/Timeseries-PILE`); MOMENT-1 released with the paper (arXiv 2402.03885, 6 Feb 2024; HF model files updated May 2024). | `cutoff_date = 2024-02-01` is the release date, not a non-overlap proof: the pile's contents (UCR/UEA, Monash, TSB-UAD anomaly archives, ...) and collection dates must be checked against each fold's test start by the protocol runner. |

Unspecified / chosen here: left zero-padding of short windows; F features as C independent channels; last-bar score; no
fine-tuning; `hf_name` selects the checkpoint; the tutorial dataset's scaler is replaced by the scaffold's L1.

---

## B7 CATCH (Wu, Qiu, Li, Wang, Hu, Guo, Xiong, Yang, ICLR 2025)

Registered `cg_catch`; official repo https://github.com/decisionintelligence/CATCH (branch `master`, TFB benchmark layout:
`ts_benchmark/baselines/catch/CATCH.py`, `models/CATCH_model.py`, `layers/channel_mask.py`, `layers/cross_channel_Transformer.py`,
`layers/RevIN.py`; evaluation `ts_benchmark/evaluation/strategy/anomaly_detect.py`, `evaluation/metrics/*`); status
`reimplemented-fallback` (`official` when `third_party/CATCH` makes `ts_benchmark` importable).

| Ch. | Official code / paper | Causal mode here |
|---|---|---|
| L1 | `CATCH.py` `__init__`: `self.scaler = StandardScaler()`; `detect_fit`: `train_data_value, valid_data = train_val_split(train_data, 0.8, None)`, `self.scaler.fit(train_data_value.values)` (first 80 % of the train split), both splits transformed; `detect_label` / `detect_score`: `test = pd.DataFrame(self.scaler.transform(test.values), ...)`. Plus per-window RevIN inside the model (`CATCH_model.py` `forward`: `z = self.revin_layer(z, 'norm')` ... `'denorm'`, hyper `revin: 1`). | `StreamingNormaliser` + window-internal RevIN. Published: `OfflineScaler` on the train fold. |
| L2 | `detect_label`: train energy from `self.train_data_loader`, test energy from the test loader, `combined_energy = np.concatenate([train_energy, test_energy], axis=0)`; `for ratio in self.config.anomaly_ratio: threshold = np.percentile(combined_energy, 100 - ratio); preds[ratio] = (test_energy > threshold)`; default `anomaly_ratio: [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25]` (`DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS`), the MSL label script passes `"anomaly_ratio": 5.0`. `detect_score` returns the raw energies. | Delayed-ACI gate. Published: `combined_quantile`, `anomaly_ratio: 0.05`. |
| L3 | Early stopping (patience 3) on the last 20 % of the train split (`detect_validate`); the label protocol evaluates one prediction per anomaly ratio on the test labels (`anomaly_detect.py` `AnomalyDetect.execute`: one result row per ratio) - which ratio is reported is decided by the report step (`report_config.aggregate_type: mean`, not verified). | Fixed epochs, one ratio. |
| L4 | TFB metrics `"all"`: label metrics `classification_metrics_label.py` include `adjust_precision / adjust_recall / adjust_f_score` (point-adjust via `adjust_predicts`) and `affiliation_f` (the reported metric in `config/unfixed_detect_label_multi_config.json`); score metrics `classification_metrics_score.py` `best_f_score / best_precision / best_recall` (threshold searched on the test labels through `precision_recall_curve`), `auc_roc` (reported in `unfixed_detect_score_multi_config.json`), `auc_pr`, `R_AUC_*`, `VUS_*`. | None. |
| L5 | n/a | - |
| L6 | n/a | - |
| L7 | Detection only; `anomaly_detection_data_provider(..., step=1)` sliding windows; every point scored. | Window ends at t; score = per-point energy at the last bar; fill at t + latency. |
| L8 | None. | - |

Model (official `CATCH_model.py` `forward`): RevIN -> `torch.fft.fft(z)` -> real / imag `unfold(patch_size, patch_stride)` ->
per (window, patch) channel tokens `z_cat = cat(real, imag)` -> `channel_mask_generator(z_cat)` (`channel_mask.py`:
zero-initialised `Linear(input_size * 2, n_vars, bias=False)` + Sigmoid, `_bernoulli_gumbel_rsample` with
`gumbel_softmax(..., hard=True)`, diagonal forced to 1) -> `Trans_C(z_cat, channel_mask)` returning `(z, dcloss)` -> `get_r`,
`get_i` -> `Flatten_Head` x2 -> `torch.complex` -> `torch.fft.ifft` -> `self.ircom(cat(real, imag))` -> RevIN denorm.
Loss (`detect_fit`): `rec_loss + dc_lambda * dcloss + auxi_lambda * auxi_loss` with `frequency_loss` on
`(output_complex, revin_layer(input, 'transform'))`. Score: `temp_score + score_lambda * freq_score` per point.

Unspecified / chosen here (compact fallback): dcloss omitted and the mask applied multiplicatively to the softmax
attention weights (the `Trans_C` masking rule and its contrastive regulariser were not read); auxiliary loss = MAE on
real/imag of the spectrum; score = temporal MSE only (no `frequency_criterion` term); mask deterministic (p > 0.5) at scoring
time (the official module samples); heads = one Linear per real/imag (official three residual ReLU layers + Linear); one Adam
with two parameter groups (lr / Mlr) instead of two optimisers with OneCycleLR; sizes from config (official seq_len 192,
patch 16 / stride 8, d_model 128-256, 3 layers, 3-5 epochs, batch 128). Smoke: patch 8 / stride 8, d_model 16, 1 layer,
1 epoch, lookback 48.

---

## Notes for the protocol runner

* Published mode with `published_norm: train` needs `cfg['norm_rows']` (the train fold index) or the scaffold falls back to
  the global scaler.
* B3 and B4 override `_collect` / `_windows` to keep the batch (t, asset) bookkeeping that the panel profile (B3) and the
  streaming memory (B4) need; both still return the scaffold's `(X, y, t)`.
* Replay detection: a bar whose rows straddle two batches makes the next batch start at the previous batch's last bar.
  B4 therefore treats only a *strictly earlier* first bar as a replay (`t.min() < last_t`), the same rule the scaffold's gate
  reset now uses (`common.py` `predict`, `int(t.min()) < self._last_t`). With the earlier `<=` comparison the delayed-ACI
  state was reset at 8 of 12 batch boundaries in the smoke run (batch 256, N = 6).
* All five trainable baselines fit + predict on the planted panel (T = 1800, N = 6, one fold) in 1-4 s CPU with the smoke
  configs; the full configs are 5-20x larger.
