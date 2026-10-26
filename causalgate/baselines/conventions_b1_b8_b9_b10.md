# Convention sheets: B1 CPD-DMN, B8 STAGE, B9 Al Ridhawi et al., B10 Kończal & Połoczański

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
