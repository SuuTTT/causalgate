# Convention sheet, baselines B2-B7 (CausalGate track)

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
