# Benchmark Audit — datasets and baselines with verification status

Purpose: a CCF-A submission cannot cite a dataset or a method that does not exist. This file records, for every item in the
task brief, whether it could be verified, what the correct year/venue is, and which verified substitute is wired into the code
when it could not. "†" entries in `../paper_catalogue.md` are themselves flagged there as partially unverified — check the DOI /
arXiv id before citing.

Status legend: **PUBLIC** freely downloadable · **LICENSED** academic licence or vendor data · **CONSTRUCTED** assembled from
public APIs with the recipe in the builder docstring · **PLACEHOLDER** no verifiable dataset/paper by that name.

## A. Datasets (builders in `warn2trade/data/datasets/`)

| # | Brief name | Status | Verified source | What is wired / substitute | Label source | Freq | Modalities |
|---|---|---|---|---|---|---|---|
| D1 | StockNet / Tweet-Stock | PUBLIC | Xu & Cohen, ACL 2018, github.com/yumoxu/stocknet-dataset (88 stocks, 2014–2015) | `stocknet` | rule labels (abnormal move + volume) | daily | price + text + sector graph |
| D2 | LOBSTER L3 | LICENSED | lobsterdata.com (academic) ; free substitute FI-2010 (Ntakaris et al. 2018) | `lobster` (variant fi2010) | spoof injection protocol (Lin & Yang 2025, arXiv:2508.17086) + rule labels; no public spoofing ground truth | 100 ms | LOB + price |
| D3 | FNSPID | PUBLIC | Dong et al., KDD 2024, HF `Zihan1004/FNSPID` | `fnspid` | rule labels / earnings-event table | daily | price + news + co-mention graph |
| D4 | SEC EDGAR 10-K/10-Q MD&A | PUBLIC text / LICENSED labels | EDGAR full text, EDGAR-CORPUS (Loukas et al. 2021); AAER fraud labels (Bao et al., JAR 2020, public); Audit Analytics restatements (licensed) | `edgar` (label_source aaer) | AAER / restatement flag per filing | filing-day | text + price + auditor/industry graph |
| D5 | Crypto pump-and-dump / "DEX-Anomalies" | PUBLIC / PLACEHOLDER | La Morgia et al., ACM TOIT 2023 (arXiv:2105.00733) Telegram pump dataset; Hu et al., SIGMOD 2023 (arXiv:2204.12929). "DEX-Anomalies" not found → rug-pull sets: Yaremus et al. 2025 (TON, arXiv:2509.01168), Cao et al. 2026 (BSC, arXiv:2603.13830) | `crypto_pd` | curated pump timestamps (exact peak → signed WLT) | 1 min | price + Telegram text + channel graph |
| D6 | CSI 300 limit-up/down | CONSTRUCTED | Qlib CN bundle / Tushare / AkShare; LOBench (Zhong, Lin, Yang 2025, arXiv:2505.02139) for A-share LOB | `csi300_limit` | limit-hit event table (exact) | daily | price + announcements + industry graph; tradability mask is the key feature |
| D7 | S&P 500 tick-level crashes | LICENSED / CONSTRUCTED | NYSE TAQ (licensed) or Polygon/IEX minute bars; mini-flash-crash definition of Johnson et al. 2013; Liu, Xu, Zheng SSRN 4992077 | `sp500_tick` | mini-flash-crash rule + LULD halt calendar | 1 min | price/LOB-lite |
| D8 | Qlib Alpha158 (US & CN) | PUBLIC | Microsoft Qlib | `qlib_alpha158` | rule labels (extreme outliers) | daily | tabular + industry graph |
| D9 | WSB / StockTwits short squeeze | PUBLIC / API | Kaggle "Reddit WallStreetBets Posts" (G. Preda), Pushshift; StockTwits API; FINRA short interest | `wsb_squeeze` | squeeze events (short interest + return rule), broker-restriction tradability mask | daily | price + posts + co-mention graph |
| D10 | "FinWeb-Multimodal Fraud/Scam" | PLACEHOLDER | not found. Substitutes: Elliptic++ (Elmougy & Liu, KDD 2023) + BTC hourly prices (closes the loop); DGraph-Fin (Huang et al., NeurIPS 2022 D&B) detection-only | `fin_graph` (variant ellipticpp) | illicit-entity labels | 1 h | graph + price |

Recommendation: keep ten slots, but present D2/D7 as "licensed, with free substitutes reported", and rename D5/D10 to the
verified datasets. Reviewers check dataset names.

## B. Baselines (`warn2trade/baselines/`)

| # | Brief name (year as given) | Verified? | Correct citation | Code status | Note |
|---|---|---|---|---|---|
| B1 | MASTER / MAN-SF (2021) | partly | MAN-SF: Sawhney et al., **EMNLP 2020**. MASTER: Li et al., **AAAI 2024** | `man_sf`, `master` wrapper-needed | years in the brief are wrong; both are real |
| B2 | PatchTST / DLinear (2022) | yes | PatchTST: Nie et al., **ICLR 2023**; DLinear: Zeng et al., **AAAI 2023** (arXiv 2022) | `dlinear` implemented, `patchtst` wrapper-needed | |
| B3 | TradeMaster / FinGPT (2023) | yes | Sun et al., NeurIPS 2023 D&B; Yang et al. 2023 (arXiv:2306.06031) | wrapper-needed | FinGPT used as sentiment → rule |
| B4 | StockFormer / Time-LLM (2024) | partly | StockFormer: Gao et al., **IJCAI 2023**; Time-LLM: Jin et al., ICLR 2024 | wrapper-needed | |
| B5 | FinAgent / Market-Agent (2024) | partly | FinAgent: Zhang et al., KDD 2024. "Market-Agent" not identified | `finagent` wrapper-needed | |
| B6 | UniTS / FinDiffusion (2025) | partly | UniTS: Gao et al., **NeurIPS 2024**. "FinDiffusion": no paper found | `units` wrapper-needed; `findiffusion` placeholder | LOB generative substitutes: Hultin et al. 2023; Nagy et al. 2023 (arXiv:2309.00638) |
| B7 | Causal-FinRL (2025) | no | not found | placeholder | substitute: MacroHFT (Zong et al., KDD 2024) or Asadi & Safabakhsh 2026 (arXiv:2608.28252†) |
| B8 | Graph-LLM Financial Auditor / FinKGC (2025) | no | not found | placeholder | substitute: Xiang et al., WWW 2025 dynamic-graph spoofing (arXiv:2510.05562†); PERSEUS (arXiv:2503.01686) |
| B9 | DynoAlpha-2026 | no | not found | placeholder | substitute (2026, from the catalogue): PumpSense, IEEE ICBC 2026 (arXiv:2605.09431); Al Ridhawi et al. 2026 (arXiv:2603.19136) |
| B10 | OmniFin-LLM (2026) | no | not found | placeholder | substitute (2026): Yang 2026 (arXiv:2604.10996); Asadi & Safabakhsh 2026 (arXiv:2608.28252) |

Extra baselines that are *closer to the paper's question* than several in the brief, all verified in the catalogue:
- **CPD + Deep Momentum Network** (Wood, Roberts, Zohren, JFDS 2022) — change-point score switches momentum/reversion: the
  best-known "anomaly → trade" method. Wired as `cpd_dmn`. Strongly recommended as the headline baseline.
- **Deep OFI** (Kolm, Turiel, Westray, Math. Finance 2023) — LOB alpha without anomaly conditioning. `deep_ofi`.
- **Cascaded contrastive LOB spoofing** (Lin & Yang 2025) — detection-only SOTA for D2. `lob_cascade_contrastive`.
- **Open-loop reference** `detect_then_rule` — our own detector trained with plain BCE and handed to a fixed rule. This is the
  single most important comparison for the "closing the loop" claim and is fully implemented.
- Rules `zscore_momentum`, `zscore_reversal` — what practitioners actually do with a surveillance alert.

### Recommended final set of ten (all verifiable, ≥ 2 from 2026)
1. CPD-DMN (2022) 2. DLinear/PatchTST (2023) 3. MAN-SF (2020) or MASTER (2024) 4. TradeMaster-PPO (2023) 5. FinGPT-sentiment (2023)
6. StockFormer (2023) 7. FinAgent (2024) 8. MacroHFT (2024) 9. UniTS (2024) or Time-LLM (2024)
10. 2026 pair: PumpSense (ICBC 2026) for the manipulation datasets and Asadi & Safabakhsh 2026 / Yang 2026 for the regime/LLM-RL
datasets — confirm each arXiv id resolves before citing (several 2026 ids in the catalogue carry the † flag).

## C. Verification to do before submission
- [ ] Resolve every arXiv id above; drop anything that does not resolve.
- [ ] Obtain LOBSTER and TAQ licences or state the FI-2010 / Polygon substitutes in the paper.
- [ ] Confirm Audit Analytics access; otherwise use AAER only and say so.
- [ ] Rename D5 and D10 in the paper to the datasets actually used.
