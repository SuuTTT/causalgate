# Download recipes for the D4 / D6 / D7 / D9 / D10 CausalGate loaders

Companion to `causalgate/data/loaders/{edgar,csi300_limit,sp500_tick,wsb_squeeze,fin_graph}.py`
(registered as `cg_edgar`, `cg_csi300_limit`, `cg_sp500_tick`, `cg_wsb_squeeze`, `cg_fin_graph`) and
`configs/causalgate/datasets/<name>.yaml`. To be merged into `scripts/cg_prepare_data.py` (owned by the D1/D2/D3/D5/D8
sub-task). Nothing below is downloaded by the loaders themselves: every loader raises `FileNotFoundError` with the
recipe when a required file is missing and never synthesises prices, documents, labels or timestamps.

Conventions shared by the five loaders
- Panel `mid` = close; `price_feat` is RAW (`meta['raw_features'] = True`, column 0 = log return, `meta['ret_col'] = 0`,
  names in `meta['feat_names']`); no full-sample statistics are used anywhere in a loader.
- PIT fields are attached through `common.finish_panel` and listed in `meta['pit_fields']`; `meta['pit_notes']` has one
  line per field; `meta['licence']`, `meta['source']`, `meta['layout']` and `meta['assumptions']` are always present.
- Column names are resolved case-insensitively with the documented fallbacks; uncertain names are called out below.
- `root` in the yaml is the dataset folder; file names are relative to it.

---

## D4  `cg_edgar`  — EDGAR 10-K/10-Q + AAER labels + Yahoo daily prices   (`root: data_raw/edgar`)

```
data_raw/edgar/
  filings.csv          ticker|cik, form, fyear|fiscal_year (or period), accepted|acceptance_datetime (ET) | date_filed,
                       sic (opt), accession (opt), emb_row (opt)
  aaer.csv             ticker|cik|gvkey, fyear, misstate, aaer_release_date (user-added), p_aaer (opt)
  prices/<TICKER>.csv  Yahoo daily (Date, Open, High, Low, Close, Adj Close, Volume)
  tickers.csv          opt  cik, ticker            (sec.gov/files/company_tickers.json -> csv)
  gvkey_cik.csv        opt  gvkey, cik             (Compustat, LICENSED; only if aaer.csv is gvkey-keyed)
  embeddings.npy       opt  float32 (n_filings, D) or (n_filings, C, D), row-aligned with filings.csv (or via emb_row)
  sic.csv              opt  ticker|cik, sic
  restatements.csv     opt  Audit Analytics export (LICENSED) -- only read when label_source: restatement
```

Steps
1. Filing index with acceptance datetimes. The EDGAR quarterly full index
   (`https://www.sec.gov/Archives/edgar/full-index/<YYYY>/QTR<n>/master.idx`) gives CIK, form type, date filed and the
   filing path but **not** the acceptance time; take `<ACCEPTANCE-DATETIME>` from each filing's header
   (`https://www.sec.gov/Archives/edgar/data/<CIK>/<accession>.txt`, 10 req/s, declare a User-Agent) or use the EDGAR
   full-text search API (`efts.sec.gov/LATEST/search-index?q=...`, field `file_date` + `period_ending`). EDGAR-CORPUS
   (`https://huggingface.co/datasets/eloukas/edgar-corpus`, Loukas et al. 2021) provides the parsed item text (incl. Item 7
   MD&A) keyed by `cik` and `year`, but no acceptance time: join it to the index by CIK + fiscal year.
   A date-only `date_filed` is treated by the loader as accepted after the close (next session), the conservative reading.
2. AAER labels: `https://github.com/JarFraud/FraudDetection` (Bao, Ke, Yu, Zhang & Zhang, JAR 2020). The public file
   (`uscecchini28.csv`, to the best of our knowledge) carries `fyear, gvkey, p_aaer, new_p_aaer, misstate` + 28 financial
   items; it does **not** carry the AAER release date. Build `aaer_release_date` by joining `p_aaer` (the AAER number) to
   the SEC AAER index (`https://www.sec.gov/enforcement-litigation/accounting-auditing-enforcement-releases`), and map
   gvkey -> cik -> ticker (Compustat link table is licensed; a public alternative is the SEC's company_tickers.json by
   CIK plus a manual gvkey map for the ~1 000 fraud firm-years). Rows with `misstate = 1` and no release date are kept
   with `avail_label = NEVER` and counted in `meta['n_fraud_without_release']`.
3. Prices: Yahoo daily CSV per ticker (e.g. `yfinance` on a machine with network access; keep the raw csv).
4. Optional MD&A embeddings: chunk Item 7 / Item 2 text, embed with a checkpoint whose training cutoff is declared
   (L8), save `embeddings.npy` and the `emb_row` column.

Licence: EDGAR text public domain; AAER labels public from the JAR 2020 release (cite the paper); Yahoo prices
personal/research use per Yahoo terms; Audit Analytics LICENSED and never required.

PIT: `avail_text` = filing bar (first close at/after acceptance); `avail_label` = AAER release bar (a date-only release counts as after the close -> next session close, as for date-only filings; NEVER if unknown) for
misstate = 1, `t + clean_label_lag_bars` otherwise; `avail_graph` = t (SIC-2 static); `universe` = has a price path.
Decision points are the filing bars (`meta['decision_mask']`); the time axis is trading days so horizons are exact.

---

## D6  `cg_csi300_limit`  — A-share daily bars, limit-hit events   (`root: data_raw/csi300_limit`)

Layout (A) Qlib bundle
```
data_raw/csi300_limit/qlib_cn/
  calendars/day.txt                   one YYYY-MM-DD per line
  instruments/csi300.txt              "<INST>\t<start>\t<end>" (multiple rows per instrument = discontinuous membership)
  features/<inst lower-case>/{open,high,low,close,volume,factor}.day.bin
                                      little-endian float32; first element = calendar index of the first value
```
```
pip install pyqlib            # on a machine with network access
python -m qlib.run.get_data qlib_data --target_dir data_raw/csi300_limit/qlib_cn --region cn
```
Assumption to verify on your bundle: `$close` is adjusted and `$factor` is the adjustment factor (traded price =
`$close / $factor`). Set `qlib_prices: raw` if your bundle stores unadjusted prices.

Layout (B) Tushare / AkShare export
```
data_raw/csi300_limit/
  bars.csv  (or csv/<code>.csv)   ts_code|code, trade_date, open, high, low, close, vol, amount, pre_close (opt),
                                  up_limit, down_limit (opt, exact; Tushare stk_limit), adj_factor (opt)
  members.csv                     ts_code|code, in_date|start, out_date|end        (Tushare index_weight / index_member)
  st_flags.csv            opt     code, start, end  (ST / *ST periods; Tushare namechange)
```
Tushare needs a token (`pro.daily`, `pro.stk_limit`, `pro.index_weight`, `pro.namechange`); AkShare
`stock_zh_a_hist` / `index_stock_cons_csindex` need no token. Both are rate-limited; export once and keep the csv.

Labels (constructed, ours): close at the daily limit; 10 % default, STAR (688/689) 20 %, ChiNext (300/301) 20 % from
2020-08-24 (10 % before), BSE (8xx/4xx) 30 %, ST 5 % (needs st_flags.csv; otherwise ST hits are missed). Event = run
of consecutive same-direction limit closes; label on [t_hit - lead, run end]; `bars_to_peak = run end - t`.
`tradable = False` on every limit-hit bar (conservative whole-bar reading), direction in `meta['limit_direction']`.

Licence: Qlib bundle research use per the Qlib repository; Tushare / AkShare per their terms.

PIT: `avail_label` = last limit bar of the run + 1 (the run end is first observable then; NEVER if still open at the panel end), `t + max(horizon)` elsewhere; `universe` as-of from
the membership ranges (a change effective and knowable at its start date, L6); without a membership file the universe
degrades to "has a price" and `meta['universe_source']` says so.

---

## D7  `cg_sp500_tick`  — S&P 500 1-minute bars, LULD halts, circuit breakers   (`root: data_raw/sp500_tick`)

```
data_raw/sp500_tick/
  bars/<TICKER>.csv       timestamp|t, open|o, high|h, low|l, close|c, volume|v, vwap|vw (opt), trades|n (opt)
                          (parquet is NOT readable here -- pyarrow missing -- export csv)
  halts.csv               ticker, halt_start, halt_end
  circuit_breakers.csv    start, end          (date-only rows mark the whole session untradable)
  constituents.csv        ticker, add_date, remove_date (blank = current)
```
1. Bars: Polygon `GET /v2/aggs/ticker/{T}/range/1/minute/{from}/{to}?adjusted=false&limit=50000` (paid plan for the full
   history; results `t` = bar START in epoch ms, `o,h,l,c,v,vw,n`) or IEX Cloud `/stock/{sym}/chart/date/{YYYYMMDD}`
   per day. Write one csv per ticker. NYSE TAQ (WRDS) is LICENSED: only a user's own export can be read.
2. Halts: Nasdaq Trader "Trading Halts" (`https://www.nasdaqtrader.com/trader.aspx?id=TradeHalts`, current list; the
   historical archive requires an export) or the NYSE / CTA halt files; keep `(ticker, halt_start, halt_end)` with the
   resumption time as `halt_end`.
3. Circuit breakers: 2020-03-09 09:34-09:49, 2020-03-12 09:35-09:50, 2020-03-16 09:30-09:45, 2020-03-18 12:56-13:11 ET
   (level-1 halts; verify against the exchange notices before use) -> `circuit_breakers.csv`.
4. Constituents with add / remove dates: a vendor file or a scrape of the Wikipedia "Selected changes" table.

Labels (constructed): minute-bar ADAPTATION of Johnson et al. 2013 (Nature Sci. Rep. 3:2627; tick rule: >= 10 ticks
in one direction within 1.5 s, >= 0.8 %) -> >= 0.8 % monotone move within <= 3 bars that reverts >= 50 % within 10
bars; non-overlapping, earliest onset wins, the largest move at an onset is the peak. Stated as an assumption in the
module docstring and `meta['mfc_rule']`.

Licence: Polygon / IEX per their terms (no redistribution of bars); TAQ licensed; halt notices public.

PIT: `avail_label` = reversion bar (the label is future-defined, L4); `tradable = False` inside halts + 5 min after and
inside circuit breakers (L7); `universe` as-of constituents (L6). Timestamp = bar start (Polygon); set
`timestamp_is: close` for exports that stamp the close.

---

## D9  `cg_wsb_squeeze`  — WSB posts + FINRA short interest + Yahoo prices   (`root: data_raw/wsb_squeeze`)

```
data_raw/wsb_squeeze/
  reddit_wsb.csv        Kaggle "Reddit WallStreetBets Posts": title, score, id, url, comms_num, created, body, timestamp
  short_interest.csv    FINRA export: settlementDate, symbolCode, currentShortPositionQuantity, ... (+ publication date
                        column if available; + float column if available)
  float.csv             symbol, float            (when short_interest.csv has no float column)
  prices/<TICKER>.csv   Yahoo daily; the tickers present define the asset list
  embeddings.npy  opt   float32 (n_posts, D) row-aligned with reddit_wsb.csv
```
1. Posts: `kaggle datasets download gpreda/reddit-wallstreetsbets-posts` (licence: **check the Kaggle page**; Reddit
   content terms apply). Pushshift dumps are an alternative; both are retro-collected snapshots (`score` / `comms_num`
   are crawl-time values, deleted posts are missing) -- recorded in `meta['pit_notes']`.
2. Short interest: FINRA equity short interest files (`https://www.finra.org/finra-data/browse-catalog/equity-short-interest`,
   public, bi-monthly settlement dates). FINRA disseminates about two weeks after settlement on a published schedule:
   export the dissemination dates into a `publication_date` column; otherwise the loader assumes settlement + 14
   calendar days (stated assumption). Column names are resolved by substring (`settlement`, `short`+`position`,
   `publication`/`dissemination`, `float`); the exact FINRA header names were not verified offline.
3. Float: a static per-symbol shares-float proxy (`float.csv`), e.g. Yahoo `floatShares`.
4. Prices: Yahoo daily csv per ticker (GME, AMC, BB, KOSS, EXPR, NOK, NAKD, TR, TRVG, BBBY, AAL, SNDL + controls).

Labels (constructed): settlement-dated short interest / float > 40 % AND trailing 5-day return > 50 %; peak = argmax
close over the next 10 bars; label on [onset - 5, peak]. Broker-restricted days 2021-01-28..2021-02-04 for the list
above are `tradable = False` (source: public reporting at the time -- Robinhood / IBKR / Webull notices -- TO BE
RE-VERIFIED before publication).

PIT: `avail_text` = first close at or after the post's creation time (default metadata text block excludes crawl-time score / comms_num); the short-interest feature column is joined on the
publication bar by default (`si_join: publication`); `si_join: settlement` reproduces the leaky convention and stamps
the row's `avail_feat` with the publication bar (whole-row stamp: the Panel has one `avail_feat` per row).
`meta['si_content_dated']` / `meta['si_avail']` always carry the settlement-dated series and its publication bars.

---

## D10  `cg_fin_graph`  — Elliptic++ transaction graph on hourly BTC bars   (`root: data_raw/fin_graph`)

```
data_raw/fin_graph/
  ellipticpp/txs_features.csv      txId, Time step, <features>   (a column containing 'btc' and 'total' is the BTC amount)
  ellipticpp/txs_classes.csv       txId, class (1 illicit, 2 licit, 3|unknown)
  ellipticpp/txs_edgelist.csv      txId1, txId2
  ellipticpp/tx_times.csv          txId, timestamp      -- user-constructed (NOT shipped with Elliptic++)
     or ellipticpp/timestep_calendar.csv   time_step, start, end  -- user-constructed step calendar
  ellipticpp/wallets_*.csv, *Addr*.csv     optional, not used by the loader
  prices/BTC.csv                   timestamp (bar OPEN, epoch s/ms or ISO UTC), open, high, low, close, volume
```
1. `git clone https://github.com/git-disl/EllipticPlusPlus` (Elmougy & Liu, KDD 2023) and follow its README for the
   Google-Drive csv release; licence per the repository LICENSE (check before redistribution).
2. Timestamps: Elliptic / Elliptic++ ship only the 1..49 "Time step" (steps ~2 weeks apart, each a connected component
   of transactions within < 3 hours), not block times. Provide either `tx_times.csv` (from a published txId -> tx-hash
   de-anonymisation mapping joined to block times via a node / explorer API) or `timestep_calendar.csv`. With the
   calendar, every transaction of a step is assigned to the bar of the step END (conservative; nothing inside the step
   is spread or interpolated). The loader refuses to invent a calendar.
3. Hourly BTC-USD bars from an exchange's public OHLCV history (Kraken / Bitstamp / Binance klines; timestamp = bar
   open). Multi-asset variant: add `prices/<ASSET>.csv` and list them under `assets:`; the transaction-graph snapshot is
   shared by every asset.

Labels (constructed): illicit share of class-known transactions in the bar > `illicit_share_thr` (0.05, assumption).
`bars_to_peak` = realised |impact| argmax. `label_lag_bars` (default 168 = one week) is the paper's sensitivity axis
(`pit.label_lag.grid` in the yaml: 0, 24, 168, 720).

PIT: `avail_label = t + label_lag_bars` for every row; `avail_graph` slot 0 (structural snapshot) = t, slot 1
(label-derived snapshot: illicit share, illicit-neighbour share, ...) = t + label_lag_bars; `universe` = has a bar.

---

Format fixtures used by `tests/causalgate_parts/test_loaders_constructed.py` are written under the session scratchpad
(`cg_format_fixtures/<name>/`, each with a `FORMAT_FIXTURE_NOT_DATA.txt` marker); they are placeholder rows in the raw
layouts above and must never be mistaken for data.
