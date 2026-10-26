"""SEC enforcement anchors (US equities). STATUS: PUBLIC source, parsing + ticker/date matching to implement.

Sources
    Litigation Releases   https://www.sec.gov/litigation/litreleases   (one HTML/press page per release; links to the complaint PDF)
    Admin. Proceedings    https://www.sec.gov/litigation/admin
    Trading Suspensions   https://www.sec.gov/litigation/suspensions   (give exact suspension dates -> release_t at day resolution)
    EDGAR full-text       https://efts.sec.gov/LATEST/search-index?q=... (to resolve issuer names -> CIK -> ticker)
Case selection for the benchmark: manipulation (pump-and-dump, matched trades, spoofing/layering), insider trading
around M&A / earnings. Complaint text gives the conduct window ("between approximately March 2019 and July 2019") ->
date_resolution = "month"; use the 1st / last trading day of the stated months for onset_t / end_t.

Pipeline (scripts/build_sec_anchors.py, to write):
    1. crawl release index pages for YEAR in range -> (lr_number, release_date, title, url)
    2. download complaint PDF -> text; regex for tickers ($XXXX, "ticker symbol XXXX"), conduct windows, scheme type
    3. manual verification sheet (one row per case) -> anomaly_class, onset/end dates, instruments
    4. map dates to bar indices of the Panel with anchors.dates_to_bar_index; release_t = release_date (side='left')
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

TICKER_RE = re.compile(r"(?:ticker(?: symbol)?|symbol|NASDAQ:|NYSE:|OTC:)\s*[\"“']?([A-Z]{1,5})\b")
WINDOW_RE = re.compile(r"(?:between|from)\s+(?:approximately\s+|at least\s+)?([A-Z][a-z]+ \d{4})\s+(?:and|to|through)\s+([A-Z][a-z]+ \d{4})")
SCHEME_KEYWORDS = {"pump_dump": ["pump-and-dump", "pump and dump", "promotional campaign"], "spoofing": ["spoofing", "layering"],
                   "insider": ["insider trading", "material nonpublic"], "wash_trading": ["wash trades", "matched orders", "matched trades"],
                   "accounting": ["restatement", "accounting fraud", "improperly recognized revenue"]}


def classify_scheme(text: str) -> Optional[str]:
    t = text.lower()
    for cls, kws in SCHEME_KEYWORDS.items():
        if any(k in t for k in kws):
            return cls
    return None


def parse_complaint(text: str) -> Dict[str, object]:
    """Best-effort extraction; every row must still be manually verified before it becomes an anchor."""
    tickers = sorted(set(TICKER_RE.findall(text)))
    w = WINDOW_RE.search(text)
    return {"tickers": tickers, "window_start": w.group(1) if w else None, "window_end": w.group(2) if w else None,
            "anomaly_class": classify_scheme(text), "date_resolution": "month" if w else "unknown"}


def fetch_release_index(years: List[int]) -> pd.DataFrame:  # pragma: no cover - network
    raise NotImplementedError("Crawl https://www.sec.gov/litigation/litreleases/<year> (respect sec.gov rate limits and set a "
                              "descriptive User-Agent) -> DataFrame(lr_number, release_date, title, url).")


def build_anchor_table(verified_sheet: pd.DataFrame, panel_times, asset_map: Dict[str, str]) -> pd.DataFrame:
    """Deprecated thin wrapper; use warn2trade.data.regulatory.sheet.sheet_to_anchors, which also returns the match report."""
    from .sheet import sheet_to_anchors
    sh = verified_sheet.assign(source=verified_sheet.get("source", "SEC"))
    table, _ = sheet_to_anchors(sh, np.asarray(panel_times), list(set(asset_map.values())), asset_map)
    return table
