"""CSRC administrative sanction anchors (China A-shares). STATUS: PUBLIC source, parsing to implement.

Source: 中国证监会 行政处罚决定书 http://www.csrc.gov.cn/csrc/c101928/zfxxgk_zdgk.shtml (and the regional bureaus' pages).
Decisions typically state the stock code (6 digits), the exact conduct dates (e.g. "2019年3月4日至2019年5月20日"), the
scheme (操纵证券市场 / 内幕交易 / 信息披露违法) and are published 1-3 years after the conduct -> day-resolution anchors
with a long enforcement lag, i.e. the cleanest real instance of the release-date-aware protocol.

Pipeline (scripts/build_csrc_anchors.py, to write): crawl decision list -> download decision text -> regex codes /
date ranges / scheme keywords -> manual verification sheet -> anchors CSV (same schema as SEC).
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional
import pandas as pd

CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
DATE_RANGE_RE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日\s*(?:至|到|—|-)\s*(\d{4})年(\d{1,2})月(\d{1,2})日")
SCHEME_KEYWORDS = {"pump_dump": ["操纵证券市场", "操纵", "连续买卖", "拉抬"], "insider": ["内幕交易", "内幕信息"],
                   "accounting": ["信息披露违法", "虚增", "财务造假"], "spoofing": ["虚假申报", "频繁申报撤单"]}


def classify_scheme(text: str) -> Optional[str]:
    for cls, kws in SCHEME_KEYWORDS.items():
        if any(k in text for k in kws):
            return cls
    return None


def parse_decision(text: str) -> Dict[str, object]:
    codes = sorted(set(c for c in CODE_RE.findall(text) if c[0] in "036"))     # SH 60xxxx / SZ 00xxxx 30xxxx
    m = DATE_RANGE_RE.search(text)
    rng = None
    if m:
        y1, m1, d1, y2, m2, d2 = map(int, m.groups())
        rng = (f"{y1:04d}-{m1:02d}-{d1:02d}", f"{y2:04d}-{m2:02d}-{d2:02d}")
    return {"codes": codes, "window_start": rng[0] if rng else None, "window_end": rng[1] if rng else None,
            "anomaly_class": classify_scheme(text), "date_resolution": "day" if rng else "unknown"}


def fetch_decision_index(years: List[int]) -> pd.DataFrame:  # pragma: no cover - network
    raise NotImplementedError("Crawl the CSRC decision list pages -> DataFrame(decision_no, publish_date, title, url).")


def build_anchor_table(verified_sheet: pd.DataFrame, panel_times, asset_map: Dict[str, str]) -> pd.DataFrame:
    """Deprecated thin wrapper; use warn2trade.data.regulatory.sheet.sheet_to_anchors (codes go in the `ticker` column)."""
    from .sheet import sheet_to_anchors
    import numpy as np
    sh = verified_sheet.rename(columns={"code": "ticker"}).assign(source="CSRC")
    table, _ = sheet_to_anchors(sh, np.asarray(panel_times), list(set(asset_map.values())), asset_map)
    return table
