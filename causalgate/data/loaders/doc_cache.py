"""Offline document-embedding cache contract + point-in-time embedders + Panel.text slot placement.

Panel.text needs *pre-computed* embeddings. The loaders never embed at build time: they read

    <root>/cache/<dataset>_docs.npz
        emb             (n_docs, D) float32   document embeddings
        doc_asset       (n_docs,)   str       asset key exactly as the loader names its assets (ticker / pump key)
        doc_publish_ns  (n_docs,)   int64     UTC ns publication time = availability time a(x)
        doc_content_ns  (n_docs,)   int64     UTC ns midnight of the dataset's own date key (the naive-join key: the
                                              StockNet tweet directory date, the FNSPID `Date` column date, ...)
        doc_id          (n_docs,)   str       optional free-form identifier
        meta            str (json)  {dataset, embedder, cutoff_date, embedder_training_cutoff, dim, n_docs, n_fit_docs,
                                     created_utc, checkpoint, notes}

and build text-free panels (text=None, meta['text'] = 'missing cache') when the file is absent. The cache is written
by scripts/cg_prepare_data.py --embed.

Embedders (all return float32 rows; the cache meta records the *training cutoff* of every component so that the
L8 audit can compare it with the test-fold start):
    hash_tfidf   HashingVectorizer -> TfidfTransformer -> TruncatedSVD, every fitted component (idf, SVD basis) fitted ONLY
                 on documents published before --cutoff; later documents are transformed with the frozen operator.
                 No external model, no vocabulary file, deterministic (seeded). Default.
    finbert      hook: ProsusAI/finbert (Araci 2019) via transformers. Training cutoff not published by the authors ->
                 recorded as 'unknown (declared 2019-12-31 upper bound)'.
    chronobert   hook: ChronoBERT (He, Lv, Manela, Wu 2025, arXiv:2502.21206), monthly-cutoff checkpoints; the cutoff is
                 read from the checkpoint name (YYYYMM / YYYYMMDD) or must be given explicitly.
    pit_lm       hook: point-in-time LMs (Kelly et al. 2026, arXiv:2607.11889), monthly checkpoints; cutoff as above.
    The three transformer hooks raise ImportError with the install line when `transformers` is missing and NEVER write a
    cache they did not compute.

Slot placement convention (shared by D1 / D3 / D5): a document is placed ONCE, at the row of its *content date*
(the naive join key), with avail_text = the bar at which it became knowable (first bar close >= publication time). Hence
avail_text >= row always, avail_text > row for an after-close document, and the L5 "label-time join" leak is exactly
"use the row instead of avail_text". Downstream window encoders that want document ageing read avail_text; text_age is 0.
"""
from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from ..pit import NEVER
from .common import next_bar_index

CACHE_VERSION = 1
EMBEDDERS = ("hash_tfidf", "finbert", "chronobert", "pit_lm")


# =========================================================================================== cache file
@dataclass
class DocCache:
    emb: np.ndarray                      # (n, D) float32
    doc_asset: np.ndarray                # (n,) str
    doc_publish_ns: np.ndarray           # (n,) int64
    doc_content_ns: np.ndarray           # (n,) int64
    doc_id: np.ndarray                   # (n,) str
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.emb.shape[0])

    @property
    def dim(self) -> int:
        return int(self.emb.shape[1])


def cache_path(root: str, dataset: str) -> str:
    return os.path.join(root, "cache", f"{dataset}_docs.npz")


def read_doc_cache(path: str) -> Optional[DocCache]:
    """Return the cache or None when the file does not exist. A malformed file raises (never silently ignored)."""
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        for k in ("emb", "doc_asset", "doc_publish_ns", "meta"):
            if k not in keys:
                raise ValueError(f"{path}: missing key {k!r}; keys = {sorted(keys)}")
        emb = np.asarray(z["emb"], dtype=np.float32)
        asset = np.asarray(z["doc_asset"]).astype(str)
        pub = np.asarray(z["doc_publish_ns"], dtype=np.int64)
        content = np.asarray(z["doc_content_ns"], dtype=np.int64) if "doc_content_ns" in keys else pub.copy()
        doc_id = np.asarray(z["doc_id"]).astype(str) if "doc_id" in keys else np.array([str(i) for i in range(len(pub))])
        meta = json.loads(str(z["meta"]))
    if not (emb.shape[0] == asset.shape[0] == pub.shape[0] == content.shape[0]):
        raise ValueError(f"{path}: inconsistent lengths emb={emb.shape} asset={asset.shape} pub={pub.shape}")
    return DocCache(emb, asset, pub, content, doc_id, meta)


def write_doc_cache(path: str, emb: np.ndarray, doc_asset: Sequence[str], doc_publish_ns: np.ndarray, meta: Dict[str, Any],
                    doc_content_ns: Optional[np.ndarray] = None, doc_id: Optional[Sequence[str]] = None) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    emb = np.asarray(emb, dtype=np.float32)
    pub = np.asarray(doc_publish_ns, dtype=np.int64)
    content = pub.copy() if doc_content_ns is None else np.asarray(doc_content_ns, dtype=np.int64)
    ids = np.array([str(i) for i in range(len(pub))]) if doc_id is None else np.asarray(list(doc_id)).astype(str)
    meta = dict(meta)
    meta.setdefault("cache_version", CACHE_VERSION)
    meta.setdefault("created_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    meta["n_docs"] = int(emb.shape[0])
    meta["dim"] = int(emb.shape[1]) if emb.ndim == 2 else 0
    np.savez_compressed(path, emb=emb, doc_asset=np.asarray(list(doc_asset)).astype(str), doc_publish_ns=pub,
                        doc_content_ns=content, doc_id=ids, meta=np.array(json.dumps(meta)))
    return path


# =========================================================================================== embedders
class HashTfidfSVD:
    """Hashing vectoriser -> tf-idf -> truncated SVD. `fit` must only see documents published before the cutoff."""

    def __init__(self, dim: int = 32, n_features: int = 2 ** 18, seed: int = 0, ngram_range: Tuple[int, int] = (1, 2)):
        from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
        self.dim, self.n_features, self.seed = int(dim), int(n_features), int(seed)
        self.hv = HashingVectorizer(n_features=self.n_features, alternate_sign=False, norm=None, ngram_range=ngram_range)
        self.tfidf = TfidfTransformer(sublinear_tf=True)
        self.svd = None
        self.n_fit_docs = 0
        self.name = "hash_tfidf"

    def fit(self, texts: Sequence[str]) -> "HashTfidfSVD":
        from sklearn.decomposition import TruncatedSVD
        texts = [str(t) for t in texts]
        if len(texts) < 2:
            raise ValueError(f"hash_tfidf needs >= 2 documents before the cutoff to fit idf/SVD, got {len(texts)}")
        X = self.tfidf.fit_transform(self.hv.transform(texts))
        k = max(1, min(self.dim, len(texts) - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=self.seed, algorithm="randomized", n_iter=7)
        self.svd.fit(X)
        self.n_fit_docs = len(texts)
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self.svd is None:
            raise RuntimeError("fit before transform")
        X = self.tfidf.transform(self.hv.transform([str(t) for t in texts]))
        Z = self.svd.transform(X).astype(np.float32)
        nrm = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z / np.where(nrm > 0, nrm, 1.0)

    def training_cutoff(self, cutoff_date: str) -> str:
        return cutoff_date                      # every fitted component saw only documents published before the cutoff

    def describe(self) -> Dict[str, Any]:
        return {"embedder": self.name, "n_features": self.n_features, "dim": int(self.svd.n_components) if self.svd else self.dim,
                "n_fit_docs": self.n_fit_docs, "seed": self.seed, "pit": "idf and SVD basis fitted on pre-cutoff documents only"}


_TRANSFORMER_HOOKS: Dict[str, Dict[str, str]] = {
    "finbert": {"default_checkpoint": "ProsusAI/finbert", "paper": "Araci 2019, arXiv:1908.10063",
                "cutoff_rule": "not published by the authors; recorded as 'unknown (declared 2019-12-31 upper bound)'"},
    "chronobert": {"default_checkpoint": "", "paper": "He, Lv, Manela, Wu 2025, arXiv:2502.21206",
                   "cutoff_rule": "monthly checkpoints; cutoff parsed from the checkpoint name (YYYYMM or YYYYMMDD) or given via --checkpoint-cutoff"},
    "pit_lm": {"default_checkpoint": "", "paper": "Kelly et al. 2026, arXiv:2607.11889",
               "cutoff_rule": "monthly checkpoints; cutoff parsed from the checkpoint name (YYYYMM or YYYYMMDD) or given via --checkpoint-cutoff"},
}


def _cutoff_from_checkpoint(name: str) -> Optional[str]:
    m = re.search(r"(19|20)(\d{2})(\d{2})(\d{2})?", name or "")
    if not m:
        return None
    y, mo, d = m.group(1) + m.group(2), m.group(3), m.group(4)
    return f"{y}-{mo}-{d}" if d else f"{y}-{mo}-01"


class TransformerEmbedder:
    """Documented hook for FinBERT / ChronoBERT / PIT-LM mean-pooled embeddings. Construction raises ImportError when
    `transformers` is unavailable, so a cache can never be claimed without the model having run."""

    def __init__(self, name: str, checkpoint: Optional[str] = None, checkpoint_cutoff: Optional[str] = None,
                 max_length: int = 256, batch_size: int = 32, device: str = "cpu"):
        if name not in _TRANSFORMER_HOOKS:
            raise KeyError(f"unknown transformer hook {name!r}; choose from {sorted(_TRANSFORMER_HOOKS)}")
        self.name = name
        self.spec = _TRANSFORMER_HOOKS[name]
        self.checkpoint = checkpoint or self.spec["default_checkpoint"]
        if not self.checkpoint:
            raise ValueError(f"{name}: pass --checkpoint <hf-id or local dir> ({self.spec['paper']}); no default is assumed")
        if name == "finbert":
            self._cutoff = checkpoint_cutoff or "unknown (declared 2019-12-31 upper bound)"
        else:
            self._cutoff = checkpoint_cutoff or _cutoff_from_checkpoint(self.checkpoint)
            if self._cutoff is None:
                raise ValueError(f"{name}: cannot read the training cutoff from {self.checkpoint!r}; pass --checkpoint-cutoff YYYY-MM-DD")
        self.max_length, self.batch_size, self.device = int(max_length), int(batch_size), device
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
        except ImportError as e:                      # pragma: no cover - depends on the environment
            raise ImportError(f"{name} embedder needs `pip install transformers` (and torch); not installed: {e}. "
                              f"No cache was written. Use --embedder hash_tfidf for the model-free PIT embedding.") from e
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModel.from_pretrained(self.checkpoint).to(device).eval()

    def fit(self, texts: Sequence[str]) -> "TransformerEmbedder":
        return self                                   # frozen checkpoint: nothing is fitted on the corpus

    def transform(self, texts: Sequence[str]) -> np.ndarray:   # pragma: no cover - needs transformers
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                b = self.tok([str(t) for t in texts[i : i + self.batch_size]], padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt").to(self.device)
                h = self.model(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
                out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy().astype(np.float32))
        Z = np.concatenate(out, 0) if out else np.zeros((0, self.model.config.hidden_size), np.float32)
        nrm = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z / np.where(nrm > 0, nrm, 1.0)

    def training_cutoff(self, cutoff_date: str) -> str:
        return self._cutoff

    def describe(self) -> Dict[str, Any]:
        return {"embedder": self.name, "checkpoint": self.checkpoint, "paper": self.spec["paper"],
                "cutoff_rule": self.spec["cutoff_rule"], "pooling": "masked mean of last hidden state, L2-normalised"}


def make_embedder(name: str, dim: int = 32, seed: int = 0, checkpoint: Optional[str] = None,
                  checkpoint_cutoff: Optional[str] = None, device: str = "cpu"):
    if name == "hash_tfidf":
        return HashTfidfSVD(dim=dim, seed=seed)
    if name in _TRANSFORMER_HOOKS:
        return TransformerEmbedder(name, checkpoint, checkpoint_cutoff, device=device)
    raise KeyError(f"unknown embedder {name!r}; choose from {EMBEDDERS}")


def build_doc_cache(path: str, docs: pd.DataFrame, dataset: str, cutoff_date: str, embedder: str = "hash_tfidf",
                    dim: int = 32, seed: int = 0, checkpoint: Optional[str] = None, checkpoint_cutoff: Optional[str] = None,
                    notes: str = "", device: str = "cpu") -> Dict[str, Any]:
    """docs columns: asset (str), publish_ns (int64 UTC), content_ns (int64 UTC, optional), text (str), doc_id (optional).
    Fits the embedder on documents with publish_ns < cutoff (hash_tfidf) and writes the cache. Returns the meta dict."""
    need = {"asset", "publish_ns", "text"}
    if not need <= set(docs.columns):
        raise ValueError(f"docs needs columns {sorted(need)}, got {list(docs.columns)}")
    if len(docs) == 0:
        raise ValueError(f"{dataset}: no documents to embed; no cache written")
    cutoff_ns = int(pd.Timestamp(cutoff_date, tz="UTC").value)
    pub = docs["publish_ns"].to_numpy(dtype=np.int64)
    before = pub < cutoff_ns
    emb = make_embedder(embedder, dim=dim, seed=seed, checkpoint=checkpoint, checkpoint_cutoff=checkpoint_cutoff, device=device)
    texts = docs["text"].astype(str).tolist()
    if embedder == "hash_tfidf":
        if before.sum() < 2:
            raise ValueError(f"{dataset}: only {int(before.sum())} documents published before cutoff {cutoff_date}; "
                             f"cannot fit a point-in-time embedder")
        emb.fit([t for t, b in zip(texts, before) if b])
    Z = emb.transform(texts)
    n_zero = int((np.linalg.norm(Z, axis=1) == 0).sum())      # no overlap with the pre-cutoff vocabulary -> zero vector (kept, reported)
    meta = {"dataset": dataset, "cutoff_date": cutoff_date, "n_docs_before_cutoff": int(before.sum()), "n_zero_embeddings": n_zero,
            "embedder_training_cutoff": emb.training_cutoff(cutoff_date), "notes": notes, **emb.describe()}
    content = docs["content_ns"].to_numpy(dtype=np.int64) if "content_ns" in docs else None
    ids = docs["doc_id"].astype(str).tolist() if "doc_id" in docs else None
    write_doc_cache(path, Z, docs["asset"].astype(str).tolist(), pub, meta, doc_content_ns=content, doc_id=ids)
    return meta


# =========================================================================================== placement
def utc_date_ns(ns: np.ndarray) -> np.ndarray:
    """int64 UTC ns -> int64 ns of the UTC midnight of the same date."""
    return (np.asarray(ns, dtype=np.int64) // (86_400 * 10 ** 9)) * (86_400 * 10 ** 9)


def dates_to_ns(dates) -> np.ndarray:
    """Calendar dates (tz-naive) -> int64 ns of UTC midnight (the convention used for doc_content_ns)."""
    return pd.to_datetime(pd.Series(dates)).dt.tz_localize(None).dt.normalize().dt.tz_localize("UTC").values.astype("datetime64[ns]").astype(np.int64)


def daily_rows_and_avail(cache: DocCache, assets: Sequence[str], bar_dates, bar_close_ns: np.ndarray) -> Dict[str, np.ndarray]:
    """Content-date rows and availability bars for a daily panel.

    row   = first bar whose session date >= the document's content date (naive calendar join)
    avail = first bar whose close time >= the publication time (after-close documents -> next session), clipped to
            >= row; NEVER when no bar in the panel is late enough.
    asset = column index (-1 when the asset is not in the panel)."""
    T = len(bar_close_ns)
    date_ns = dates_to_ns(bar_dates)
    row = np.searchsorted(date_ns, utc_date_ns(cache.doc_content_ns), side="left")
    avail = next_bar_index(bar_close_ns, cache.doc_publish_ns)
    avail = np.maximum(avail, row)
    avail = np.where(avail >= T, NEVER, avail)
    idx = {a: i for i, a in enumerate(assets)}
    asset = np.array([idx.get(str(a), -1) for a in cache.doc_asset], dtype=np.int64)
    return {"row": row.astype(np.int64), "avail": avail.astype(np.int64), "asset": asset}


def fill_text_slots(T: int, N: int, M: int, emb: np.ndarray, rows: np.ndarray, avail: np.ndarray, asset: np.ndarray,
                    order_key: np.ndarray) -> Dict[str, Any]:
    """Place documents into (T, N, M, D) slots at `rows`. Per (row, asset) the M slots are chosen by AVAILABILITY
    FIRST: documents already knowable at the row (avail <= row) are kept before after-close ones (avail > row), and
    only then by largest `order_key` (publication time). So after-close documents can never evict a document that
    was visible at the row's own close, and the causal view of row t does not depend on post-close arrivals.
    Empty slots get avail_text = row (the conservative default)."""
    D = int(emb.shape[1])
    text = np.zeros((T, N, M, D), dtype=np.float32)
    mask = np.zeros((T, N, M), dtype=bool)
    age = np.zeros((T, N, M), dtype=np.float32)
    av = np.broadcast_to(np.arange(T, dtype=np.int64)[:, None, None], (T, N, M)).copy()
    ok = (rows >= 0) & (rows < T) & (asset >= 0) & (asset < N)
    stats = {"n_docs": int(len(rows)), "placed": 0, "dropped_out_of_panel": int((~ok).sum()), "dropped_overflow": 0}
    late = (np.asarray(avail, dtype=np.int64) > np.asarray(rows, dtype=np.int64)).astype(np.int64)
    order = np.lexsort((-np.asarray(order_key, dtype=np.int64), late, asset, rows))  # (row, asset): knowable first, then latest
    fill = np.zeros((T, N), dtype=np.int64)
    for j in order:
        if not ok[j]:
            continue
        r, i = int(rows[j]), int(asset[j])
        s = fill[r, i]
        if s >= M:
            stats["dropped_overflow"] += 1
            continue
        text[r, i, s] = emb[j]
        mask[r, i, s] = True
        av[r, i, s] = int(avail[j])
        fill[r, i] += 1
        stats["placed"] += 1
    return {"text": text, "text_mask": mask, "text_age": age, "avail_text": av, "stats": stats}


def text_fields_from_cache(cache: Optional[DocCache], assets: Sequence[str], bar_dates, bar_close_ns: np.ndarray, M: int,
                           meta: Dict[str, Any], label: str) -> Dict[str, Any]:
    """Daily-panel glue: cache -> (text, text_mask, text_age, avail_text) or text=None with meta['text'] = 'missing cache'."""
    T, N = len(bar_close_ns), len(assets)
    if cache is None:
        meta["text"] = "missing cache"
        meta["text_recipe"] = f"python3 scripts/cg_prepare_data.py --dataset {label} --embed --embedder hash_tfidf --cutoff <YYYY-MM-DD>"
        return {"text": None, "text_mask": None, "text_age": None, "avail_text": None}
    pl = daily_rows_and_avail(cache, assets, bar_dates, bar_close_ns)
    out = fill_text_slots(T, N, M, cache.emb, pl["row"], pl["avail"], pl["asset"], cache.doc_publish_ns)
    meta["text"] = dict(cache.meta)
    meta["text_placement"] = out.pop("stats")
    meta["text_placement"]["n_after_close"] = int(((pl["avail"] > pl["row"]) & (pl["asset"] >= 0) & (pl["row"] < T)).sum())
    return out
