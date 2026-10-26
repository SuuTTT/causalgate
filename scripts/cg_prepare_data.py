#!/usr/bin/env python3
"""CausalGate raw-data preparation: print / run the download recipe and build the offline document-embedding cache.

    python3 scripts/cg_prepare_data.py --dataset stocknet --root data_raw                 # print the recipe + check files
    python3 scripts/cg_prepare_data.py --dataset stocknet --root data_raw --download      # run the shell recipe
    python3 scripts/cg_prepare_data.py --dataset stocknet --root data_raw --embed --embedder hash_tfidf --cutoff 2015-01-01
    python3 scripts/cg_prepare_data.py --dataset all --root data_raw

--embed writes <root>/cache/<dataset>_docs.npz (contract in causalgate/data/loaders/doc_cache.py). The default embedder
hash_tfidf fits its idf / SVD basis ONLY on documents published before --cutoff (point-in-time safe, no external model).
finbert / chronobert / pit_lm are hooks that need `transformers`; they raise a clear ImportError otherwise and record the
checkpoint's training cutoff in the cache meta (FinBERT: unknown / 2019 upper bound; ChronoBERT arXiv:2502.21206 and
PIT-LM arXiv:2607.11889: monthly cutoffs parsed from the checkpoint name or --checkpoint-cutoff). Nothing is downloaded
unless --download is given, and a cache is written only after the embeddings were actually computed.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from causalgate.data.loaders import stocknet, fi2010, fnspid, crypto_pd, qlib_alpha158
from causalgate.data.loaders import edgar, csi300_limit, sp500_tick, wsb_squeeze, fin_graph
from causalgate.data.loaders.doc_cache import EMBEDDERS, build_doc_cache, cache_path, read_doc_cache

RECIPES_MD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "causalgate", "data", "loaders", "RECIPES_D4_D10.md")

DATASETS = {
    "stocknet": {"module": stocknet, "docs": stocknet.iter_documents, "cfg": {"subdir": "stocknet"}},
    "fi2010": {"module": fi2010, "docs": None, "cfg": {"subdir": "fi2010"}},
    "fnspid": {"module": fnspid, "docs": fnspid.iter_documents, "cfg": {"subdir": "fnspid"}},
    "crypto_pd": {"module": crypto_pd, "docs": crypto_pd.iter_documents, "cfg": {"subdir": "crypto_pd"}},
    "qlib_alpha158": {"module": qlib_alpha158, "docs": None, "cfg": {"subdir": "qlib_us"}},
    # D4 / D6 / D7 / D9 / D10: manual recipes (licensed or constructed sources) documented in RECIPES_D4_D10.md; the
    # loaders raise FileNotFoundError with the same recipe. Document embeddings for D4 / D9 are user-produced
    # (embeddings.npy, see the recipe); --embed is therefore not offered for them here.
    "edgar": {"module": edgar, "docs": None, "cfg": {"subdir": "edgar"}, "md": "D4"},
    "csi300_limit": {"module": csi300_limit, "docs": None, "cfg": {"subdir": "csi300_limit"}, "md": "D6"},
    "sp500_tick": {"module": sp500_tick, "docs": None, "cfg": {"subdir": "sp500_tick"}, "md": "D7"},
    "wsb_squeeze": {"module": wsb_squeeze, "docs": None, "cfg": {"subdir": "wsb_squeeze"}, "md": "D9"},
    "fin_graph": {"module": fin_graph, "docs": None, "cfg": {"subdir": "fin_graph"}, "md": "D10"},
}


def _md_section(tag: str) -> str:
    """Extract the '## D<k>' section of RECIPES_D4_D10.md (manual recipe; nothing runnable is emitted)."""
    if not os.path.exists(RECIPES_MD):
        return f"# see causalgate/data/loaders/RECIPES_D4_D10.md ({tag})"
    txt = open(RECIPES_MD).read()
    parts = txt.split("\n## ")
    for part in parts:
        if part.startswith(tag + " ") or part.startswith(tag + "\t"):
            return "# manual recipe (not executed by --download):\n" + "\n".join("# " + ln for ln in part.splitlines())
    return f"# section {tag} not found in RECIPES_D4_D10.md"


def _attr(mod, *names, default="see RECIPES_D4_D10.md"):
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    return default


def log(msg: str) -> None:
    print(f"[cg_prepare_data] {msg}", flush=True)


def recipe_for(name: str, root: str) -> str:
    mod = DATASETS[name]["module"]
    rec = getattr(mod, "RECIPE", None)
    if rec:
        return str(rec).replace("{root}", root)          # plain replace: recipes may contain literal braces
    return _md_section(DATASETS[name]["md"])


def check_files(name: str, root: str) -> bool:
    """Instantiate the builder on `root` and report whether the raw files are present (never builds the full panel)."""
    mod = DATASETS[name]["module"]
    try:
        from warn2trade.utils import DATASETS as REG
        builder = REG.get(f"cg_{name}")({"name": f"cg_{name}", "root": root})
    except Exception as e:  # pragma: no cover
        log(f"{name}: cannot instantiate builder: {e}")
        return False
    sub = DATASETS[name]["cfg"]["subdir"]
    present = os.path.isdir(os.path.join(root, sub))
    log(f"{name}: raw directory {os.path.join(root, sub)} {'present' if present else 'MISSING'}; licence: {_attr(mod, 'LICENCE', 'LICENSE', 'LICENCE_NOTE')}")
    return present


def collect_documents(name: str, root: str, cfg: dict, limit: int = 0) -> pd.DataFrame:
    it = DATASETS[name]["docs"]
    if it is None:
        raise SystemExit(f"{name}: no text modality (no documents to embed)")
    rows = []
    t0 = time.time()
    for k, d in enumerate(it(root, cfg)):
        rows.append(d)
        if limit and len(rows) >= limit:
            break
        if (k + 1) % 100_000 == 0:
            log(f"{name}: {k + 1} documents read ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows, columns=["asset", "content_ns", "publish_ns", "text", "doc_id"])


def embed(name: str, args: argparse.Namespace) -> None:
    cfg = dict(DATASETS[name]["cfg"])
    if args.tickers:
        cfg["tickers"] = args.tickers.split(",")
    docs = collect_documents(name, args.root, cfg, args.limit)
    if len(docs) == 0:
        extra = " (La Morgia's public table has no message text: add messages.csv)" if name == "crypto_pd" else ""
        log(f"{name}: 0 documents found under {args.root}{extra}; no cache written")
        return
    path = args.cache or cache_path(args.root, name)
    log(f"{name}: {len(docs)} documents, {docs['asset'].nunique()} assets, "
        f"{pd.to_datetime(docs['publish_ns'].min(), utc=True).date()} .. {pd.to_datetime(docs['publish_ns'].max(), utc=True).date()}")
    meta = build_doc_cache(path, docs, name, args.cutoff, embedder=args.embedder, dim=args.dim, seed=args.seed,
                           checkpoint=args.checkpoint, checkpoint_cutoff=args.checkpoint_cutoff, device=args.device,
                           notes=f"built by scripts/cg_prepare_data.py; tickers={args.tickers or 'all'}; limit={args.limit or 'none'}")
    log(f"{name}: wrote {path}: {meta}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS) + ["all"])
    p.add_argument("--root", default="data_raw")
    p.add_argument("--download", action="store_true", help="run the shell recipe (otherwise it is only printed)")
    p.add_argument("--embed", action="store_true", help="build <root>/cache/<dataset>_docs.npz")
    p.add_argument("--embedder", default="hash_tfidf", choices=list(EMBEDDERS))
    p.add_argument("--cutoff", default=None, help="YYYY-MM-DD: the embedder is fitted on documents published before this date (required with --embed)")
    p.add_argument("--dim", type=int, default=32, help="hash_tfidf SVD dimension")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default=None, help="HF id / local dir for finbert | chronobert | pit_lm")
    p.add_argument("--checkpoint-cutoff", default=None, help="YYYY-MM-DD training cutoff of the checkpoint when it cannot be parsed from its name")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tickers", default=None, help="comma-separated asset subset for --embed")
    p.add_argument("--limit", type=int, default=0, help="stop after this many documents (smoke runs)")
    p.add_argument("--cache", default=None, help="override the cache path")
    args = p.parse_args()
    names = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        rec = recipe_for(name, args.root)
        log(f"=== {name}: source {_attr(DATASETS[name]['module'], 'SOURCE_URL', 'SOURCE', 'SOURCES')}")
        log(f"recipe:\n{rec}")
        if args.download:
            cmds = "\n".join(ln for ln in rec.splitlines() if ln.strip() and not ln.strip().startswith("#"))
            if not cmds.strip():
                log(f"{name}: the recipe is manual (licensed / registration download); nothing to run")
            else:
                log(f"{name}: running recipe")
                subprocess.run(["bash", "-euxo", "pipefail", "-c", cmds], check=True)
        present = check_files(name, args.root)
        if args.embed:
            if DATASETS[name]["docs"] is None:
                log(f"{name}: no text modality; --embed skipped")
                continue
            if not present:
                log(f"{name}: raw files missing; --embed skipped (run with --download or place the files first)")
                continue
            if not args.cutoff:
                raise SystemExit("--embed needs --cutoff YYYY-MM-DD (point-in-time fit boundary)")
            embed(name, args)
        else:
            cp = cache_path(args.root, name)
            c = read_doc_cache(cp) if os.path.exists(cp) else None
            log(f"{name}: cache {cp} {'present: ' + str(c.meta) if c else 'absent (text=None until --embed)'}")


if __name__ == "__main__":
    main()
