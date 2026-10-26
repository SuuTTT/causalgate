"""The ten benchmark builders. Import registers them in warn2trade.utils.registry.DATASETS.

Status legend (see docs/benchmark_audit.md for sources):
    PUBLIC      freely downloadable, documented loader path
    LICENSED    academic licence / paid vendor data (LOBSTER, TAQ, Audit Analytics)
    CONSTRUCTED assembled from public APIs with the recipe in the builder docstring
    PLACEHOLDER no verifiable public dataset by that name; a recommended substitute is wired instead
"""
from . import stocknet, lobster, fnspid, edgar, crypto_pd, csi300_limit, sp500_tick, qlib_alpha158, wsb_squeeze, fin_graph  # noqa: F401
