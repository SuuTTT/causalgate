# Third-party official code for baselines B2-B7

No official repository is vendored: the repositories could not be cloned in the build session (individual source files were only read through the GitHub raw endpoints on 2026-10-03 to record the `~L` pointers in CONVENTIONS.md). Each wrapper looks for the official code
under `third_party/<name>/` (repository root), imports the official *model module* when it is importable, and otherwise
uses its own re-implementation and reports it (`bl.status`, `bl.diag['official_import']`). The training loop, the scoring
rule and the convention handling are always ours (the official `Solver` / benchmark classes read datasets from disk and
cannot be driven by the event loaders); "official" therefore means *official model module + replicated solver logic*.

```
mkdir -p third_party && cd third_party
git clone https://github.com/thuml/Anomaly-Transformer.git            Anomaly-Transformer     # B2 (MIT; check LICENSE)
git clone https://github.com/sihohan/DAMP.git                          DAMP                    # B3 python port (no licence file shown; ask the authors)
git clone https://github.com/Stream-AD/MemStream.git                   MemStream               # B4 (LICENSE file in the repo; check)
git clone https://github.com/DAMO-DI-ML/KDD2023-DCdetector.git         KDD2023-DCdetector      # B5 (check LICENSE)
pip install momentfm                                                                           # B6 (MIT); weights: HF AutonLab/MOMENT-1-large
git clone https://github.com/decisionintelligence/CATCH.git            CATCH                   # B7 (branch master; licence not shown on the landing page)
```

The authors' DAMP reference implementation is MATLAB (`DAMP_2_0.m`, https://sites.google.com/view/discord-aware-matrix-profile);
the python port above is unofficial and is cited as such in `conventions_b2_b7.md`.

| Baseline | Wrapper | Detection | `status` when found | Caveats |
|---|---|---|---|---|
| B2 Anomaly Transformer | `b2_anomaly_transformer.py` | `third_party/Anomaly-Transformer` on `sys.path`, `from model.AnomalyTransformer import AnomalyTransformer` | `official` only on a CUDA machine; on CPU the compact fallback is used and `diag['official_import']` says why | `model/attn.py` calls `.cuda()` unconditionally (~L29, ~L51). Package names `model`, `utils`, `data_factory` collide with B5's repository: import one official repo per process. |
| B3 DAMP | `b3_damp.py` | `third_party/DAMP` on `sys.path`, `from damp import DAMP_2_0` | `official` when `DAMP_2_0(x, m, 1, t_start, lookahead=0, enable_output=False)` succeeds on every (asset, feature) stream; otherwise the numpy DAMP of this file | `damp.py` imports matplotlib; asserts `10 < m <= 1000`; raises on constant / near-constant regions (then the numpy implementation is used for that stream and the status stays `reimplemented`). Full-series profile indexed by subsequence end. |
| B4 MemStream | `b4_memstream.py` | file check `third_party/MemStream/code/memstream.py` (recorded in `diag`) | always `reimplemented`: `memstream.py` parses `sys.argv` at import and runs the experiment at module level, so it cannot be imported; the class is reproduced statement for statement | Compare against the vendored file with `diff` if in doubt; `memstream-pca.py` / `-ib.py` / `-knn.py` variants are not wrapped. |
| B5 DCdetector | `b5_dcdetector.py` | `third_party/KDD2023-DCdetector` on `sys.path`, `from model.DCdetector import DCdetector` | `official` when the class builds with `(win_size, enc_in, c_out, n_heads, d_model, e_layers, patch_size, channel, dropout)` | Needs `einops` and `tkinter` (`from tkinter import _flatten`); `RevIN` is instantiated inside `forward`. Same package-name collision as B2. |
| B6 MOMENT | `b6_moment.py` | `from momentfm import MOMENTPipeline` (pip package, not `third_party/`) | `official` when the import succeeds; `wrapper-needed` otherwise (`fit` raises `ImportError` naming `momentfm`) | Needs `transformers`, `huggingface_hub` and the checkpoint download (`AutonLab/MOMENT-1-large`, 0.3B parameters); zero-shot, no training. L8 cutoff 2024-02-01. |
| B7 CATCH | `b7_catch.py` | `third_party/CATCH` on `sys.path`, `from ts_benchmark.baselines.catch.models.CATCH_model import CATCHModel` | `official` when `CATCHModel(config)` builds from a namespace with `c_in, affine, subtract_last, patch_size, patch_stride, seq_len, cf_dim, e_layers, n_heads, d_ff, head_dim, dropout, d_model, regular_lambda, temperature, individual, head_dropout` | The TFB package has its own requirements (`requirements.txt`); the wrapper then uses the official `dcloss` in the loss. |

Rules: `status = 'official'` is set only after the official class was imported *and* instantiated successfully; any failure
falls back to the re-implementation and the reason is kept in `bl.diag['official_import']`. `cfg['model']['use_official'] =
false` disables the lookup. The vendored repositories are never modified; the convention line numbers in
`conventions_b2_b7.md` refer to the default branches as fetched on 2026-10-03.
