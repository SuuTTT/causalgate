#!/usr/bin/env bash
# CausalGate experiment grid: datasets x baselines x protocols x seeds {42, 2024, 2025, 2026}.
#   bash scripts/cg_run_all.sh --protocols published,walkforward,causal --seeds 42,2024,2025,2026 --datasets all --baselines all
#   bash scripts/cg_run_all.sh --datasets cg_planted --baselines causalgate,cg_damp --protocols causal --seeds 42 --smoke --max-folds 1
# Options: --one-switch (adds causal+L1..L7, E2; L8 is an audit-card flag, not a run) --ablations (CausalGate ablation configs, Abl.) --adia (P-C/ADIA replay)
# Real datasets need their raw files under data_raw/ (scripts/cg_prepare_data.py); a dataset whose panel cannot be
# built is skipped with a message, never faked. Protocol hashes must be frozen first (python3 scripts/cg_run.py --freeze).
# Dataset names are registry keys (cg_stocknet ...); cg_common.load_cfgs resolves them to datasets/stocknet.yaml.
set -u
set -o pipefail
FAILED=0
cd "$(dirname "$0")/.."
PROTOCOLS="published,walkforward,causal"
SEEDS="42,2024,2025,2026"
DATASETS="all"
BASELINES="all"
EXTRA=""
ONE_SWITCH=0; ABLATIONS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --protocols) PROTOCOLS="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --datasets) DATASETS="$2"; shift 2;;
    --baselines) BASELINES="$2"; shift 2;;
    --one-switch) ONE_SWITCH=1; shift;;
    --ablations) ABLATIONS=1; shift;;
    --smoke) EXTRA="$EXTRA --smoke"; shift;;
    --max-folds) EXTRA="$EXTRA --max-folds $2"; shift 2;;
    --device) EXTRA="$EXTRA --device $2"; shift 2;;
    --adia) EXTRA="$EXTRA --adia"; shift;;
    --no-freeze-check) EXTRA="$EXTRA --no-freeze-check"; shift;;
    *) echo "unknown option $1"; exit 1;;
  esac
done
ALL_DATASETS="cg_planted cg_stocknet cg_fi2010 cg_fnspid cg_edgar cg_crypto_pd cg_csi300_limit cg_sp500_tick cg_qlib_alpha158 cg_wsb_squeeze cg_fin_graph"
ALL_BASELINES="causalgate cg_cpd_dmn cg_anomaly_transformer cg_damp cg_memstream cg_dcdetector cg_moment cg_catch cg_stage cg_alridhawi cg_konczal_evt"
[[ "$DATASETS" == "all" ]] && DATASETS="$ALL_DATASETS" || DATASETS="${DATASETS//,/ }"
[[ "$BASELINES" == "all" ]] && BASELINES="$ALL_BASELINES" || BASELINES="${BASELINES//,/ }"
PROTO_LIST="${PROTOCOLS//,/ }"
[[ $ONE_SWITCH -eq 1 ]] && PROTO_LIST="$PROTO_LIST causal+L1 causal+L2 causal+L3 causal+L4 causal+L5 causal+L6 causal+L7"
[[ -f configs/causalgate/PROTOCOL_HASH ]] || { echo "[cg_run_all] protocol hashes not frozen: run python3 scripts/cg_run.py --freeze"; exit 1; }
mkdir -p results/causalgate/logs
for ds in $DATASETS; do
  python3 -c "
import sys; sys.path.insert(0, 'scripts'); from cg_common import import_all, load_cfgs, build_panel
import argparse; import_all()
a = argparse.Namespace(dataset='$ds', baseline='causalgate', smoke=False, override=[], max_folds=None)
cfg, ds_cfg, _ = load_cfgs(a); build_panel(ds_cfg)" >/dev/null 2>results/causalgate/logs/${ds}_panel.err || { echo "[skip] $ds (panel could not be built; first lines of results/causalgate/logs/${ds}_panel.err:)"; tail -3 "results/causalgate/logs/${ds}_panel.err" | sed 's/^/    /'; continue; }
  for s in ${SEEDS//,/ }; do
    for b in $BASELINES; do
      for pr in $PROTO_LIST; do
        echo "[run] $ds | $b | $pr | seed $s"
        python3 scripts/cg_run.py --dataset "$ds" --baseline "$b" --protocol "$pr" --seed "$s" $EXTRA > >(tee -a "results/causalgate/logs/${ds}_${b}_${pr}_s${s}.log" | grep -E "^\[|INFO|Error|error" | tail -3) 2>&1 \
          || { echo "[FAIL] $ds | $b | $pr | seed $s (see log)"; FAILED=$((FAILED+1)); }
      done
      if [[ $ABLATIONS -eq 1 && "$b" == "causalgate" ]]; then
        for abl in configs/causalgate/ablations/*.yaml; do
          name=$(basename "$abl" .yaml)
          pr=causal; [[ "$name" == "no_fg" ]] && pr="causal+L1"
          echo "[abl] $ds | causalgate/$name | $pr | seed $s"
          # one leaf-level override per line, compact JSON values, read into an array (no bash word-splitting)
          OV=()
          while IFS= read -r line; do OV+=("$line"); done < <(python3 -c "
import yaml, json
d = yaml.safe_load(open('$abl')) or {}; d.pop('_base_', None); d.pop('ablation', None)
def flat(p, v):
    if isinstance(v, dict):
        for k, x in v.items(): yield from flat(p + '.' + str(k), x)
    else: yield p + '=' + json.dumps(v, separators=(',', ':'))
for k, v in d.items():
    for item in flat('baseline.' + str(k), v): print(item)")
          python3 scripts/cg_run.py --dataset "$ds" --baseline causalgate --protocol "$pr" --seed "$s" --tag "abl_$name" \
            --override "${OV[@]}" $EXTRA > >(tee -a "results/causalgate/logs/${ds}_causalgate_${name}_s${s}.log" | grep -E "^\[|INFO|Error|error" | tail -3) 2>&1 \
            || { echo "[FAIL] abl $name | $ds | seed $s (see log)"; FAILED=$((FAILED+1)); }
        done
      fi
    done
  done
done
python3 scripts/cg_make_tables.py --results results/causalgate --fmt markdown
[[ $FAILED -eq 0 ]] || { echo "[cg_run_all] $FAILED run(s) failed"; exit 1; }
