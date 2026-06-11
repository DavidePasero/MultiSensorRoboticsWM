#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable_worldmodel}"
PARALLEL_JOBS="${PARALLEL_JOBS:-8}"
CACHE_KEYS_RAW="${CACHE_KEYS:-action proprio force_torque tactile}"
DRY_RUN=false
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/experiments/results/probing_suite_$TIMESTAMP}"
RAW_RECORDS="$RESULTS_DIR/probing_runs.jsonl"
REPORT_MD="$RESULTS_DIR/probing_report.md"

if [[ -n "${PROBE_TYPE:-}" ]]; then
  PROBE_TYPES_RAW="${PROBE_TYPE}"
else
  PROBE_TYPES_RAW="${PROBE_TYPES:-linear mlp knn}"
fi

usage() {
  cat <<'EOF'
Usage:
  job_dir/run_probing_suite.sh [probe args...]
  job_dir/run_probing_suite.sh [models_root] [probe args...]
  job_dir/run_probing_suite.sh --models-root /path/to/group [probe args...]

Behavior:
  - by default runs the button-press, drawer-open, and bin-picking non-masked model preset
  - with models_root, discovers model subfolders under models_root
  - picks the latest *_object.ckpt in each subfolder
  - runs linear, mlp, and knn probes by default
  - uses 8 parallel jobs by default
  - caches only action, proprio, force_torque, and tactile by default
  - writes JSON results, per-run logs, and a Markdown report

Examples:
  job_dir/run_probing_suite.sh --dry-run

  job_dir/run_probing_suite.sh --device cuda

  job_dir/run_probing_suite.sh /home/disco/.stable_worldmodel/button_press \
    --dataset-name metaworld_button_press_probing

  PROBE_TYPES="linear knn" PARALLEL_JOBS=3 \
  job_dir/run_probing_suite.sh /home/disco/.stable_worldmodel/bin_picking \
    --dataset-name metaworld_bin_picking_probing --device cuda
EOF
}

MODELS_ROOT="${MODELS_ROOT:-}"
EXTRA_ARGS=()
while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --models-root)
      MODELS_ROOT="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      RAW_RECORDS="$RESULTS_DIR/probing_runs.jsonl"
      REPORT_MD="$RESULTS_DIR/probing_report.md"
      shift 2
      ;;
    --parallel-jobs)
      PARALLEL_JOBS="$2"
      shift 2
      ;;
    --probe-types)
      PROBE_TYPES_RAW="$2"
      shift 2
      ;;
    --keys-to-cache|--cache-keys)
      CACHE_KEYS_RAW="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --config|--cache-dir|--dataset-name|--representation|--dropout|--knn-k|--knn-distance|--extract-batch-size|--probe-batch-size|--num-workers|--max-samples|--train-fraction|--val-fraction|--probe-step|--num-epochs|--lr|--weight-decay|--patience|--seed|--device)
      if (($# < 2)); then
        echo "Missing value for $1" >&2
        exit 1
      fi
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --hidden-dims|--experiments)
      EXTRA_ARGS+=("$1")
      shift
      while (($#)) && [[ "$1" != -* ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    -*)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    *)
      if [[ -z "$MODELS_ROOT" ]]; then
        MODELS_ROOT="$1"
      else
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

mkdir -p "$RESULTS_DIR"
: > "$RAW_RECORDS"

mapfile -t PROBE_TYPES < <(
  "$PYTHON_BIN" - <<'PY' "$PROBE_TYPES_RAW"
import re
import sys

raw = sys.argv[1]
items = [part.strip() for part in re.split(r"[\s,]+", raw) if part.strip()]
allowed = {"linear", "mlp", "knn"}
for item in items:
    if item not in allowed:
        raise SystemExit(f"Unsupported probe type '{item}'. Allowed: {sorted(allowed)}")
for item in items:
    print(item)
PY
)

if ((${#PROBE_TYPES[@]} == 0)); then
  echo "No probe types selected." >&2
  exit 1
fi

mapfile -t CACHE_KEYS < <(
  "$PYTHON_BIN" - <<'PY' "$CACHE_KEYS_RAW"
import re
import sys

raw = sys.argv[1]
for item in [part.strip() for part in re.split(r"[\s,]+", raw) if part.strip()]:
    print(item)
PY
)

discover_models() {
  "$PYTHON_BIN" - <<'PY' "$1"
from __future__ import annotations

import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
if not root.exists():
    raise SystemExit(f"models_root does not exist: {root}")
if not root.is_dir():
    raise SystemExit(f"models_root is not a directory: {root}")

epoch_pattern = re.compile(r"_epoch_(\d+)_object\.ckpt$")


def choose_checkpoint(directory: Path) -> Path | None:
    candidates = [p for p in directory.rglob("*_object.ckpt") if p.is_file()]
    if not candidates:
        return None

    def sort_key(path: Path):
        match = epoch_pattern.search(path.name)
        epoch = int(match.group(1)) if match else -1
        return (epoch, path.stat().st_mtime_ns, path.name)

    return max(candidates, key=sort_key)


entries: list[tuple[str, Path]] = []
for child in sorted(root.iterdir()):
    if not child.is_dir():
        continue
    checkpoint = choose_checkpoint(child)
    if checkpoint is not None:
        entries.append((child.name, checkpoint))

if not entries:
    checkpoint = choose_checkpoint(root)
    if checkpoint is not None:
        entries.append((root.name, checkpoint))

if not entries:
    raise SystemExit(f"No *_object.ckpt checkpoints found under {root}")

for label, checkpoint in entries:
    print(f"{label}\t{checkpoint}")
PY
}

if [[ -z "$MODELS_ROOT" ]]; then
  mapfile -t MODEL_ROWS < <(
    "$PYTHON_BIN" - <<'PY' "$STABLEWM_HOME"
from __future__ import annotations

import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
epoch_pattern = re.compile(r"_epoch_(\d+)_object\.ckpt$")


def choose_checkpoint(directory: Path) -> Path | None:
    candidates = [p for p in directory.rglob("*_object.ckpt") if p.is_file()]
    if not candidates:
        return None

    def sort_key(path: Path):
        match = epoch_pattern.search(path.name)
        epoch = int(match.group(1)) if match else -1
        return (epoch, path.stat().st_mtime_ns, path.name)

    return max(candidates, key=sort_key)


def resolve_model(candidates: list[str], search_roots: list[Path]) -> tuple[str, Path] | None:
    for model_name in candidates:
        for search_root in search_roots:
            model_dir = search_root / model_name
            if not model_dir.is_dir():
                continue
            checkpoint = choose_checkpoint(model_dir)
            if checkpoint is not None:
                return model_name, checkpoint
    return None


tasks = [
    {
        "task": "button_press",
        "dataset": "metaworld_button_press_probing",
        "search_roots": [root / "button_press", root],
        "models": [
            ["metaworld_coproj_button_press"],
            [
                "metaworld_selfattention_button_press_low_sigreg",
                "metaworld_selfattention_button_press_5",
                "metaworld_selfattention_button_press",
            ],
            ["metaworld_gated_button_press"],
            ["metaworld_pixels_button_press", "metaworld_pixels_button_press_2"],
        ],
    },
    {
        "task": "drawer_open",
        "dataset": "metaworld_drawer_open_probing",
        "search_roots": [root / "drawer_open", root],
        "models": [
            ["metaworld_coproj_drawer_open"],
            ["metaworld_selfattention_drawer_open", "metaworld_selfattention_drawer_open_low_sigreg"],
            ["metaworld_gated_drawer_open"],
            ["metaworld_pixels_drawer_open"],
        ],
    },
    {
        "task": "bin_picking",
        "dataset": "metaworld_bin_picking_probing",
        "search_roots": [root / "bin_picking", root],
        "models": [
            ["metaworld_coproj_bin_picking"],
            ["metaworld_selfattention_bin_picking"],
            ["metaworld_gated_bin_picking"],
            ["metaworld_pixels_bin_picking"],
        ],
    },
]

missing: list[str] = []
for spec in tasks:
    for candidates in spec["models"]:
        resolved = resolve_model(candidates, spec["search_roots"])
        if resolved is None:
            missing.append(f"{spec['task']}:{'|'.join(candidates)}")
            continue
        model_name, checkpoint = resolved
        print(f"{model_name}\t{checkpoint}\t{spec['dataset']}")

if missing:
    print(
        "Missing expected model(s): " + ", ".join(missing),
        file=sys.stderr,
    )

PY
  )
else
  mapfile -t MODEL_ROWS < <(discover_models "$MODELS_ROOT")
fi

if ((${#MODEL_ROWS[@]} == 0)); then
  echo "No model checkpoints discovered." >&2
  exit 1
fi

echo "Probing suite output directory: $RESULTS_DIR"
if [[ -n "$MODELS_ROOT" ]]; then
  echo "Models root: $MODELS_ROOT"
else
  echo "Models preset: button_press, drawer_open, bin_picking non-masked models"
fi
echo "Probe types: ${PROBE_TYPES[*]}"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "Keys to cache: ${CACHE_KEYS[*]:-(none)}"
echo "Discovered models:"
for row in "${MODEL_ROWS[@]}"; do
  IFS=$'\t' read -r model_label checkpoint dataset_name <<< "$row"
  if [[ -n "${dataset_name:-}" ]]; then
    echo "  - $model_label [$dataset_name] -> $checkpoint"
  else
    echo "  - $model_label -> $checkpoint"
  fi
done

if [[ "$DRY_RUN" == true ]]; then
  echo ""
  echo "Dry run only. No probe jobs launched."
  exit 0
fi

append_record() {
  RECORD_PATH="$1" \
  RECORD_MODEL_LABEL="$2" \
  RECORD_PROBE_TYPE="$3" \
  RECORD_CHECKPOINT="$4" \
  RECORD_OUTPUT_JSON="$5" \
  RECORD_OUTPUT_LOG="$6" \
  RECORD_STATUS="$7" \
  RECORD_EXIT_CODE="$8" \
  RECORD_DATASET_NAME="$9" \
  "$PYTHON_BIN" -c '
import json
import os

record = {
    "model_label": os.environ["RECORD_MODEL_LABEL"],
    "probe_type": os.environ["RECORD_PROBE_TYPE"],
    "checkpoint": os.environ["RECORD_CHECKPOINT"],
    "dataset_name": os.environ["RECORD_DATASET_NAME"],
    "output_json": os.environ["RECORD_OUTPUT_JSON"],
    "output_log": os.environ["RECORD_OUTPUT_LOG"],
    "status": os.environ["RECORD_STATUS"],
    "exit_code": int(os.environ["RECORD_EXIT_CODE"]),
}
with open(os.environ["RECORD_PATH"], "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")
'
}

render_report() {
  "$PYTHON_BIN" - <<'PY' "$RAW_RECORDS" "$REPORT_MD" "${MODELS_ROOT:-preset}" "${PROBE_TYPES[*]}" "${CACHE_KEYS[*]}"
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

raw_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
models_root = sys.argv[3]
probe_types = [item for item in sys.argv[4].split() if item]
cache_keys = [item for item in sys.argv[5].split() if item]

records = []
if raw_path.exists():
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

results_by_key = {}
for record in records:
    key = (record["model_label"], record["probe_type"])
    payload = None
    output_json = Path(record["output_json"])
    if record["status"] == "ok" and output_json.exists():
        with output_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    results_by_key[key] = {"record": record, "payload": payload}

models = sorted({record["model_label"] for record in records})
experiments = []
for entry in results_by_key.values():
    payload = entry["payload"]
    if payload is None:
        continue
    for experiment in payload.get("results", {}):
        if experiment not in experiments:
            experiments.append(experiment)

metric_priority = {
    "regression": [("rmse", True), ("mae", True), ("mse", True), ("loss", True)],
    "binary": [("accuracy", False), ("f1", False), ("precision", False), ("recall", False), ("loss", True)],
}


def first_metric(metrics: dict, task_type: str):
    for name, lower_is_better in metric_priority.get(task_type, [("loss", True)]):
        if name in metrics:
            return name, float(metrics[name]), lower_is_better
    if metrics:
        name = next(iter(metrics))
        return name, float(metrics[name]), True
    return None, None, True


def format_cell(model: str, probe_type: str, experiment: str):
    entry = results_by_key.get((model, probe_type))
    if not entry:
        return "n/a"
    if entry["record"]["status"] != "ok" or entry["payload"] is None:
        return f"failed ({entry['record']['exit_code']})"

    experiment_result = entry["payload"]["results"].get(experiment)
    if not experiment_result:
        return "n/a"
    test_metrics = experiment_result.get("metrics", {}).get("test", {})
    metric_name, metric_value, _ = first_metric(
        test_metrics,
        experiment_result.get("task_type", "regression"),
    )
    if metric_name is None or metric_value is None or math.isnan(metric_value):
        return "n/a"

    if metric_name in {"accuracy", "precision", "recall", "f1"}:
        return f"{metric_name}={metric_value:.4f}"
    return f"{metric_name}={metric_value:.6f}"


lines = []
lines.append("# Probing Suite Report")
lines.append("")
lines.append(f"- Models root: `{models_root}`")
lines.append(f"- Probe types: `{', '.join(probe_types)}`")
lines.append(f"- Keys cached: `{', '.join(cache_keys) if cache_keys else '(none)'}`")
lines.append(f"- Raw records: `{raw_path}`")
lines.append("")

lines.append("## Run Status")
lines.append("")
lines.append("| Model | Dataset | Probe | Status | Exit | Checkpoint | JSON | Log |")
lines.append("|---|---|---|---|---:|---|---|---|")
for model in models:
    for probe_type in probe_types:
        entry = results_by_key.get((model, probe_type))
        if not entry:
            lines.append(f"| {model} | n/a | {probe_type} | missing | n/a | n/a | n/a | n/a |")
            continue
        record = entry["record"]
        lines.append(
            f"| {model} | {record.get('dataset_name', '')} | {probe_type} | "
            f"{record['status']} | {record['exit_code']} | "
            f"`{record['checkpoint']}` | `{record['output_json']}` | `{record['output_log']}` |"
        )
lines.append("")

for experiment in experiments:
    lines.append(f"## {experiment}")
    lines.append("")
    lines.append("| Model | " + " | ".join(probe_types) + " |")
    lines.append("| " + " | ".join(["---"] * (len(probe_types) + 1)) + " |")
    for model in models:
        cells = [format_cell(model, probe_type, experiment) for probe_type in probe_types]
        lines.append("| " + model + " | " + " | ".join(cells) + " |")
    lines.append("")

tmp_path = report_path.with_suffix(report_path.suffix + ".tmp")
tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
tmp_path.replace(report_path)
print(report_path)
PY
}

run_one() {
  local model_label="$1"
  local checkpoint="$2"
  local dataset_name="$3"
  local probe_type="$4"
  shift 4
  local output_json="$RESULTS_DIR/${model_label}_${probe_type}.json"
  local output_log="$RESULTS_DIR/${model_label}_${probe_type}.log"
  local cache_args=()
  if ((${#CACHE_KEYS[@]} > 0)); then
    cache_args=(--keys-to-cache "${CACHE_KEYS[@]}")
  fi
  local dataset_args=()
  if [[ -n "$dataset_name" ]]; then
    dataset_args=(--dataset-name "$dataset_name")
  fi

  echo "[$model_label][$probe_type] Starting probe run"
  "$ROOT_DIR/job_dir/run_probe_experiments.sh" \
    "$checkpoint" \
    "$probe_type" \
    --output "$output_json" \
    "${dataset_args[@]}" \
    "${cache_args[@]}" \
    "$@" >"$output_log" 2>&1
}

declare -A PID_TO_MODEL=()
declare -A PID_TO_PROBE=()
declare -A PID_TO_CKPT=()
declare -A PID_TO_JSON=()
declare -A PID_TO_LOG=()
declare -A PID_TO_DATASET=()

reap_one() {
  local finished_pid=""
  local status=0
  set +e
  wait -n -p finished_pid
  status=$?
  set -e

  local model_label="${PID_TO_MODEL[$finished_pid]}"
  local probe_type="${PID_TO_PROBE[$finished_pid]}"
  local checkpoint="${PID_TO_CKPT[$finished_pid]}"
  local output_json="${PID_TO_JSON[$finished_pid]}"
  local output_log="${PID_TO_LOG[$finished_pid]}"
  local dataset_name="${PID_TO_DATASET[$finished_pid]}"

  local record_status="ok"
  if (( status != 0 )) || [[ ! -f "$output_json" ]]; then
    record_status="failed"
  fi

  append_record \
    "$RAW_RECORDS" \
    "$model_label" \
    "$probe_type" \
    "$checkpoint" \
    "$output_json" \
    "$output_log" \
    "$record_status" \
    "$status" \
    "$dataset_name"

  unset PID_TO_MODEL["$finished_pid"]
  unset PID_TO_PROBE["$finished_pid"]
  unset PID_TO_CKPT["$finished_pid"]
  unset PID_TO_JSON["$finished_pid"]
  unset PID_TO_LOG["$finished_pid"]
  unset PID_TO_DATASET["$finished_pid"]

  return "$status"
}

active_jobs=0
failures=0

for row in "${MODEL_ROWS[@]}"; do
  IFS=$'\t' read -r model_label checkpoint dataset_name <<< "$row"
  for probe_type in "${PROBE_TYPES[@]}"; do
    output_json="$RESULTS_DIR/${model_label}_${probe_type}.json"
    output_log="$RESULTS_DIR/${model_label}_${probe_type}.log"

    run_one "$model_label" "$checkpoint" "${dataset_name:-}" "$probe_type" "${EXTRA_ARGS[@]}" &
    pid=$!
    PID_TO_MODEL["$pid"]="$model_label"
    PID_TO_PROBE["$pid"]="$probe_type"
    PID_TO_CKPT["$pid"]="$checkpoint"
    PID_TO_JSON["$pid"]="$output_json"
    PID_TO_LOG["$pid"]="$output_log"
    PID_TO_DATASET["$pid"]="${dataset_name:-}"
    ((active_jobs+=1))

    if (( active_jobs >= PARALLEL_JOBS )); then
      set +e
      reap_one
      status=$?
      set -e
      active_jobs=$((active_jobs - 1))
      if (( status != 0 )); then
        ((failures+=1))
      fi
    fi
  done
done

while (( active_jobs > 0 )); do
  set +e
  reap_one
  status=$?
  set -e
  active_jobs=$((active_jobs - 1))
  if (( status != 0 )); then
    ((failures+=1))
  fi
done

render_report >/dev/null

echo ""
echo "Completed probing suite. Outputs:"
echo "  - $REPORT_MD"
echo "  - $RAW_RECORDS"
echo "  - $RESULTS_DIR/*.json"
echo "  - $RESULTS_DIR/*.log"

if (( failures != 0 )); then
  echo ""
  echo "Suite finished with $failures failed run(s). Check $REPORT_MD and the per-run logs."
  exit 1
fi
