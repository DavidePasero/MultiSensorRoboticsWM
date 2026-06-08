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
PARALLEL_JOBS="${PARALLEL_JOBS:-5}"
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
  job_dir/run_probing_suite.sh [models_root] [probe args...]
  job_dir/run_probing_suite.sh --models-root /path/to/group [probe args...]

Behavior:
  - discovers model subfolders under models_root
  - picks the latest *_object.ckpt in each subfolder
  - runs linear, mlp, and knn probes by default
  - writes JSON results, per-run logs, and a Markdown report

Examples:
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

if [[ -z "$MODELS_ROOT" ]]; then
  echo "Missing models_root. Pass a folder path or use --models-root." >&2
  usage >&2
  exit 1
fi

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

mapfile -t MODEL_ROWS < <(discover_models "$MODELS_ROOT")

if ((${#MODEL_ROWS[@]} == 0)); then
  echo "No model checkpoints discovered under $MODELS_ROOT" >&2
  exit 1
fi

echo "Probing suite output directory: $RESULTS_DIR"
echo "Models root: $MODELS_ROOT"
echo "Probe types: ${PROBE_TYPES[*]}"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "Discovered models:"
for row in "${MODEL_ROWS[@]}"; do
  IFS=$'\t' read -r model_label checkpoint <<< "$row"
  echo "  - $model_label -> $checkpoint"
done

append_record() {
  RECORD_PATH="$1" \
  RECORD_MODEL_LABEL="$2" \
  RECORD_PROBE_TYPE="$3" \
  RECORD_CHECKPOINT="$4" \
  RECORD_OUTPUT_JSON="$5" \
  RECORD_OUTPUT_LOG="$6" \
  RECORD_STATUS="$7" \
  RECORD_EXIT_CODE="$8" \
  "$PYTHON_BIN" -c '
import json
import os

record = {
    "model_label": os.environ["RECORD_MODEL_LABEL"],
    "probe_type": os.environ["RECORD_PROBE_TYPE"],
    "checkpoint": os.environ["RECORD_CHECKPOINT"],
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
  "$PYTHON_BIN" - <<'PY' "$RAW_RECORDS" "$REPORT_MD" "$MODELS_ROOT" "${PROBE_TYPES[*]}"
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
lines.append(f"- Raw records: `{raw_path}`")
lines.append("")

lines.append("## Run Status")
lines.append("")
lines.append("| Model | Probe | Status | Exit | Checkpoint | JSON | Log |")
lines.append("|---|---|---|---:|---|---|---|")
for model in models:
    for probe_type in probe_types:
        entry = results_by_key.get((model, probe_type))
        if not entry:
            lines.append(f"| {model} | {probe_type} | missing | n/a | n/a | n/a | n/a |")
            continue
        record = entry["record"]
        lines.append(
            f"| {model} | {probe_type} | {record['status']} | {record['exit_code']} | "
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
  local probe_type="$3"
  shift 3
  local output_json="$RESULTS_DIR/${model_label}_${probe_type}.json"
  local output_log="$RESULTS_DIR/${model_label}_${probe_type}.log"

  echo "[$model_label][$probe_type] Starting probe run"
  "$ROOT_DIR/job_dir/run_probe_experiments.sh" \
    "$checkpoint" \
    "$probe_type" \
    --output "$output_json" \
    "$@" >"$output_log" 2>&1
}

declare -A PID_TO_MODEL=()
declare -A PID_TO_PROBE=()
declare -A PID_TO_CKPT=()
declare -A PID_TO_JSON=()
declare -A PID_TO_LOG=()

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
    "$status"

  unset PID_TO_MODEL["$finished_pid"]
  unset PID_TO_PROBE["$finished_pid"]
  unset PID_TO_CKPT["$finished_pid"]
  unset PID_TO_JSON["$finished_pid"]
  unset PID_TO_LOG["$finished_pid"]

  return "$status"
}

active_jobs=0
failures=0

for row in "${MODEL_ROWS[@]}"; do
  IFS=$'\t' read -r model_label checkpoint <<< "$row"
  for probe_type in "${PROBE_TYPES[@]}"; do
    output_json="$RESULTS_DIR/${model_label}_${probe_type}.json"
    output_log="$RESULTS_DIR/${model_label}_${probe_type}.log"

    run_one "$model_label" "$checkpoint" "$probe_type" "${EXTRA_ARGS[@]}" &
    pid=$!
    PID_TO_MODEL["$pid"]="$model_label"
    PID_TO_PROBE["$pid"]="$probe_type"
    PID_TO_CKPT["$pid"]="$checkpoint"
    PID_TO_JSON["$pid"]="$output_json"
    PID_TO_LOG["$pid"]="$output_log"
    ((active_jobs+=1))

    if (( active_jobs >= PARALLEL_JOBS )); then
      set +e
      reap_one
      status=$?
      set -e
      ((active_jobs-=1))
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
  ((active_jobs-=1))
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
