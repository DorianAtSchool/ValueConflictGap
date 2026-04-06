#!/usr/bin/env bash

set -euo pipefail

INTERVAL="${1:-10}"
PROJECT_ROOT="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULTS_DIR="${PROJECT_ROOT}/pipeline/results/scenario_conversation"
CONV_DIR="${RESULTS_DIR}/conversations"
CKPT_DIR="${RESULTS_DIR}/checkpoints"
RUNS_DIR="${RESULTS_DIR}/runs"
LOG_FILE="${RESULTS_DIR}/experiment.log"

if ! [[ "${INTERVAL}" =~ ^[0-9]+$ ]] || [ "${INTERVAL}" -lt 1 ]; then
  echo "Usage: $0 [interval_seconds] [project_root]" >&2
  exit 1
fi

count_files() {
  local dir="$1"
  if [ -d "${dir}" ]; then
    find "${dir}" -type f 2>/dev/null | wc -l | tr -d ' '
  else
    echo "0"
  fi
}

latest_files() {
  local dir="$1"
  local limit="${2:-5}"
  if [ -d "${dir}" ]; then
    find "${dir}" -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' 2>/dev/null | sort | tail -n "${limit}"
  fi
}

print_gpu_summary() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', ' '{printf "GPU %s %-18s mem %s/%s MB util %s%%\n", $1, $2, $3, $4, $5}'
  else
    echo "nvidia-smi not found"
  fi
}

print_processes() {
  ps -eo pid,etime,%cpu,%mem,cmd \
    | grep -E 'run_runpod.py|run_scenario_conversation_experiment.py' \
    | grep -v grep || true
}

while true; do
  clear
  echo "=== PersonaDrifting Progress ==="
  echo "Time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "Project: ${PROJECT_ROOT}"
  echo

  echo "Counts"
  echo "  conversations: $(count_files "${CONV_DIR}")"
  echo "  checkpoints:   $(count_files "${CKPT_DIR}")"
  echo "  run summaries: $(count_files "${RUNS_DIR}")"
  echo

  echo "Processes"
  print_processes
  echo

  echo "GPU Summary"
  print_gpu_summary
  echo

  echo "Recent Conversations"
  latest_files "${CONV_DIR}" 5 || true
  echo

  echo "Recent Checkpoints"
  latest_files "${CKPT_DIR}" 5 || true
  echo

  echo "Experiment Log Tail"
  if [ -f "${LOG_FILE}" ]; then
    tail -n 20 "${LOG_FILE}"
  else
    echo "No log yet: ${LOG_FILE}"
  fi
  echo
  echo "Refreshing every ${INTERVAL}s. Ctrl+C to exit."
  sleep "${INTERVAL}"
done
