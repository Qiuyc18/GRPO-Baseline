#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENTRY=""
INTERVAL=60
CONSECUTIVE_IDLE=3
UTIL_THRESHOLD=5
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE=""
RUN_MONITOR=0

usage() {
  cat <<EOF
Usage:
  $(basename "$0") -e train/train_xxx.sh [options]

Options:
  -e, --entry PATH        训练脚本路径，支持相对项目根目录路径
  -i, --interval SEC      检查间隔秒数，默认: ${INTERVAL}
  -n, --consecutive NUM   连续空闲次数，默认: ${CONSECUTIVE_IDLE}
  -u, --util-threshold P  GPU 利用率阈值(%)，低于等于该值视为空闲，默认: ${UTIL_THRESHOLD}
  -l, --log-dir DIR       日志目录，默认: ${LOG_DIR}
  -h, --help              显示帮助

Example:
  $(basename "$0") -e train/train_qwen3_4B_math_grpo.sh
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  echo "Error: $*" >&2
  exit 1
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --monitor)
        RUN_MONITOR=1
        shift
        ;;
      --log-file)
        [[ $# -ge 2 ]] || die "--log-file requires a value"
        LOG_FILE="$2"
        shift 2
        ;;
      -e|--entry)
        [[ $# -ge 2 ]] || die "$1 requires a value"
        ENTRY="$2"
        shift 2
        ;;
      -i|--interval)
        [[ $# -ge 2 ]] || die "$1 requires a value"
        INTERVAL="$2"
        shift 2
        ;;
      -n|--consecutive)
        [[ $# -ge 2 ]] || die "$1 requires a value"
        CONSECUTIVE_IDLE="$2"
        shift 2
        ;;
      -u|--util-threshold)
        [[ $# -ge 2 ]] || die "$1 requires a value"
        UTIL_THRESHOLD="$2"
        shift 2
        ;;
      -l|--log-dir)
        [[ $# -ge 2 ]] || die "$1 requires a value"
        LOG_DIR="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown argument: $1"
        ;;
    esac
  done
}

validate_args() {
  [[ -n "${ENTRY}" ]] || die "missing required argument: -e/--entry"
  [[ "${INTERVAL}" =~ ^[0-9]+$ ]] || die "--interval must be a positive integer"
  [[ "${CONSECUTIVE_IDLE}" =~ ^[0-9]+$ ]] || die "--consecutive must be a positive integer"
  [[ "${UTIL_THRESHOLD}" =~ ^[0-9]+$ ]] || die "--util-threshold must be a non-negative integer"
  (( INTERVAL > 0 )) || die "--interval must be > 0"
  (( CONSECUTIVE_IDLE > 0 )) || die "--consecutive must be > 0"
}

resolve_entry_path() {
  if [[ "${ENTRY}" = /* ]]; then
    [[ -f "${ENTRY}" ]] || die "entry script not found: ${ENTRY}"
    ENTRY="$(cd "$(dirname "${ENTRY}")" && pwd)/$(basename "${ENTRY}")"
    return
  fi

  if [[ -f "${PROJECT_ROOT}/${ENTRY}" ]]; then
    ENTRY="$(cd "$(dirname "${PROJECT_ROOT}/${ENTRY}")" && pwd)/$(basename "${ENTRY}")"
    return
  fi

  if [[ -f "${ENTRY}" ]]; then
    ENTRY="$(cd "$(dirname "${ENTRY}")" && pwd)/$(basename "${ENTRY}")"
    return
  fi

  die "entry script not found: ${ENTRY}"
}

collect_gpu_utils() {
  if ! command -v rocm-smi >/dev/null 2>&1; then
    return 1
  fi

  local output=""
  if output="$(rocm-smi --showuse --json 2>/dev/null)"; then
    if [[ -n "${output}" ]]; then
      if python3 - "${output}" <<'PY'; then
import json
import re
import sys

vals = []

def walk(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_text = str(key).lower()
            if "gpu use" in key_text or key_text in {"gpu%", "gfx activity"}:
                match = re.search(r"-?\d+(?:\.\d+)?", str(value))
                if match:
                    vals.append(float(match.group(0)))
            walk(value)
    elif isinstance(obj, list):
        for item in obj:
            walk(item)

try:
    walk(json.loads(sys.argv[1]))
except Exception:
    sys.exit(1)

if not vals:
    sys.exit(1)

for value in vals:
    print(value)
PY
        return 0
      fi
    fi
  fi

  output="$(rocm-smi --showuse 2>/dev/null)" || return 1
  [[ -n "${output}" ]] || return 1

  python3 - "${output}" <<'PY' || return 1
import re
import sys

vals = []
for raw in sys.argv[1].splitlines():
    line = raw.strip()
    if not line or set(line) <= set("=- "):
        continue
    if "GPU use" in line:
        match = re.search(r"GPU use.*?(-?\d+(?:\.\d+)?)\s*%?\s*$", line)
        if match:
            vals.append(float(match.group(1)))
            continue
        numbers = re.findall(r"-?\d+(?:\.\d+)?", line)
        if numbers:
            vals.append(float(numbers[-1]))
            continue
    if re.match(r"^(GPU\[\d+\]|\d+)\b", line) and "%" in line:
        percents = re.findall(r"(-?\d+(?:\.\d+)?)\s*%", line)
        if percents:
            vals.append(float(percents[-1]))

if not vals:
    sys.exit(1)

for value in vals:
    print(value)
PY
}

collect_gpu_process_pids() {
  command -v lsof >/dev/null 2>&1 || return 1

  local device_paths=()
  if [[ -e /dev/kfd ]]; then
    device_paths+=("/dev/kfd")
  fi

  local render_node=""
  for render_node in /dev/dri/renderD*; do
    [[ -e "${render_node}" ]] || continue
    device_paths+=("${render_node}")
  done

  (( ${#device_paths[@]} > 0 )) || return 1

  local output=""
  output="$(lsof "${device_paths[@]}" 2>/dev/null || true)"
  printf '%s\n' "${output}" \
    | awk 'NR > 1 && $2 ~ /^[0-9]+$/ {print $2}' \
    | awk '!seen[$1]++'
  return 0
}

gpu_status_summary() {
  local utils_available=0
  local pids_available=0
  local utils_output=""
  local pids_output=""

  if utils_output="$(collect_gpu_utils)"; then
    utils_available=1
  fi
  if pids_output="$(collect_gpu_process_pids)"; then
    pids_available=1
  fi

  if (( !utils_available && !pids_available )); then
    echo "UNKNOWN|无法检测 GPU 状态；需要 rocm-smi 或可访问的 /dev/kfd、/dev/dri/renderD*"
    return 0
  fi

  local busy=0
  local summary_parts=()

  if (( utils_available )); then
    local utils_csv=""
    local max_util=0
    utils_csv="$(printf '%s\n' "${utils_output}" | awk 'NF {printf("%s%s", sep, $1); sep=","}')"
    max_util="$(printf '%s\n' "${utils_output}" | awk 'NF {if ($1 > max) max = $1} END {print max + 0}')"
    summary_parts+=("gpu_util=[${utils_csv}]")
    summary_parts+=("max_gpu_util=${max_util}%")

    if awk -v threshold="${UTIL_THRESHOLD}" 'NF && $1 > threshold {found=1} END {exit(found ? 0 : 1)}' <<<"${utils_output}"; then
      busy=1
    fi
  else
    summary_parts+=("gpu_util=unavailable")
  fi

  if (( pids_available )); then
    local pid_csv=""
    pid_csv="$(printf '%s\n' "${pids_output}" | awk 'NF {printf("%s%s", sep, $1); sep=","}')"
    if [[ -n "${pid_csv}" ]]; then
      summary_parts+=("gpu_pids=${pid_csv}")
      busy=1
    else
      summary_parts+=("gpu_pids=none")
    fi
  else
    summary_parts+=("gpu_pids=unavailable")
  fi

  if (( busy )); then
    echo "BUSY|$(IFS=' '; echo "${summary_parts[*]}")"
  else
    echo "IDLE|$(IFS=' '; echo "${summary_parts[*]}")"
  fi
}

run_monitor() {
  local idle_streak=0

  log "Timer monitor started"
  log "Project root: ${PROJECT_ROOT}"
  log "Entry script: ${ENTRY}"
  log "Interval: ${INTERVAL}s"
  log "Required idle streak: ${CONSECUTIVE_IDLE}"
  log "Util threshold: ${UTIL_THRESHOLD}%"
  [[ -n "${LOG_FILE}" ]] && log "Log file: ${LOG_FILE}"

  while true; do
    local status_line=""
    status_line="$(gpu_status_summary)"
    local status="${status_line%%|*}"
    local summary="${status_line#*|}"

    case "${status}" in
      IDLE)
        idle_streak=$((idle_streak + 1))
        log "GPU check: IDLE (${idle_streak}/${CONSECUTIVE_IDLE}) ${summary}"
        ;;
      BUSY)
        if (( idle_streak > 0 )); then
          log "GPU check: BUSY, idle streak reset to 0 ${summary}"
        else
          log "GPU check: BUSY ${summary}"
        fi
        idle_streak=0
        ;;
      *)
        log "GPU check: UNKNOWN ${summary}"
        return 1
        ;;
    esac

    if (( idle_streak >= CONSECUTIVE_IDLE )); then
      log "GPU idle confirmed ${CONSECUTIVE_IDLE} times, starting entry script"
      set +e
      (
        cd "${PROJECT_ROOT}"
        bash "${ENTRY}"
      )
      local status_code=$?
      set -e
      log "Entry script exited with status ${status_code}"
      return "${status_code}"
    fi

    sleep "${INTERVAL}"
  done
}

launch_background_monitor() {
  mkdir -p "${LOG_DIR}"

  local timestamp=""
  timestamp="$(date '+%Y%m%d_%H%M%S')"
  LOG_FILE="${LOG_DIR%/}/timer_${timestamp}.log"
  local pid_file="${LOG_FILE%.log}.pid"

  nohup bash "$0" \
    --monitor \
    --log-file "${LOG_FILE}" \
    --entry "${ENTRY}" \
    --interval "${INTERVAL}" \
    --consecutive "${CONSECUTIVE_IDLE}" \
    --util-threshold "${UTIL_THRESHOLD}" \
    --log-dir "${LOG_DIR}" \
    >"${LOG_FILE}" 2>&1 &

  local monitor_pid=$!
  echo "${monitor_pid}" > "${pid_file}"

  echo "Monitor started"
  echo "  PID: ${monitor_pid}"
  echo "  Log: ${LOG_FILE}"
  echo "  PID file: ${pid_file}"
  echo "  Stop: kill ${monitor_pid}"
}

main() {
  parse_args "$@"
  validate_args
  resolve_entry_path

  if (( RUN_MONITOR )); then
    run_monitor
  else
    launch_background_monitor
  fi
}

main "$@"
