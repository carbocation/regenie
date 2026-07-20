#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_profiled.sh --label LABEL --system-label SYSTEM --output-root DIR \
    [options] -- COMMAND [ARG ...]

Options:
  --system-label LABEL     Stable hardware/configuration label (required)
  --gpu-device INDEX       GPU to sample with nvidia-smi (default: none)
  --sample-interval SEC    Telemetry interval in seconds (default: 1)
  --revision REVISION      Revision recorded in metadata (default: auto-detect)
  -h, --help               Show this help

The command is run once. The wrapper creates a timestamped run directory with
raw and timestamped console output, GNU time statistics, system metadata,
vmstat/iostat samples when available, NVIDIA telemetry when requested, and a
machine-readable summary. The command's exit status is preserved.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
label=""
system_label=""
output_root=""
gpu_device=""
sample_interval="1"
revision=""

while (( $# > 0 )); do
  case "$1" in
    --label)
      label="${2:?--label requires a value}"
      shift 2
      ;;
    --system-label)
      system_label="${2:?--system-label requires a value}"
      shift 2
      ;;
    --output-root)
      output_root="${2:?--output-root requires a value}"
      shift 2
      ;;
    --gpu-device)
      gpu_device="${2:?--gpu-device requires a value}"
      shift 2
      ;;
    --sample-interval)
      sample_interval="${2:?--sample-interval requires a value}"
      shift 2
      ;;
    --revision)
      revision="${2:?--revision requires a value}"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown wrapper option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${label}" || -z "${system_label}" || -z "${output_root}" || $# -eq 0 ]]; then
  usage >&2
  exit 2
fi
if [[ ! "${label}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "--label may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi
if [[ ! "${system_label}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "--system-label may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi
if [[ ! "${sample_interval}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   ! awk -v value="${sample_interval}" 'BEGIN { exit !(value > 0) }'; then
  echo "--sample-interval must be a positive number" >&2
  exit 2
fi
if [[ -n "${gpu_device}" && ! "${gpu_device}" =~ ^[0-9]+$ ]]; then
  echo "--gpu-device must be a non-negative integer" >&2
  exit 2
fi
if [[ ! -x "$1" ]]; then
  echo "Benchmark command is not executable: $1" >&2
  exit 2
fi
if [[ ! -x /usr/bin/time ]] ||
   ! /usr/bin/time --version 2>&1 | grep -Eq 'GNU [Tt]ime'; then
  echo "GNU /usr/bin/time is required" >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${output_root%/}/${label}-${timestamp}"
mkdir -p "${run_dir}"

command_text=""
printf -v command_text '%q ' "$@"
command_text="${command_text% }"
printf '%s\n' "${command_text}" > "${run_dir}/command.txt"

if [[ -z "${revision}" ]]; then
  command_dir="$(cd "$(dirname "$1")" && pwd)"
  revision="$(git -C "${command_dir}" rev-parse HEAD 2>/dev/null || true)"
fi
if [[ -z "${revision}" ]]; then
  revision="unknown"
fi

binary_sha256="$(sha256sum "$1" | awk '{print $1}')"
if command -v ldd >/dev/null 2>&1; then
  ldd "$1" > "${run_dir}/binary_libraries.txt" 2>&1 || true
fi
{
  printf 'key\tvalue\n'
  printf 'label\t%s\n' "${label}"
  printf 'system_label\t%s\n' "${system_label}"
  printf 'run_dir\t%s\n' "${run_dir}"
  printf 'start_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'revision\t%s\n' "${revision}"
  printf 'binary\t%s\n' "$1"
  printf 'binary_sha256\t%s\n' "${binary_sha256}"
  printf 'gpu_device\t%s\n' "${gpu_device:-none}"
  printf 'sample_interval_seconds\t%s\n' "${sample_interval}"
  printf 'command\t%s\n' "${command_text}"
} > "${run_dir}/metadata.tsv"

{
  uname -srm
  if command -v lscpu >/dev/null 2>&1; then
    lscpu || true
  fi
  if command -v free >/dev/null 2>&1; then
    free -h || true
  fi
  if command -v lsblk >/dev/null 2>&1; then
    lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS || true
  fi
  df -hT 2>/dev/null || df -h
} > "${run_dir}/host.txt" 2>&1

monitor_pids=()
stop_monitors() {
  local pid
  for pid in "${monitor_pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${monitor_pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap stop_monitors EXIT INT TERM

if command -v vmstat >/dev/null 2>&1; then
  vmstat -w -t 1 > "${run_dir}/vmstat.log" 2>&1 &
  monitor_pids+=("$!")
fi
if command -v iostat >/dev/null 2>&1; then
  iostat -dx -t 1 > "${run_dir}/iostat.log" 2>&1 &
  monitor_pids+=("$!")
fi

if [[ -n "${gpu_device}" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "--gpu-device was supplied but nvidia-smi is unavailable" >&2
    exit 2
  fi
  nvidia-smi --id="${gpu_device}" \
    --query-gpu=index,name,compute_cap,memory.total,power.limit,driver_version \
    --format=csv,noheader > "${run_dir}/gpu.txt"
  if ! nvidia-smi --id="${gpu_device}" \
    --query-gpu=index,name,pstate,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,clocks.current.sm,clocks.current.memory,temperature.gpu \
    --format=csv,noheader,nounits >/dev/null; then
    echo "The requested nvidia-smi telemetry fields are unavailable" >&2
    exit 2
  fi
  (
    printf '%s\n' 'epoch_s,index,name,pstate,gpu_util_pct,memory_util_pct,memory_used_mib,memory_total_mib,power_w,power_limit_w,sm_clock_mhz,memory_clock_mhz,temperature_c'
    while :; do
      printf '%s,' "$(date +%s.%N)"
      nvidia-smi --id="${gpu_device}" \
        --query-gpu=index,name,pstate,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,clocks.current.sm,clocks.current.memory,temperature.gpu \
        --format=csv,noheader,nounits
      sleep "${sample_interval}"
    done
  ) > "${run_dir}/gpu.csv" 2> "${run_dir}/gpu-monitor.err" &
  monitor_pids+=("$!")
fi

set +e
/usr/bin/time -o "${run_dir}/resource.tsv" \
  -f $'key\tvalue\ncommand_exit_status\t%x\nwall_seconds\t%e\nuser_seconds\t%U\nsystem_seconds\t%S\nmax_rss_kb\t%M\nfilesystem_inputs\t%I\nfilesystem_outputs\t%O\nmajor_page_faults\t%F\nminor_page_faults\t%R\nvoluntary_context_switches\t%w\ninvoluntary_context_switches\t%c\naverage_cpu_percent\t%P' \
  stdbuf -oL -eL "$@" 2>&1 \
  | tee "${run_dir}/console.log" \
  | awk '{ print systime() "\t" $0; fflush() }' \
  | tee "${run_dir}/console.timestamped.log"
command_status=${PIPESTATUS[0]}
set -e

stop_monitors
monitor_pids=()
trap - EXIT INT TERM

{
  printf 'end_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'wrapper_exit_status\t%s\n' "${command_status}"
} >> "${run_dir}/metadata.tsv"

python3 "${script_dir}/summarize_profile.py" "${run_dir}"
printf 'BENCHMARK_RUN_DIR=%s\n' "${run_dir}"
exit "${command_status}"
