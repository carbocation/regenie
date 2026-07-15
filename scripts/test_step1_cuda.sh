#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${BGEN_PATH:?Set BGEN_PATH to a built BGEN v1.1.7 source directory}"

build_dir="${BUILD_DIR:-${repo_root}/build-cuda}"
device="${GPU_DEVICE:-0}"
jobs="${JOBS:-4}"
benchmark_blocks="${BENCHMARK_BLOCKS:-512}"
benchmark_samples="${BENCHMARK_SAMPLES:-20000}"
benchmark_phenotypes="${BENCHMARK_PHENOTYPES:-10}"
benchmark_repeats="${BENCHMARK_REPEATS:-3}"
stream_chunk_mb="${CUDA_STREAM_CHUNK_MB:-64}"
resident_mb="${CUDA_RESIDENT_MB:-${REGENIE_CUDA_RESIDENT_MB:-1024}}"
validation_dir="${build_dir}/a100-validation"
if [[ ! "${resident_mb}" =~ ^[0-9]+$ ]]; then
  echo "resident_mb must be a non-negative integer (received '${resident_mb}')" >&2
  exit 2
fi

run_with_memory_log() {
  local label="$1"
  shift
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    "$@"
    return
  fi

  local memory_log="${validation_dir}/memory_${label}.txt"
  : > "${memory_log}"
  nvidia-smi --id="${device}" --query-gpu=memory.used \
    --format=csv,noheader,nounits >> "${memory_log}" 2>/dev/null || true
  (
    while :; do
      nvidia-smi --id="${device}" --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null || true
      sleep 0.2
    done
  ) >> "${memory_log}" &
  local monitor_pid=$!
  local command_status=0
  "$@" || command_status=$?
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true

  local peak_memory
  local memory_samples
  peak_memory="$(awk '
    BEGIN { peak = 0 }
    /^[[:space:]]*[0-9]+([.][0-9]+)?[[:space:]]*$/ {
      value = $1 + 0
      if (value > peak) peak = value
    }
    END { print peak }
  ' "${memory_log}")"
  memory_samples="$(awk '
    /^[[:space:]]*[0-9]+([.][0-9]+)?[[:space:]]*$/ { count += 1 }
    END { print count + 0 }
  ' "${memory_log}")"
  echo "STEP1_CUDA_MEMORY label=${label} peak_mib=${peak_memory} samples=${memory_samples}"
  if (( memory_samples == 0 )) && (( command_status == 0 )); then
    echo "No numeric GPU memory samples were recorded for ${label}" >&2
    return 98
  fi
  return "${command_status}"
}

nvcc --version
export REGENIE_CUDA_RESIDENT_MB="${resident_mb}"
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version \
    --format=csv,noheader; then
    nvidia-smi
  fi
fi

cmake -S "${repo_root}" -B "${build_dir}" \
  -DREGENIE_WITH_CUDA=ON \
  -DREGENIE_CUDA_ARCHITECTURES=80 \
  -DBUILD_TESTING=ON
cmake --build "${build_dir}" --target regenie step1_compute_test -j "${jobs}"
mkdir -p "${validation_dir}"

"${build_dir}/step1_compute_test" --backend cpu
cuda_output="$("${build_dir}/step1_compute_test" --backend cuda --device "${device}")"
printf '%s\n' "${cuda_output}"
grep -q '^STEP1_BACKEND_TEST backend=cuda status=PASS$' <<<"${cuda_output}"
if (( resident_mb > 0 )); then
  grep -q '^STEP1_BACKEND_TEST case=genotype_preprocessing backend_processed=1 .* status=PASS$' \
    <<<"${cuda_output}"
else
  grep -q '^STEP1_BACKEND_TEST case=genotype_preprocessing backend_processed=0 .* status=PASS$' \
    <<<"${cuda_output}"
fi

if (( resident_mb > 0 )); then
  fallback_output="$(REGENIE_CUDA_RESIDENT_MB=0 \
    "${build_dir}/step1_compute_test" --backend cuda --device "${device}")"
  printf '%s\n' "${fallback_output}"
  grep -q '^STEP1_BACKEND_TEST case=genotype_preprocessing backend_processed=0 .* status=PASS$' \
    <<<"${fallback_output}"
  grep -q '^STEP1_BACKEND_TEST backend=cuda status=PASS$' <<<"${fallback_output}"
fi
auto_output="$("${build_dir}/step1_compute_test" --backend auto --device "${device}")"
printf '%s\n' "${auto_output}"
grep -q '^STEP1_BACKEND_TEST backend=cuda status=PASS$' <<<"${auto_output}"

if [[ "${RUN_COMPUTE_SANITIZER:-1}" != "0" ]] && \
   command -v compute-sanitizer >/dev/null 2>&1; then
  REGENIE_CUDA_CHUNK_MB="${stream_chunk_mb}" \
    compute-sanitizer --tool memcheck --error-exitcode 99 \
    "${build_dir}/step1_compute_test" --backend cuda --device "${device}"
fi

"${build_dir}/step1_compute_test" --backend cpu --benchmark \
  --blocks "${benchmark_blocks}" --samples "${benchmark_samples}" \
  --phenotypes "${benchmark_phenotypes}" --repeats "${benchmark_repeats}"
run_with_memory_log backend_benchmark \
  env REGENIE_CUDA_CHUNK_MB="${stream_chunk_mb}" \
  "${build_dir}/step1_compute_test" --backend cuda --device "${device}" --benchmark \
  --blocks "${benchmark_blocks}" --samples "${benchmark_samples}" \
  --phenotypes "${benchmark_phenotypes}" --repeats "${benchmark_repeats}"

awk '
  NR == FNR {
    if (FNR > 1) event[$1 FS $2] = $3
    next
  }
  FNR == 1 {
    print "FID IID TIME EVENT"
    next
  }
  {
    time = $3
    if (time < 0) time = -time
    print $1, $2, time + 0.1, event[$1 FS $2]
  }
' "${repo_root}/example/phenotype_bin.txt" \
  "${repo_root}/example/phenotype.txt" > "${validation_dir}/phenotype_t2e.txt"

qt_common_args=(
  --step 1
  --bed "${repo_root}/example/example"
  --covarFile "${repo_root}/example/covariates.txt"
  --phenoFile "${repo_root}/example/phenotype.txt"
  --remove "${repo_root}/example/fid_iid_to_remove.txt"
  --qt
  --bsize 100
  --threads 1
  --seed 12345
  --step1-profile
)

count_common_args=(
  --step 1
  --bed "${repo_root}/example/example"
  --covarFile "${repo_root}/example/covariates.txt"
  --phenoFile "${repo_root}/example/phenotype_bin.txt"
  --remove "${repo_root}/example/fid_iid_to_remove.txt"
  --ct
  --bsize 100
  --threads 1
  --seed 12345
  --step1-profile
)

binary_common_args=(
  --step 1
  --bed "${repo_root}/example/example"
  --exclude "${repo_root}/example/snplist_rm.txt"
  --covarFile "${repo_root}/example/covariates.txt"
  --phenoFile "${repo_root}/example/phenotype_bin.txt"
  --remove "${repo_root}/example/fid_iid_to_remove.txt"
  --bt
  --bsize 100
  --threads 1
  --seed 12345
  --step1-profile
)

t2e_common_args=(
  --step 1
  --bed "${repo_root}/example/example"
  --covarFile "${repo_root}/example/covariates.txt"
  --phenoFile "${validation_dir}/phenotype_t2e.txt"
  --remove "${repo_root}/example/fid_iid_to_remove.txt"
  --t2e
  --phenoColList TIME
  --eventColList EVENT
  --bsize 100
  --threads 1
  --seed 12345
  --step1-profile
)

run_end_to_end_pair() {
  local label="$1"
  local profile_mode="$2"
  shift 2
  local cpu_prefix="${validation_dir}/cpu_${label}"
  local cuda_prefix="${validation_dir}/cuda_${label}"

  "${build_dir}/regenie" "$@" \
    --compute-backend cpu --out "${cpu_prefix}"
  run_with_memory_log "${label}" "${build_dir}/regenie" "$@" \
    --compute-backend cuda --gpu-device "${device}" --out "${cuda_prefix}"

  grep -Fq 'Step 1 compute backend : [cuda]' "${cuda_prefix}.log"
  grep -q "^STEP1_PROFILE version=4 backend=cuda mode=${profile_mode} " "${cuda_prefix}.log"

  shopt -s nullglob
  local cpu_loco_files=("${cpu_prefix}"_*.loco)
  if (( ${#cpu_loco_files[@]} == 0 )); then
    echo "No CPU LOCO files were produced for ${label}" >&2
    exit 1
  fi
  local cpu_file suffix cuda_file
  for cpu_file in "${cpu_loco_files[@]}"; do
    suffix="${cpu_file#${cpu_prefix}_}"
    cuda_file="${cuda_prefix}_${suffix}"
    python3 "${repo_root}/scripts/compare_numeric_files.py" \
      "${cpu_file}" "${cuda_file}" \
      --rtol "${LOCO_RTOL:-1e-7}" --atol "${LOCO_ATOL:-1e-9}"
  done
}

run_end_to_end_pair kfold kfold "${qt_common_args[@]}"
run_end_to_end_pair loocv loocv "${qt_common_args[@]}" --loocv
run_end_to_end_pair lowmem_kfold kfold "${qt_common_args[@]}" \
  --lowmem --lowmem-prefix "${validation_dir}/lowmem_l0"
run_end_to_end_pair test_l0_kfold kfold \
  "${qt_common_args[@]}" --test-l0 --l0-pval-thr 0.05 --phenoCol Y1
run_end_to_end_pair test_l0_loocv loocv \
  "${qt_common_args[@]}" --loocv --test-l0 --l0-pval-thr 0.05
run_end_to_end_pair count_kfold kfold "${count_common_args[@]}"
run_end_to_end_pair count_loocv loocv "${count_common_args[@]}" --loocv
run_end_to_end_pair binary_loocv loocv "${binary_common_args[@]}"
run_end_to_end_pair binary_full loocv \
  "${binary_common_args[@]}" --loocv --l1-full
run_end_to_end_pair t2e_kfold kfold "${t2e_common_args[@]}"

echo "STEP1_CUDA_VALIDATION status=PASS device=${device} results=${validation_dir}"
