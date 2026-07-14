#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${BGEN_PATH:?Set BGEN_PATH to a built BGEN v1.1.7 source directory}"

build_dir="${BUILD_DIR:-${repo_root}/build-cuda}"
device="${GPU_DEVICE:-0}"
jobs="${JOBS:-4}"
cuda_architectures="${CUDA_ARCHITECTURES:-80}"
validation_label="${GPU_VALIDATION_LABEL:-a100}"
benchmark_blocks="${BENCHMARK_BLOCKS:-512}"
benchmark_samples="${BENCHMARK_SAMPLES:-20000}"
benchmark_phenotypes="${BENCHMARK_PHENOTYPES:-10}"
benchmark_repeats="${BENCHMARK_REPEATS:-3}"
benchmark_warmup_repeats="${BENCHMARK_WARMUP_REPEATS:-1}"
stream_chunk_mb="${CUDA_STREAM_CHUNK_MB:-64}"
resident_mb="${CUDA_RESIDENT_MB:-${REGENIE_CUDA_RESIDENT_MB:-1024}}"
level1_resident_mb="${CUDA_LEVEL1_RESIDENT_MB:-${REGENIE_CUDA_LEVEL1_RESIDENT_MB:-}}"
gram_precision="${CUDA_GRAM_PRECISION:-${REGENIE_CUDA_GRAM_PRECISION:-fp64}}"
fp32_gram_chunk_samples="${CUDA_FP32_GRAM_CHUNK_SAMPLES:-${REGENIE_CUDA_FP32_GRAM_CHUNK_SAMPLES:-128}}"
pinned_staging_mb="${CUDA_PINNED_STAGING_MB:-${REGENIE_CUDA_PINNED_STAGING_MB:-64}}"
pgen_prefetch_mb="${STEP1_PGEN_PREFETCH_MB:-${REGENIE_STEP1_PGEN_PREFETCH_MB:-4096}}"
pgen_tile_variants="${STEP1_PGEN_TILE_VARIANTS:-${REGENIE_STEP1_PGEN_TILE_VARIANTS:-8}}"
pgen_packed="${STEP1_PGEN_PACKED:-${REGENIE_STEP1_PGEN_PACKED:-1}}"
validation_dir="${VALIDATION_DIR:-${build_dir}/${validation_label}-validation}"
run_synthetic_benchmark="${RUN_SYNTHETIC_BENCHMARK:-0}"
synthetic_samples="${SYNTHETIC_SAMPLES:-20000}"
synthetic_variants="${SYNTHETIC_VARIANTS:-20000}"
synthetic_phenotypes="${SYNTHETIC_PHENOTYPES:-4}"
synthetic_chromosomes="${SYNTHETIC_CHROMOSOMES:-22}"
synthetic_bsize="${SYNTHETIC_BSIZE:-512}"
synthetic_threads="${SYNTHETIC_THREADS:-${jobs}}"
synthetic_seed="${SYNTHETIC_SEED:-20260712}"
synthetic_max_bed_gb="${SYNTHETIC_MAX_BED_GB:-4}"

if [[ -z "${cuda_architectures}" ]]; then
  echo "CUDA_ARCHITECTURES must not be empty" >&2
  exit 2
fi
if [[ ! "${validation_label}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "GPU_VALIDATION_LABEL must contain only letters, digits, '.', '_', or '-'" >&2
  exit 2
fi
for numeric_setting in jobs benchmark_blocks benchmark_samples \
  benchmark_phenotypes benchmark_repeats benchmark_warmup_repeats \
  stream_chunk_mb; do
  numeric_value="${!numeric_setting}"
  if [[ ! "${numeric_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${numeric_setting} must be a positive integer (received '${numeric_value}')" >&2
    exit 2
  fi
done
if [[ ! "${resident_mb}" =~ ^[0-9]+$ ]]; then
  echo "resident_mb must be a non-negative integer (received '${resident_mb}')" >&2
  exit 2
fi
if [[ "${gram_precision}" != "fp64" && "${gram_precision}" != "fp32" ]]; then
  echo "gram_precision must be fp64 or fp32 (received '${gram_precision}')" >&2
  exit 2
fi
if [[ ! "${fp32_gram_chunk_samples}" =~ ^[1-9][0-9]*$ ]]; then
  echo "fp32_gram_chunk_samples must be a positive integer (received '${fp32_gram_chunk_samples}')" >&2
  exit 2
fi
if [[ ! "${run_synthetic_benchmark}" =~ ^[01]$ ]]; then
  echo "RUN_SYNTHETIC_BENCHMARK must be 0 or 1" >&2
  exit 2
fi
if [[ ! "${pgen_packed}" =~ ^[01]$ ]]; then
  echo "STEP1_PGEN_PACKED must be 0 or 1" >&2
  exit 2
fi
if [[ "${run_synthetic_benchmark}" == "1" ]]; then
  for numeric_setting in synthetic_samples synthetic_variants \
    synthetic_phenotypes synthetic_chromosomes synthetic_bsize \
    synthetic_threads synthetic_seed; do
    numeric_value="${!numeric_setting}"
    if [[ ! "${numeric_value}" =~ ^[1-9][0-9]*$ ]]; then
      echo "${numeric_setting} must be a positive integer (received '${numeric_value}')" >&2
      exit 2
    fi
  done
  if (( synthetic_chromosomes > synthetic_variants )); then
    echo "synthetic_chromosomes must not exceed synthetic_variants" >&2
    exit 2
  fi
  if [[ ! "${synthetic_max_bed_gb}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] ||
    ! awk -v value="${synthetic_max_bed_gb}" 'BEGIN { exit !(value > 0) }'; then
    echo "synthetic_max_bed_gb must be positive (received '${synthetic_max_bed_gb}')" >&2
    exit 2
  fi
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
echo "STEP1_CUDA_VALIDATION_CONFIG architectures=${cuda_architectures} \
label=${validation_label} device=${device} validation_dir=${validation_dir} \
benchmark_blocks=${benchmark_blocks} benchmark_samples=${benchmark_samples} \
benchmark_phenotypes=${benchmark_phenotypes} \
benchmark_repeats=${benchmark_repeats} \
benchmark_warmup_repeats=${benchmark_warmup_repeats} \
stream_chunk_mb=${stream_chunk_mb} \
resident_mb=${resident_mb} \
level1_resident_mb=${level1_resident_mb:-auto} \
gram_precision=${gram_precision} \
fp32_gram_chunk_samples=${fp32_gram_chunk_samples} \
pinned_staging_mb=${pinned_staging_mb} pgen_prefetch_mb=${pgen_prefetch_mb} pgen_tile_variants=${pgen_tile_variants} pgen_packed=${pgen_packed} \
run_synthetic_benchmark=${run_synthetic_benchmark} \
synthetic_samples=${synthetic_samples} synthetic_variants=${synthetic_variants} \
synthetic_phenotypes=${synthetic_phenotypes} \
synthetic_chromosomes=${synthetic_chromosomes} synthetic_bsize=${synthetic_bsize} \
synthetic_threads=${synthetic_threads} synthetic_seed=${synthetic_seed} \
synthetic_max_bed_gb=${synthetic_max_bed_gb}"
export REGENIE_CUDA_RESIDENT_MB="${resident_mb}"
if [[ -n "${level1_resident_mb}" ]]; then
  export REGENIE_CUDA_LEVEL1_RESIDENT_MB="${level1_resident_mb}"
else
  unset REGENIE_CUDA_LEVEL1_RESIDENT_MB
fi
export REGENIE_CUDA_GRAM_PRECISION="${gram_precision}"
export REGENIE_CUDA_FP32_GRAM_CHUNK_SAMPLES="${fp32_gram_chunk_samples}"
export REGENIE_CUDA_PINNED_STAGING_MB="${pinned_staging_mb}"
export REGENIE_STEP1_PGEN_PREFETCH_MB="${pgen_prefetch_mb}"
export REGENIE_STEP1_PGEN_TILE_VARIANTS="${pgen_tile_variants}"
export REGENIE_STEP1_PGEN_PACKED="${pgen_packed}"
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version \
    --format=csv,noheader; then
    nvidia-smi
  fi
fi

cmake -S "${repo_root}" -B "${build_dir}" \
  -DREGENIE_WITH_CUDA=ON \
  -DREGENIE_CUDA_ARCHITECTURES="${cuda_architectures}" \
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
  grep -q '^STEP1_BACKEND_TEST case=packed_hardcall_preprocessing supported=1 .* status=PASS$' \
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

if [[ "${RUN_COMPUTE_SANITIZER:-1}" != "0" ]]; then
  if ! command -v compute-sanitizer >/dev/null 2>&1; then
    echo "compute-sanitizer is required unless RUN_COMPUTE_SANITIZER=0" >&2
    exit 2
  fi
  REGENIE_CUDA_CHUNK_MB="${stream_chunk_mb}" \
    compute-sanitizer --tool memcheck --error-exitcode 99 \
    "${build_dir}/step1_compute_test" --backend cuda --device "${device}"
fi

"${build_dir}/step1_compute_test" --backend cpu --benchmark \
  --blocks "${benchmark_blocks}" --samples "${benchmark_samples}" \
  --phenotypes "${benchmark_phenotypes}" --repeats "${benchmark_repeats}" \
  --warmup-repeats "${benchmark_warmup_repeats}"
run_with_memory_log backend_benchmark \
  env REGENIE_CUDA_CHUNK_MB="${stream_chunk_mb}" \
  "${build_dir}/step1_compute_test" --backend cuda --device "${device}" --benchmark \
  --blocks "${benchmark_blocks}" --samples "${benchmark_samples}" \
  --phenotypes "${benchmark_phenotypes}" --repeats "${benchmark_repeats}" \
  --warmup-repeats "${benchmark_warmup_repeats}"

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

binary_kfold_prefix="${validation_dir}/binary_kfold"
python3 "${repo_root}/scripts/generate_step1_bed.py" \
  --prefix "${binary_kfold_prefix}" \
  --samples 6000 \
  --variants 1000 \
  --phenotypes 2 \
  --chromosomes 2 \
  --seed 20260714 \
  --max-bed-gb 1 \
  --trait-type bt \
  --missingness-profile none

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

binary_kfold_common_args=(
  --step 1
  --bed "${binary_kfold_prefix}"
  --covarFile "${binary_kfold_prefix}.covar"
  --phenoFile "${binary_kfold_prefix}.pheno"
  --bt
  --bsize 100
  --threads 1
  --seed 20260714
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
  grep -q "^STEP1_PROFILE version=7 backend=cuda mode=${profile_mode} " "${cuda_prefix}.log"
  grep -q '^STEP1_PROFILE_FINAL version=1 backend=cuda ' "${cuda_prefix}.log"

  compare_loco_files "${label}" "${cpu_prefix}" "${cuda_prefix}"
}

compare_loco_files() {
  local label="$1"
  local cpu_prefix="$2"
  local cuda_prefix="$3"
  loco_files_compared=0
  loco_numeric_values=0
  loco_maximum_absolute_error=0
  shopt -s nullglob
  local cpu_loco_files=("${cpu_prefix}"_*.loco)
  if (( ${#cpu_loco_files[@]} == 0 )); then
    echo "No CPU LOCO files were produced for ${label}" >&2
    exit 1
  fi
  local cpu_file suffix cuda_file compare_output compare_values compare_error
  for cpu_file in "${cpu_loco_files[@]}"; do
    suffix="${cpu_file#${cpu_prefix}_}"
    cuda_file="${cuda_prefix}_${suffix}"
    compare_output="$(python3 "${repo_root}/scripts/compare_numeric_files.py" \
      "${cpu_file}" "${cuda_file}" \
      --rtol "${LOCO_RTOL:-1e-7}" --atol "${LOCO_ATOL:-1e-9}")"
    printf '%s\n' "${compare_output}"
    compare_values="$(awk '{
      for (field = 1; field <= NF; ++field) {
        if ($field ~ /^values=/) {
          sub(/^values=/, "", $field)
          print $field
          exit
        }
      }
    }' <<<"${compare_output}")"
    compare_error="$(awk '{
      for (field = 1; field <= NF; ++field) {
        if ($field ~ /^maximum_absolute_error=/) {
          sub(/^maximum_absolute_error=/, "", $field)
          print $field
          exit
        }
      }
    }' <<<"${compare_output}")"
    if [[ ! "${compare_values}" =~ ^[0-9]+$ ]] || [[ -z "${compare_error}" ]]; then
      echo "Could not parse numeric comparison result for ${label}" >&2
      exit 1
    fi
    loco_files_compared=$((loco_files_compared + 1))
    loco_numeric_values=$((loco_numeric_values + compare_values))
    loco_maximum_absolute_error="$(awk \
      -v current="${loco_maximum_absolute_error}" -v candidate="${compare_error}" \
      'BEGIN { print (candidate > current ? candidate : current) }')"
  done
}

elapsed_seconds() {
  local log_file="$1"
  awk '
    /^Elapsed time : / {
      value = $4
      sub(/s$/, "", value)
      elapsed = value
    }
    END {
      if (elapsed == "") exit 1
      print elapsed
    }
  ' "${log_file}"
}

run_synthetic_end_to_end_benchmark() {
  local synthetic_case="n${synthetic_samples}_m${synthetic_variants}_p${synthetic_phenotypes}_s${synthetic_seed}"
  local synthetic_dir="${validation_dir}/synthetic/${synthetic_case}"
  local synthetic_prefix="${synthetic_dir}/step1"
  local cpu_prefix="${synthetic_dir}/cpu_kfold"
  local cuda_prefix="${synthetic_dir}/cuda_kfold"

  python3 "${repo_root}/scripts/generate_step1_bed.py" \
    --prefix "${synthetic_prefix}" \
    --samples "${synthetic_samples}" \
    --variants "${synthetic_variants}" \
    --phenotypes "${synthetic_phenotypes}" \
    --chromosomes "${synthetic_chromosomes}" \
    --seed "${synthetic_seed}" \
    --max-bed-gb "${synthetic_max_bed_gb}"

  local synthetic_common_args=(
    --step 1
    --bed "${synthetic_prefix}"
    --covarFile "${synthetic_prefix}.covar"
    --phenoFile "${synthetic_prefix}.pheno"
    --qt
    --bsize "${synthetic_bsize}"
    --threads "${synthetic_threads}"
    --seed "${synthetic_seed}"
    --step1-profile
  )

  "${build_dir}/regenie" "${synthetic_common_args[@]}" \
    --compute-backend cpu --out "${cpu_prefix}"
  run_with_memory_log synthetic_kfold \
    env REGENIE_CUDA_CHUNK_MB="${stream_chunk_mb}" \
    "${build_dir}/regenie" "${synthetic_common_args[@]}" \
    --compute-backend cuda --gpu-device "${device}" --out "${cuda_prefix}"

  grep -Fq 'Step 1 compute backend : [cuda]' "${cuda_prefix}.log"
  grep -q '^STEP1_PROFILE version=7 backend=cuda mode=kfold ' "${cuda_prefix}.log"
  grep -q '^STEP1_PROFILE_FINAL version=1 backend=cuda ' "${cuda_prefix}.log"
  compare_loco_files synthetic_kfold "${cpu_prefix}" "${cuda_prefix}"

  local cpu_elapsed_s
  local cuda_elapsed_s
  local speedup
  local peak_mib
  cpu_elapsed_s="$(elapsed_seconds "${cpu_prefix}.log")"
  cuda_elapsed_s="$(elapsed_seconds "${cuda_prefix}.log")"
  speedup="$(awk -v cpu="${cpu_elapsed_s}" -v cuda="${cuda_elapsed_s}" \
    'BEGIN { if (cuda <= 0) exit 1; printf "%.6g", cpu / cuda }')"
  peak_mib="$(awk '
    BEGIN { peak = 0 }
    /^[[:space:]]*[0-9]+([.][0-9]+)?[[:space:]]*$/ {
      value = $1 + 0
      if (value > peak) peak = value
    }
    END { print peak }
  ' "${validation_dir}/memory_synthetic_kfold.txt")"
  echo "STEP1_SYNTHETIC_BENCHMARK status=PASS samples=${synthetic_samples} \
variants=${synthetic_variants} phenotypes=${synthetic_phenotypes} \
chromosomes=${synthetic_chromosomes} bsize=${synthetic_bsize} \
threads=${synthetic_threads} seed=${synthetic_seed} \
cpu_elapsed_s=${cpu_elapsed_s} cuda_elapsed_s=${cuda_elapsed_s} speedup=${speedup} \
peak_mib=${peak_mib} loco_files=${loco_files_compared} \
loco_numeric_values=${loco_numeric_values} \
maximum_absolute_error=${loco_maximum_absolute_error} \
data_prefix=${synthetic_prefix}"
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
run_end_to_end_pair binary_kfold kfold "${binary_kfold_common_args[@]}"
run_end_to_end_pair binary_loocv loocv "${binary_common_args[@]}" --loocv
run_end_to_end_pair binary_full loocv \
  "${binary_common_args[@]}" --loocv --l1-full
run_end_to_end_pair t2e_kfold kfold "${t2e_common_args[@]}"
if [[ "${run_synthetic_benchmark}" == "1" ]]; then
  run_synthetic_end_to_end_benchmark
fi

echo "STEP1_CUDA_VALIDATION status=PASS device=${device} architectures=${cuda_architectures} label=${validation_label} results=${validation_dir}"
