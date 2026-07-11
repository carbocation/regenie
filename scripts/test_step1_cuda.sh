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
validation_dir="${build_dir}/a100-validation"

nvcc --version
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

"${build_dir}/step1_compute_test" --backend cpu
"${build_dir}/step1_compute_test" --backend cuda --device "${device}"
auto_output="$("${build_dir}/step1_compute_test" --backend auto --device "${device}")"
printf '%s\n' "${auto_output}"
grep -q '^STEP1_BACKEND_TEST backend=cuda status=PASS$' <<<"${auto_output}"

"${build_dir}/step1_compute_test" --backend cpu --benchmark \
  --blocks "${benchmark_blocks}" --samples "${benchmark_samples}" \
  --phenotypes "${benchmark_phenotypes}" --repeats "${benchmark_repeats}"
"${build_dir}/step1_compute_test" --backend cuda --device "${device}" --benchmark \
  --blocks "${benchmark_blocks}" --samples "${benchmark_samples}" \
  --phenotypes "${benchmark_phenotypes}" --repeats "${benchmark_repeats}"

mkdir -p "${validation_dir}"
common_args=(
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

run_end_to_end_pair() {
  local mode="$1"
  shift
  local cpu_prefix="${validation_dir}/cpu_${mode}"
  local cuda_prefix="${validation_dir}/cuda_${mode}"

  "${build_dir}/regenie" "${common_args[@]}" "$@" \
    --compute-backend cpu --out "${cpu_prefix}"
  "${build_dir}/regenie" "${common_args[@]}" "$@" \
    --compute-backend cuda --gpu-device "${device}" --out "${cuda_prefix}"

  grep -Fq 'Step 1 compute backend : [cuda]' "${cuda_prefix}.log"
  grep -q "^STEP1_PROFILE version=2 backend=cuda mode=${mode} " "${cuda_prefix}.log"

  shopt -s nullglob
  local cpu_loco_files=("${cpu_prefix}"_*.loco)
  if (( ${#cpu_loco_files[@]} == 0 )); then
    echo "No CPU LOCO files were produced for ${mode}" >&2
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

run_end_to_end_pair kfold
run_end_to_end_pair loocv --loocv

echo "STEP1_CUDA_VALIDATION status=PASS device=${device} results=${validation_dir}"
