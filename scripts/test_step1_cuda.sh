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

cmake -S "${repo_root}" -B "${build_dir}" \
  -DREGENIE_WITH_CUDA=ON \
  -DREGENIE_CUDA_ARCHITECTURES=80 \
  -DBUILD_TESTING=ON
cmake --build "${build_dir}" --target regenie step1_compute_test -j "${jobs}"

"${build_dir}/step1_compute_test" --backend cpu
"${build_dir}/step1_compute_test" --backend cuda --device "${device}"

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
  --loocv
  --bsize 100
  --threads 1
  --step1-profile
)

"${build_dir}/regenie" "${common_args[@]}" \
  --compute-backend cpu --out "${validation_dir}/cpu"
"${build_dir}/regenie" "${common_args[@]}" \
  --compute-backend cuda --gpu-device "${device}" --out "${validation_dir}/cuda"

shopt -s nullglob
cpu_loco_files=("${validation_dir}"/cpu_*.loco)
if (( ${#cpu_loco_files[@]} == 0 )); then
  echo "No CPU LOCO files were produced" >&2
  exit 1
fi
for cpu_file in "${cpu_loco_files[@]}"; do
  cuda_file="${cpu_file%/cpu_*}/cuda_${cpu_file##*/cpu_}"
  python3 "${repo_root}/scripts/compare_numeric_files.py" \
    "${cpu_file}" "${cuda_file}" \
    --rtol "${LOCO_RTOL:-1e-7}" --atol "${LOCO_ATOL:-1e-9}"
done

echo "STEP1_CUDA_VALIDATION status=PASS device=${device} results=${validation_dir}"
