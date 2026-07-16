#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

profile="${CUDA_PROFILE:-datacenter}"
cuda_architectures="${CUDA_ARCHITECTURES:-}"
cpu_architecture="${CPU_ARCHITECTURE:-x86-64-v3}"
cpu_tune="${CPU_TUNE:-generic}"
jobs="${JOBS:-}"
cuda_validation="${CUDA_VALIDATION:-auto}"
build_root="${BUILD_ROOT:-}"
output_dir="${OUTPUT_DIR:-}"
clean=0
clean_dependencies=0

usage() {
  cat <<'USAGE'
Build a complete, upload-ready Linux REGENIE release set.

Usage:
  scripts/build/build_linux_release_bundle.sh [options]

Options:
  --profile NAME                 CUDA profile passed to the CUDA builder
                                 (default: datacenter).
  --cuda-architectures LIST      Explicit semicolon-separated CUDA targets.
  --cpu-architecture ARCH        Shared GCC/NVCC host -march target
                                 (default: x86-64-v3).
  --cpu-tune CPU                 Shared GCC/NVCC host -mtune target
                                 (default: generic).
  --native-cpu                   Use -march=native -mtune=native; the artifacts
                                 may not run on a different host CPU.
  --jobs N                       Parallel build jobs (default: nproc).
  --cuda-validation MODE         auto, always, or never (default: auto).
  --build-root PATH              Work and log directory.
  --output-dir PATH              Upload-ready release directory.
  --clean                        Remove the selected work and output directories.
  --clean-dependencies           Rebuild managed BGEN from its verified archive.
  -h, --help                     Show this help.

The command always uses checksum-verified managed BGEN, builds CPU and CUDA
artifacts from the same committed source revision, validates both extracted
packages, and writes release-manifest.json plus SHA256SUMS beside the assets.
USAGE
}

die() {
  echo "LINUX_RELEASE_BUNDLE_ERROR: $*" >&2
  exit 1
}

require_value() {
  if (( $# < 2 )) || [[ -z "$2" ]]; then
    die "$1 requires a value"
  fi
}

while (( $# > 0 )); do
  case "$1" in
    --profile)
      require_value "$@"
      profile="$2"
      shift 2
      ;;
    --cuda-architectures)
      require_value "$@"
      cuda_architectures="$2"
      shift 2
      ;;
    --cpu-architecture)
      require_value "$@"
      cpu_architecture="$2"
      shift 2
      ;;
    --cpu-tune)
      require_value "$@"
      cpu_tune="$2"
      shift 2
      ;;
    --native-cpu)
      cpu_architecture="native"
      cpu_tune="native"
      shift
      ;;
    --jobs)
      require_value "$@"
      jobs="$2"
      shift 2
      ;;
    --cuda-validation)
      require_value "$@"
      cuda_validation="$2"
      shift 2
      ;;
    --build-root)
      require_value "$@"
      build_root="$2"
      shift 2
      ;;
    --output-dir)
      require_value "$@"
      output_dir="$2"
      shift 2
      ;;
    --clean)
      clean=1
      shift
      ;;
    --clean-dependencies)
      clean_dependencies=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option '$1' (use --help)"
      ;;
  esac
done

for command_name in awk cmake env find git grep nproc python3 tee tr uname wc; do
  command -v "${command_name}" >/dev/null 2>&1 ||
    die "required command '${command_name}' was not found"
done

[[ "$(uname -s)" == "Linux" ]] || die "Linux release bundles require Linux"
[[ "$(uname -m)" == "x86_64" ]] || die "Linux release bundles require x86-64"
if [[ -z "${jobs}" ]]; then
  jobs="$(nproc)"
fi
[[ "${jobs}" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
case "${cuda_validation}" in
  auto|always|never) ;;
  *) die "--cuda-validation must be auto, always, or never" ;;
esac

if ! git -C "${repo_root}" diff --quiet ||
   ! git -C "${repo_root}" diff --cached --quiet; then
  die "tracked source changes are present; commit or stash them before building a release set"
fi

git_commit="$(git -C "${repo_root}" rev-parse HEAD)"
git_short="$(git -C "${repo_root}" rev-parse --short=12 HEAD)"
version="$(tr -d '[:space:]' < "${repo_root}/VERSION")"
if [[ -z "${build_root}" ]]; then
  build_root="${repo_root}/build/linux-release-g${git_short}"
fi
if [[ -z "${output_dir}" ]]; then
  output_dir="${repo_root}/dist/regenie-${version}-g${git_short}-linux-release"
fi

if (( clean == 1 )); then
  cmake -E remove_directory "${build_root}"
  cmake -E remove_directory "${output_dir}"
fi
mkdir -p "${build_root}/logs" "${output_dir}"
if find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  die "output directory is not empty; select another directory or pass --clean"
fi

echo "LINUX_RELEASE_BUNDLE_BEGIN"
echo "LINUX_RELEASE_BUNDLE_COMMIT=${git_commit}"
echo "LINUX_RELEASE_BUNDLE_VERSION=${version}"
echo "LINUX_RELEASE_BUNDLE_CPU architecture=${cpu_architecture} tune=${cpu_tune}"
echo "LINUX_RELEASE_BUNDLE_CUDA_PROFILE=${profile}"
echo "LINUX_RELEASE_BUNDLE_OUTPUT=${output_dir}"

cpu_args=(
  --cpu-architecture "${cpu_architecture}"
  --cpu-tune "${cpu_tune}"
  --build-dir "${build_root}/cpu"
  --output-dir "${output_dir}"
  --jobs "${jobs}"
  --clean
)
if (( clean_dependencies == 1 )); then
  cpu_args+=(--clean-dependencies)
fi

set +e
env -u BGEN_PATH -u SOURCE_DATE_EPOCH \
  "${script_dir}/build_cpu_release.sh" "${cpu_args[@]}" \
  2>&1 | tee "${build_root}/logs/cpu.log"
cpu_status=${PIPESTATUS[0]}
set -e
(( cpu_status == 0 )) || die "CPU release build failed with status ${cpu_status}"

cuda_args=(
  --profile "${profile}"
  --cpu-architecture "${cpu_architecture}"
  --cpu-tune "${cpu_tune}"
  --build-dir "${build_root}/cuda"
  --output-dir "${output_dir}"
  --jobs "${jobs}"
  --cuda-validation "${cuda_validation}"
  --clean
)
if [[ -n "${cuda_architectures}" ]]; then
  cuda_args+=(--cuda-architectures "${cuda_architectures}")
fi

set +e
env -u BGEN_PATH -u SOURCE_DATE_EPOCH \
  "${script_dir}/build_cuda_release.sh" "${cuda_args[@]}" \
  2>&1 | tee "${build_root}/logs/cuda.log"
cuda_status=${PIPESTATUS[0]}
set -e
(( cuda_status == 0 )) || die "CUDA release build failed with status ${cuda_status}"

python3 "${script_dir}/verify_release_assets.py" \
  "${output_dir}" \
  --expected-commit "${git_commit}" \
  --require-kind cpu-mkl \
  --require-kind cuda \
  --require-verified-dependencies \
  --manifest "${output_dir}/release-manifest.json" \
  --sha256sums "${output_dir}/SHA256SUMS" \
  | tee "${build_root}/logs/verification.log"

asset_count="$(find "${output_dir}" -maxdepth 1 -name '*.tar.gz' | wc -l | awk '{print $1}')"
echo "LINUX_RELEASE_BUNDLE_STATUS=PASS"
echo "LINUX_RELEASE_BUNDLE_ASSETS=${asset_count}"
echo "LINUX_RELEASE_BUNDLE_MANIFEST=${output_dir}/release-manifest.json"
echo "LINUX_RELEASE_BUNDLE_SHA256SUMS=${output_dir}/SHA256SUMS"
echo "LINUX_RELEASE_BUNDLE_OUTPUT=${output_dir}"
echo "LINUX_RELEASE_BUNDLE_END"
