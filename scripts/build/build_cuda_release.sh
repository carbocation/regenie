#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

profile="${CUDA_PROFILE:-datacenter}"
cuda_architectures="${CUDA_ARCHITECTURES:-${REGENIE_CUDA_ARCHITECTURES:-}}"
cpu_architecture="${CPU_ARCHITECTURE:-x86-64-v3}"
cpu_tune="${CPU_TUNE:-generic}"
build_dir="${BUILD_DIR:-}"
output_dir="${OUTPUT_DIR:-${repo_root}/dist}"
jobs="${JOBS:-}"
run_tests=1
cuda_validation="${CUDA_VALIDATION:-auto}"
clean_build=0

usage() {
  cat <<'USAGE'
Build and package a portable, performance-oriented CUDA-enabled REGENIE binary.

Usage:
  scripts/build/build_cuda_release.sh [options]

CUDA profiles:
  datacenter       T4, A100/A30, A10/A40, L4/L40, H100/H200 (default)
                   CUDA architectures: 75;80;86;89;90
  datacenter-v100  The datacenter profile plus V100 (requires CUDA 12.x)
                   CUDA architectures: 70;75;80;86;89;90
  tested           T4 and A100 only
                   CUDA architectures: 75;80

Options:
  --profile NAME                 Select a CUDA profile.
  --cuda-architectures LIST      Override the profile with a semicolon-separated
                                 list such as '75;80;89;90'.
  --cpu-architecture ARCH        GCC -march target (default: x86-64-v3).
  --cpu-tune CPU                 GCC -mtune target (default: generic).
  --native-cpu                   Use -march=native -mtune=native. The artifact
                                 may not run on a different host CPU.
  --build-dir PATH               CMake build directory.
  --output-dir PATH              Package output directory (default: ./dist).
  --jobs N                       Parallel build jobs (default: nproc).
  --cuda-validation MODE         auto, always, or never (default: auto).
  --skip-tests                   Skip CTest and backend unit tests.
  --clean                        Remove the selected CMake build directory first.
  -h, --help                     Show this help.

Environment equivalents:
  CUDA_PROFILE, CUDA_ARCHITECTURES, CPU_ARCHITECTURE, CPU_TUNE, BUILD_DIR,
  OUTPUT_DIR, JOBS, CUDA_VALIDATION, BGEN_PATH, MKLROOT, SOURCE_DATE_EPOCH.

The script expects a Linux x86-64 build host with a CUDA toolkit and static
oneMKL development libraries. If BGEN_PATH is unset, BGEN v1.1.7 is downloaded
and built under ${XDG_CACHE_HOME:-$HOME/.cache}/regenie-build-deps.
USAGE
}

die() {
  echo "CUDA_RELEASE_BUILD_ERROR: $*" >&2
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
    --build-dir)
      require_value "$@"
      build_dir="$2"
      shift 2
      ;;
    --output-dir)
      require_value "$@"
      output_dir="$2"
      shift 2
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
    --skip-tests)
      run_tests=0
      shift
      ;;
    --clean)
      clean_build=1
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

case "${profile}" in
  datacenter|gcp)
    profile_architectures="75;80;86;89;90"
    ;;
  datacenter-v100|gcp-v100)
    profile_architectures="70;75;80;86;89;90"
    ;;
  tested)
    profile_architectures="75;80"
    ;;
  *)
    die "unknown CUDA profile '${profile}'"
    ;;
esac
if [[ -z "${cuda_architectures}" ]]; then
  cuda_architectures="${profile_architectures}"
fi

[[ "$(uname -s)" == "Linux" ]] || die "release builds currently require Linux"
[[ "$(uname -m)" == "x86_64" ]] || die "release builds currently require x86-64"
[[ "${cuda_architectures}" =~ ^[0-9]+([;][0-9]+)*$ ]] ||
  die "CUDA architectures must be semicolon-separated integers"
[[ "${jobs:-}" =~ ^[1-9][0-9]*$ ]] || {
  if [[ -n "${jobs}" ]]; then
    die "--jobs must be a positive integer"
  fi
  jobs="$(nproc)"
}
case "${cuda_validation}" in
  auto|always|never) ;;
  *) die "--cuda-validation must be auto, always, or never" ;;
esac

if ! git -C "${repo_root}" diff --quiet ||
   ! git -C "${repo_root}" diff --cached --quiet; then
  die "tracked source changes are present; commit or stash them before making a release artifact"
fi

for command_name in cmake ctest g++ gfortran git gzip make nproc nvcc python3 readelf tar; do
  command -v "${command_name}" >/dev/null 2>&1 ||
    die "required command '${command_name}' was not found"
done
if command -v sha256sum >/dev/null 2>&1; then
  sha256_command=(sha256sum)
elif command -v shasum >/dev/null 2>&1; then
  sha256_command=(shasum -a 256)
else
  die "sha256sum or shasum is required"
fi

if [[ -z "${MKLROOT:-}" && -f /opt/intel/oneapi/setvars.sh ]]; then
  # Intel's environment scripts inspect optional variables without guarding
  # every expansion. Limit nounset relaxation to this third-party boundary.
  set +u
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/setvars.sh >/dev/null
  set -u
fi
[[ -n "${MKLROOT:-}" ]] ||
  die "set MKLROOT or install oneMKL under /opt/intel/oneapi"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"

mkl_lib_dir=""
for candidate in "${MKLROOT}/lib/intel64" "${MKLROOT}/lib"; do
  if [[ -f "${candidate}/libmkl_intel_lp64.a" &&
        -f "${candidate}/libmkl_gnu_thread.a" &&
        -f "${candidate}/libmkl_core.a" ]]; then
    mkl_lib_dir="${candidate}"
    break
  fi
done
[[ -n "${mkl_lib_dir}" ]] ||
  die "static oneMKL archives were not found beneath MKLROOT=${MKLROOT}"

nvcc_supported="$(nvcc --list-gpu-arch 2>/dev/null || true)"
if [[ -n "${nvcc_supported}" ]]; then
  IFS=';' read -r -a requested_architectures <<<"${cuda_architectures}"
  for architecture in "${requested_architectures[@]}"; do
    if ! grep -qx "compute_${architecture}" <<<"${nvcc_supported}"; then
      die "nvcc does not support CUDA architecture ${architecture}; supported targets: $(tr '\n' ' ' <<<"${nvcc_supported}")"
    fi
  done
fi

if [[ -z "${BGEN_PATH:-}" ]]; then
  deps_dir="${XDG_CACHE_HOME:-$HOME/.cache}/regenie-build-deps"
  BGEN_PATH="${deps_dir}/v1.1.7"
  bgen_archive="${deps_dir}/v1.1.7.tgz"
  mkdir -p "${deps_dir}"
  if [[ ! -f "${BGEN_PATH}/build/libbgen.a" ]]; then
    if [[ ! -f "${bgen_archive}" ]]; then
      if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 3 --output "${bgen_archive}" \
          http://code.enkre.net/bgen/tarball/release/v1.1.7
      elif command -v wget >/dev/null 2>&1; then
        wget --output-document="${bgen_archive}" \
          http://code.enkre.net/bgen/tarball/release/v1.1.7
      else
        die "curl or wget is required to download BGEN"
      fi
    fi
    tar -xzf "${bgen_archive}" -C "${deps_dir}"
    (
      cd "${BGEN_PATH}"
      python3 waf configure
      python3 waf
    )
  fi
fi
export BGEN_PATH
[[ -f "${BGEN_PATH}/build/libbgen.a" ]] ||
  die "BGEN_PATH does not contain a built BGEN v1.1.7 library: ${BGEN_PATH}"

cpu_label="${cpu_architecture//[^[:alnum:]._-]/-}"
architecture_label="$(tr ';' '-' <<<"${cuda_architectures}" | sed 's/[0-9][0-9]*/sm&/g')"
if [[ -z "${build_dir}" ]]; then
  build_dir="${repo_root}/build/cuda-release-${profile}-${cpu_label}"
fi
if (( clean_build == 1 )) && [[ -d "${build_dir}" ]]; then
  cmake -E remove_directory "${build_dir}"
fi
mkdir -p "${build_dir}" "${output_dir}"

version="$(tr -d '[:space:]' < "${repo_root}/VERSION")"
git_commit="$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || echo unknown)"
git_short="$(git -C "${repo_root}" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
git_describe="$(git -C "${repo_root}" describe --always --dirty 2>/dev/null || echo unknown)"
source_date_epoch="${SOURCE_DATE_EPOCH:-$(git -C "${repo_root}" log -1 --format=%ct 2>/dev/null || date +%s)}"
[[ "${source_date_epoch}" =~ ^[0-9]+$ ]] ||
  die "SOURCE_DATE_EPOCH must be a non-negative integer"
artifact_base="regenie-${version}-g${git_short}-linux-x86_64-${cpu_label}-cuda-${architecture_label}"
stage_root="${build_dir}/package"
stage_dir="${stage_root}/${artifact_base}"
archive_path="${output_dir}/${artifact_base}.tar.gz"

echo "CUDA_RELEASE_BUILD_BEGIN"
echo "CUDA_RELEASE_BUILD_SOURCE=${repo_root}"
echo "CUDA_RELEASE_BUILD_COMMIT=${git_commit}"
echo "CUDA_RELEASE_BUILD_PROFILE=${profile}"
echo "CUDA_RELEASE_BUILD_ARCHITECTURES=${cuda_architectures}"
echo "CUDA_RELEASE_BUILD_CPU architecture=${cpu_architecture} tune=${cpu_tune}"
echo "CUDA_RELEASE_BUILD_TOOLKIT=$(nvcc --version | tail -n 1)"
echo "CUDA_RELEASE_BUILD_COMPILER=$(g++ --version | head -n 1)"
echo "CUDA_RELEASE_BUILD_BGEN=${BGEN_PATH}"
echo "CUDA_RELEASE_BUILD_MKL=${MKLROOT}"

export STATIC=1
if (( run_tests == 1 )); then
  cmake_build_testing=ON
else
  cmake_build_testing=OFF
fi
cmake -S "${repo_root}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  "-DBUILD_TESTING=${cmake_build_testing}" \
  -DREGENIE_WITH_CUDA=ON \
  "-DREGENIE_CUDA_ARCHITECTURES=${cuda_architectures}" \
  "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -march=${cpu_architecture} -mtune=${cpu_tune}"
build_targets=(regenie)
if (( run_tests == 1 )); then
  build_targets+=(step1_compute_test cox_firth_test)
fi
cmake --build "${build_dir}" \
  --target "${build_targets[@]}" \
  --parallel "${jobs}"

if (( run_tests == 1 )); then
  (
    cd "${build_dir}"
    ctest --output-on-failure
  )
  "${build_dir}/step1_compute_test" --backend cpu

  run_cuda_validation=0
  if [[ "${cuda_validation}" == "always" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 &&
       nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
      run_cuda_validation=1
    else
      die "CUDA validation was requested but no usable NVIDIA GPU was found"
    fi
  elif [[ "${cuda_validation}" == "auto" ]] &&
       command -v nvidia-smi >/dev/null 2>&1 &&
       nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
    run_cuda_validation=1
  fi
  if (( run_cuda_validation == 1 )); then
    CUDA_ARCHITECTURES="${cuda_architectures}" \
    BUILD_DIR="${build_dir}" \
    JOBS="${jobs}" \
      "${repo_root}/scripts/test_step1_cuda.sh"
  elif [[ "${cuda_validation}" == "never" ]]; then
    echo "CUDA_RELEASE_BUILD_CUDA_VALIDATION=SKIPPED reason=disabled"
  else
    echo "CUDA_RELEASE_BUILD_CUDA_VALIDATION=SKIPPED reason=no_usable_gpu"
  fi
else
  echo "CUDA_RELEASE_BUILD_TESTS=SKIPPED"
fi

binary_path="${build_dir}/regenie"
[[ -x "${binary_path}" ]] || die "built executable is missing: ${binary_path}"
dynamic_entries="$(readelf -d "${binary_path}")"
if grep -Eq 'NEEDED.*lib(mkl|stdc\+\+|gcc_s)' <<<"${dynamic_entries}"; then
  die "the release binary unexpectedly has dynamic oneMKL or GNU C++ runtime dependencies"
fi
for cuda_library in libcudart libcublas libcusolver; do
  grep -q "NEEDED.*${cuda_library}" <<<"${dynamic_entries}" ||
    die "the release binary does not record the expected ${cuda_library} dependency"
done
if command -v ldd >/dev/null 2>&1 && ldd "${binary_path}" | grep -q 'not found'; then
  ldd "${binary_path}" >&2
  die "the release binary has unresolved shared-library dependencies on the build host"
fi

cmake -E remove_directory "${stage_root}"
mkdir -p "${stage_dir}/bin"
cp "${binary_path}" "${stage_dir}/bin/regenie"
cp "${repo_root}/LICENSE" "${repo_root}/README.md" "${repo_root}/VERSION" \
  "${stage_dir}/"
{
  printf 'REGENIE_VERSION=%s\n' "${version}"
  printf 'GIT_COMMIT=%s\n' "${git_commit}"
  printf 'GIT_DESCRIBE=%s\n' "${git_describe}"
  printf 'CUDA_PROFILE=%s\n' "${profile}"
  printf 'CUDA_ARCHITECTURES=%s\n' "${cuda_architectures}"
  printf 'CPU_ARCHITECTURE=%s\n' "${cpu_architecture}"
  printf 'CPU_TUNE=%s\n' "${cpu_tune}"
  printf 'NVCC=%s\n' "$(nvcc --version | tail -n 1)"
  printf 'CXX=%s\n' "$(g++ --version | head -n 1)"
  printf 'CMAKE=%s\n' "$(cmake --version | head -n 1)"
  printf 'MKLROOT=%s\n' "${MKLROOT}"
  printf 'BGEN_PATH=%s\n' "${BGEN_PATH}"
  printf 'SOURCE_DATE_EPOCH=%s\n' "${source_date_epoch}"
} > "${stage_dir}/BUILD-METADATA.txt"

archive_tmp="${archive_path}.tmp"
tar --sort=name \
  --mtime="@${source_date_epoch}" \
  --owner=0 --group=0 --numeric-owner \
  -C "${stage_root}" -cf - "${artifact_base}" |
  gzip -n > "${archive_tmp}"
mv "${archive_tmp}" "${archive_path}"
(
  cd "${output_dir}"
  "${sha256_command[@]}" "$(basename "${archive_path}")" \
    > "$(basename "${archive_path}").sha256"
)

echo "CUDA_RELEASE_BUILD_STATUS=PASS"
echo "CUDA_RELEASE_BUILD_BINARY=${binary_path}"
echo "CUDA_RELEASE_BUILD_ARCHIVE=${archive_path}"
echo "CUDA_RELEASE_BUILD_CHECKSUM=${archive_path}.sha256"
echo "CUDA_RELEASE_BUILD_END"
