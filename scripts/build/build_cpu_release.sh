#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

cpu_architecture="${CPU_ARCHITECTURE:-x86-64-v3}"
cpu_tune="${CPU_TUNE:-generic}"
build_dir="${BUILD_DIR:-}"
output_dir="${OUTPUT_DIR:-${repo_root}/dist}"
jobs="${JOBS:-}"
run_tests=1
clean_build=0

usage() {
  cat <<'USAGE'
Build and package a portable, performance-oriented CPU-only REGENIE binary.

Usage:
  scripts/build/build_cpu_release.sh [options]

Options:
  --cpu-architecture ARCH        GCC -march target (default: x86-64-v3).
  --cpu-tune CPU                 GCC -mtune target (default: generic).
  --native-cpu                   Use -march=native -mtune=native. The artifact
                                 may not run on a different host CPU.
  --build-dir PATH               CMake build directory.
  --output-dir PATH              Package output directory (default: ./dist).
  --jobs N                       Parallel build jobs (default: nproc).
  --skip-tests                   Skip CTest and the regression suite.
  --clean                        Remove the selected CMake build directory first.
  -h, --help                     Show this help.

Environment equivalents:
  CPU_ARCHITECTURE, CPU_TUNE, BUILD_DIR, OUTPUT_DIR, JOBS, BGEN_PATH, MKLROOT,
  SOURCE_DATE_EPOCH.

The script expects a Linux x86-64 build host with static oneMKL development
libraries. If BGEN_PATH is unset, BGEN v1.1.7 is downloaded and built under
${XDG_CACHE_HOME:-$HOME/.cache}/regenie-build-deps.
USAGE
}

die() {
  echo "CPU_RELEASE_BUILD_ERROR: $*" >&2
  exit 1
}

require_value() {
  if (( $# < 2 )) || [[ -z "$2" ]]; then
    die "$1 requires a value"
  fi
}

while (( $# > 0 )); do
  case "$1" in
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

[[ "$(uname -s)" == "Linux" ]] || die "release builds currently require Linux"
[[ "$(uname -m)" == "x86_64" ]] || die "release builds currently require x86-64"
[[ "${jobs:-}" =~ ^[1-9][0-9]*$ ]] || {
  if [[ -n "${jobs}" ]]; then
    die "--jobs must be a positive integer"
  fi
  jobs="$(nproc)"
}

if ! git -C "${repo_root}" diff --quiet ||
   ! git -C "${repo_root}" diff --cached --quiet; then
  die "tracked source changes are present; commit or stash them before making a release artifact"
fi

for command_name in cmake ctest g++ gfortran git gzip make nproc python3 readelf tar; do
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

deps_dir="${XDG_CACHE_HOME:-$HOME/.cache}/regenie-build-deps"
bgen_compatibility_patch=0
bgen_source_sha256="unverified-external"
bgen_provenance="provided"
if [[ -z "${BGEN_PATH:-}" ]]; then
  BGEN_PATH="${deps_dir}/v1.1.7"
  bgen_archive="${deps_dir}/v1.1.7.tgz"
  bgen_url="https://enkre.net/cgi-bin/code/bgen/tarball/release/v1.1.7"
  bgen_expected_sha256="6476b077af6c8e98e85fd7e09f58cb3fdf143ff91850c984248fd4dc2d74a8c3"
  bgen_stamp="${BGEN_PATH}/build/.regenie-${bgen_expected_sha256}-streampos-v1"
  bgen_source_sha256="${bgen_expected_sha256}"
  mkdir -p "${deps_dir}"
  if [[ ! -f "${BGEN_PATH}/build/libbgen.a" || ! -f "${bgen_stamp}" ]]; then
    if [[ ! -f "${bgen_archive}" ]]; then
      if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 3 --output "${bgen_archive}" \
          "${bgen_url}"
      elif command -v wget >/dev/null 2>&1; then
        wget --output-document="${bgen_archive}" \
          "${bgen_url}"
      else
        die "curl or wget is required to download BGEN"
      fi
    fi
    bgen_actual_sha256="$("${sha256_command[@]}" "${bgen_archive}" | awk '{print $1}')"
    [[ "${bgen_actual_sha256}" == "${bgen_expected_sha256}" ]] ||
      die "BGEN archive checksum mismatch: expected ${bgen_expected_sha256}, received ${bgen_actual_sha256}"
    cmake -E remove_directory "${BGEN_PATH}"
    tar -xzf "${bgen_archive}" -C "${deps_dir}"
    bgen_view="${BGEN_PATH}/src/View.cpp"
    [[ -f "${bgen_view}" ]] || die "verified BGEN archive has no src/View.cpp"
    grep -q 'std::ios::streampos origin' "${bgen_view}" ||
      die "verified BGEN source does not contain the expected compatibility token"
    # BGEN 1.1.7 predates current libstdc++ releases. streampos belongs to
    # namespace std, not the std::ios class; correct only that exact token.
    sed -i 's/std::ios::streampos origin/std::streampos origin/' "${bgen_view}"
    bgen_compatibility_patch=1
    (
      cd "${BGEN_PATH}"
      python3 waf configure
      python3 waf
    )
    cmake -E touch "${bgen_stamp}"
    bgen_provenance="verified-fresh-build"
  else
    bgen_provenance="verified-managed-cache"
  fi
fi
export BGEN_PATH
[[ -f "${BGEN_PATH}/build/libbgen.a" ]] ||
  die "BGEN_PATH does not contain a built BGEN v1.1.7 library: ${BGEN_PATH}"

cpu_label="${cpu_architecture//[^[:alnum:]._-]/-}"
if [[ -z "${build_dir}" ]]; then
  build_dir="${repo_root}/build/cpu-release-${cpu_label}"
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
artifact_base="regenie-${version}-g${git_short}-linux-x86_64-${cpu_label}-cpu-mkl"
stage_root="${build_dir}/package"
stage_dir="${stage_root}/${artifact_base}"
archive_path="${output_dir}/${artifact_base}.tar.gz"

echo "CPU_RELEASE_BUILD_BEGIN"
echo "CPU_RELEASE_BUILD_SOURCE=${repo_root}"
echo "CPU_RELEASE_BUILD_COMMIT=${git_commit}"
echo "CPU_RELEASE_BUILD_CPU architecture=${cpu_architecture} tune=${cpu_tune}"
echo "CPU_RELEASE_BUILD_COMPILER=$(g++ --version | head -n 1)"
echo "CPU_RELEASE_BUILD_BGEN=${BGEN_PATH}"
echo "CPU_RELEASE_BUILD_BGEN_PROVENANCE=${bgen_provenance}"
echo "CPU_RELEASE_BUILD_BGEN_SOURCE_SHA256=${bgen_source_sha256}"
echo "CPU_RELEASE_BUILD_BGEN_COMPATIBILITY_PATCH=${bgen_compatibility_patch}"
echo "CPU_RELEASE_BUILD_MKL=${MKLROOT}"

export STATIC=1
if (( run_tests == 1 )); then
  cmake_build_testing=ON
else
  cmake_build_testing=OFF
fi
cmake -S "${repo_root}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  "-DBUILD_TESTING=${cmake_build_testing}" \
  -DREGENIE_WITH_CUDA=OFF \
  "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -march=${cpu_architecture} -mtune=${cpu_tune}"
build_targets=(regenie)
if (( run_tests == 1 )); then
  build_targets+=(step1_compute_test cox_firth_test)
fi
cmake --build "${build_dir}" \
  --target "${build_targets[@]}" \
  --parallel "${jobs}"

binary_path="${build_dir}/regenie"
[[ -x "${binary_path}" ]] || die "built executable is missing: ${binary_path}"

if (( run_tests == 1 )); then
  (
    cd "${build_dir}"
    ctest --output-on-failure
  )
  "${build_dir}/step1_compute_test" --backend cpu

  # The upstream regression driver discovers a binary in its working tree and
  # writes outputs beside its fixtures. Give it an isolated copy so a release
  # build neither consumes nor modifies unrelated files in the source checkout.
  regression_dir="${build_dir}/regression"
  cmake -E remove_directory "${regression_dir}"
  mkdir -p "${regression_dir}/src"
  cp -R "${repo_root}/example" "${regression_dir}/example"
  cp -R "${repo_root}/test" "${regression_dir}/test"
  ln -s "${binary_path}" "${regression_dir}/regenie"
  "${regression_dir}/test/test_bash.sh" --path "${regression_dir}"
  echo "CPU_RELEASE_BUILD_REGRESSION=PASS"
else
  echo "CPU_RELEASE_BUILD_TESTS=SKIPPED"
fi

dynamic_entries="$(readelf -d "${binary_path}")"
if grep -Eq 'NEEDED.*lib(mkl|stdc\+\+|gcc_s)' <<<"${dynamic_entries}"; then
  die "the release binary unexpectedly has dynamic oneMKL or GNU C++ runtime dependencies"
fi
if grep -Eq 'NEEDED.*lib(cudart|cublas|cusolver)' <<<"${dynamic_entries}"; then
  die "the CPU-only release binary unexpectedly has CUDA runtime dependencies"
fi
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
  printf 'BUILD_KIND=cpu-mkl\n'
  printf 'CPU_ARCHITECTURE=%s\n' "${cpu_architecture}"
  printf 'CPU_TUNE=%s\n' "${cpu_tune}"
  printf 'CXX=%s\n' "$(g++ --version | head -n 1)"
  printf 'CMAKE=%s\n' "$(cmake --version | head -n 1)"
  printf 'MKLROOT=%s\n' "${MKLROOT}"
  printf 'BGEN_PATH=%s\n' "${BGEN_PATH}"
  printf 'BGEN_PROVENANCE=%s\n' "${bgen_provenance}"
  printf 'BGEN_SOURCE_SHA256=%s\n' "${bgen_source_sha256}"
  printf 'BGEN_COMPATIBILITY_PATCH=%s\n' "${bgen_compatibility_patch}"
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

echo "CPU_RELEASE_BUILD_STATUS=PASS"
echo "CPU_RELEASE_BUILD_BINARY=${binary_path}"
echo "CPU_RELEASE_BUILD_ARCHIVE=${archive_path}"
echo "CPU_RELEASE_BUILD_CHECKSUM=${archive_path}.sha256"
echo "CPU_RELEASE_BUILD_END"
