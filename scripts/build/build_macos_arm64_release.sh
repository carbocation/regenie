#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

deployment_target="${MACOSX_DEPLOYMENT_TARGET:-13.0}"
cpu_target="${CPU_TARGET:-apple-m1}"
openblas_threads="${OPENBLAS_NUM_THREADS:-128}"
build_dir="${BUILD_DIR:-}"
output_dir="${OUTPUT_DIR:-${repo_root}/dist}"
jobs="${JOBS:-}"
run_tests=1
clean_build=0
clean_dependencies=0

usage() {
  cat <<'USAGE'
Build and package a portable Apple Silicon REGENIE release.

Usage:
  scripts/build/build_macos_arm64_release.sh [options]

Options:
  --deployment-target VERSION  Minimum macOS version (default: 13.0).
  --cpu-target CPU             Apple Clang -mcpu target (default: apple-m1).
  --native-cpu                 Use -mcpu=native. The artifact may not run on
                               a different Apple Silicon generation.
  --openblas-threads N         OpenBLAS compile-time thread ceiling
                               (default: 128).
  --build-dir PATH             Release work directory.
  --output-dir PATH            Package output directory (default: ./dist).
  --jobs N                     Parallel build jobs (default: logical CPUs).
  --skip-tests                 Skip CTest and both regression-suite passes.
  --clean                      Remove the selected release work directory.
  --clean-dependencies         Rebuild all managed dependencies from their
                               checksum-verified source archives.
  -h, --help                   Show this help.

Environment equivalents:
  MACOSX_DEPLOYMENT_TARGET, CPU_TARGET, OPENBLAS_NUM_THREADS, BUILD_DIR,
  OUTPUT_DIR, JOBS, SOURCE_DATE_EPOCH, XDG_CACHE_HOME.

The builder requires an Apple Silicon Mac, Xcode command-line tools, CMake,
Python 3, and 7-Zip (`brew install cmake sevenzip`). It downloads pinned BGEN,
OpenBLAS, LLVM OpenMP, and Ventura-compatible Apple Silicon gfortran sources or
packages into ${XDG_CACHE_HOME:-$HOME/.cache}/regenie-build-deps/macos-arm64.
Every archive is checked against a SHA-256 value embedded in this script.
USAGE
}

die() {
  echo "MACOS_ARM64_RELEASE_BUILD_ERROR: $*" >&2
  exit 1
}

require_value() {
  if (( $# < 2 )) || [[ -z "$2" ]]; then
    die "$1 requires a value"
  fi
}

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

ensure_archive() {
  local url="$1"
  local archive="$2"
  local expected_sha256="$3"

  if [[ ! -f "${archive}" ]]; then
    curl --fail --location --retry 3 --output "${archive}" "${url}"
  fi
  local actual_sha256
  actual_sha256="$(sha256_file "${archive}")"
  [[ "${actual_sha256}" == "${expected_sha256}" ]] ||
    die "archive checksum mismatch for ${archive}: expected ${expected_sha256}, received ${actual_sha256}"
}

macos_version_at_least() {
  awk -v actual="$1" -v required="$2" '
    BEGIN {
      split(actual, a, ".")
      split(required, r, ".")
      for (i = 1; i <= 3; ++i) {
        av = (a[i] == "" ? 0 : a[i]) + 0
        rv = (r[i] == "" ? 0 : r[i]) + 0
        if (av > rv) exit 0
        if (av < rv) exit 1
      }
      exit 0
    }
  '
}

binary_minos() {
  vtool -show-build "$1" 2>/dev/null |
    awk '$1 == "minos" { print $2; exit }'
}

while (( $# > 0 )); do
  case "$1" in
    --deployment-target)
      require_value "$@"
      deployment_target="$2"
      shift 2
      ;;
    --cpu-target)
      require_value "$@"
      cpu_target="$2"
      shift 2
      ;;
    --native-cpu)
      cpu_target="native"
      shift
      ;;
    --openblas-threads)
      require_value "$@"
      openblas_threads="$2"
      shift 2
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

[[ "$(uname -s)" == "Darwin" ]] || die "release builds require macOS"
[[ "$(uname -m)" == "arm64" ]] || die "release builds require Apple Silicon"
[[ "${deployment_target}" =~ ^[0-9]+\.[0-9]+([.][0-9]+)?$ ]] ||
  die "--deployment-target must be a dotted macOS version"
macos_version_at_least "${deployment_target}" "13.0" ||
  die "the pinned Apple Silicon gfortran toolchain requires macOS 13.0 or newer"
[[ "${openblas_threads}" =~ ^[1-9][0-9]*$ ]] ||
  die "--openblas-threads must be a positive integer"
if [[ -z "${jobs}" ]]; then
  jobs="$(sysctl -n hw.logicalcpu)"
fi
[[ "${jobs}" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"

if ! git -C "${repo_root}" diff --quiet ||
   ! git -C "${repo_root}" diff --cached --quiet; then
  die "tracked source changes are present; commit or stash them before making a release artifact"
fi

for command_name in 7zz awk bash clang clang++ cmake codesign ctest curl \
  file git gzip install_name_tool make otool perl pkgutil python3 shasum \
  sysctl tar vtool xcrun; do
  command -v "${command_name}" >/dev/null 2>&1 ||
    die "required command '${command_name}' was not found"
done
xcrun --show-sdk-path >/dev/null

version="$(tr -d '[:space:]' < "${repo_root}/VERSION")"
git_commit="$(git -C "${repo_root}" rev-parse HEAD)"
git_short="$(git -C "${repo_root}" rev-parse --short=12 HEAD)"
git_describe="$(git -C "${repo_root}" describe --always --dirty)"
source_date_epoch="${SOURCE_DATE_EPOCH:-$(git -C "${repo_root}" log -1 --format=%ct)}"
[[ "${source_date_epoch}" =~ ^[0-9]+$ ]] ||
  die "SOURCE_DATE_EPOCH must be a non-negative integer"

target_label="${deployment_target//[^[:alnum:]._-]/-}"
cpu_label="${cpu_target//[^[:alnum:]._-]/-}"
if [[ -z "${build_dir}" ]]; then
  build_dir="${repo_root}/build/macos-arm64-release-${target_label}-${cpu_label}"
fi
if (( clean_build == 1 )) && [[ -d "${build_dir}" ]]; then
  cmake -E remove_directory "${build_dir}"
fi
mkdir -p "${build_dir}" "${output_dir}"

cache_root="${XDG_CACHE_HOME:-$HOME/.cache}/regenie-build-deps/macos-arm64"
mkdir -p "${cache_root}"
if (( clean_dependencies == 1 )); then
  cmake -E remove_directory "${cache_root}"
  mkdir -p "${cache_root}"
fi

bgen_version="1.1.7"
bgen_url="https://enkre.net/cgi-bin/code/bgen/tarball/release/v1.1.7"
bgen_sha256="6476b077af6c8e98e85fd7e09f58cb3fdf143ff91850c984248fd4dc2d74a8c3"
bgen_archive="${cache_root}/bgen-v${bgen_version}.tgz"
bgen_root="${cache_root}/bgen-v${bgen_version}-macos-${target_label}"
bgen_stamp="${bgen_root}/build/.regenie-${bgen_sha256}-macos-${target_label}-streampos-v1"
ensure_archive "${bgen_url}" "${bgen_archive}" "${bgen_sha256}"
bgen_provenance="verified-managed-cache"
if [[ ! -f "${bgen_root}/build/libbgen.a" || ! -f "${bgen_stamp}" ]]; then
  bgen_provenance="verified-fresh-build"
  bgen_extract="${cache_root}/bgen-extract-${target_label}"
  cmake -E remove_directory "${bgen_root}"
  cmake -E remove_directory "${bgen_extract}"
  mkdir -p "${bgen_extract}"
  tar -xzf "${bgen_archive}" -C "${bgen_extract}"
  [[ -d "${bgen_extract}/v${bgen_version}" ]] ||
    die "verified BGEN archive has an unexpected top-level directory"
  mv "${bgen_extract}/v${bgen_version}" "${bgen_root}"
  cmake -E remove_directory "${bgen_extract}"
  bgen_view="${bgen_root}/src/View.cpp"
  grep -q 'std::ios::streampos origin' "${bgen_view}" ||
    die "verified BGEN source does not contain the expected compatibility token"
  perl -pi -e 's/std::ios::streampos origin/std::streampos origin/' "${bgen_view}"
  (
    cd "${bgen_root}"
    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
    export MACOSX_DEPLOYMENT_TARGET="${deployment_target}"
    python3 waf configure
    python3 waf
  )
  cmake -E touch "${bgen_stamp}"
fi

gfortran_version="12.2"
gfortran_url="https://github.com/fxcoudert/gfortran-for-macOS/releases/download/12.2-ventura/gfortran-ARM-12.2-Ventura.dmg"
gfortran_sha256="159f9761abbf748accf5945713785f2d2a3f5036ceaa0ce9ecabb5d1b0698289"
gfortran_archive="${cache_root}/gfortran-ARM-${gfortran_version}-Ventura.dmg"
gfortran_root="${cache_root}/gfortran-${gfortran_version}-ventura-arm64"
gfortran_stamp="${gfortran_root}/.regenie-${gfortran_sha256}-extracted-v1"
ensure_archive "${gfortran_url}" "${gfortran_archive}" "${gfortran_sha256}"
gfortran_provenance="verified-managed-cache"
if [[ ! -x "${gfortran_root}/bin/gfortran" || ! -f "${gfortran_stamp}" ]]; then
  gfortran_provenance="verified-fresh-extraction"
  gfortran_extract="${cache_root}/gfortran-extract"
  gfortran_expanded="${cache_root}/gfortran-package"
  cmake -E remove_directory "${gfortran_root}"
  cmake -E remove_directory "${gfortran_extract}"
  cmake -E remove_directory "${gfortran_expanded}"
  mkdir -p "${gfortran_extract}"
  7zz x -y "-o${gfortran_extract}" "${gfortran_archive}" >/dev/null
  gfortran_package="$(find "${gfortran_extract}" -name gfortran.pkg -print -quit)"
  [[ -n "${gfortran_package}" ]] ||
    die "verified gfortran disk image contains no gfortran.pkg"
  pkgutil --expand-full "${gfortran_package}" "${gfortran_expanded}"
  gfortran_payload="${gfortran_expanded}/Payload/usr/local/gfortran"
  [[ -x "${gfortran_payload}/bin/gfortran" ]] ||
    die "verified gfortran package has an unexpected payload"
  mv "${gfortran_payload}" "${gfortran_root}"
  cmake -E remove_directory "${gfortran_extract}"
  cmake -E remove_directory "${gfortran_expanded}"
  cmake -E touch "${gfortran_stamp}"
fi
[[ "$(binary_minos "${gfortran_root}/bin/gfortran")" == "13.0" ]] ||
  die "pinned gfortran compiler does not declare the expected macOS 13.0 minimum"
for runtime_name in libgfortran.a libquadmath.a; do
  [[ -f "${gfortran_root}/lib/${runtime_name}" ]] ||
    die "pinned gfortran package is missing ${runtime_name}"
done
export PATH="${gfortran_root}/bin:${PATH}"

openblas_version="0.3.33"
openblas_url="https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v${openblas_version}.tar.gz"
openblas_sha256="6761af1d9f5d353ab4f0b7497be2643313b36c8f31caec0144bfef198e71e6ab"
openblas_archive="${cache_root}/OpenBLAS-${openblas_version}.tar.gz"
openblas_source="${cache_root}/OpenBLAS-${openblas_version}-source-${target_label}"
openblas_root="${cache_root}/OpenBLAS-${openblas_version}-macos-${target_label}"
openblas_target="VORTEX"
openblas_stamp="${openblas_root}/.regenie-${openblas_sha256}-macos-${target_label}-target-${openblas_target}-threads-${openblas_threads}-sequential-goals-v2"
ensure_archive "${openblas_url}" "${openblas_archive}" "${openblas_sha256}"
openblas_provenance="verified-managed-cache"
if [[ ! -f "${openblas_root}/lib/libopenblas.a" || ! -f "${openblas_stamp}" ]]; then
  openblas_provenance="verified-fresh-build"
  openblas_extract="${cache_root}/openblas-extract-${target_label}"
  cmake -E remove_directory "${openblas_source}"
  cmake -E remove_directory "${openblas_root}"
  cmake -E remove_directory "${openblas_extract}"
  mkdir -p "${openblas_extract}"
  tar -xzf "${openblas_archive}" -C "${openblas_extract}"
  [[ -d "${openblas_extract}/OpenBLAS-${openblas_version}" ]] ||
    die "verified OpenBLAS archive has an unexpected top-level directory"
  mv "${openblas_extract}/OpenBLAS-${openblas_version}" "${openblas_source}"
  cmake -E remove_directory "${openblas_extract}"
  openblas_args=(
    CC=/usr/bin/clang
    HOSTCC=/usr/bin/clang
    "FC=${gfortran_root}/bin/gfortran"
    DYNAMIC_ARCH=0
    "TARGET=${openblas_target}"
    NO_SVE=1
    NO_SHARED=1
    USE_OPENMP=0
    USE_THREAD=1
    "NUM_THREADS=${openblas_threads}"
  )
  (
    export MACOSX_DEPLOYMENT_TARGET="${deployment_target}"
    # OpenBLAS's `libs` and `netlib` top-level goals can concurrently update
    # the same static archive when named in one parallel make invocation.
    # Build the goals separately while retaining parallelism within each goal.
    make -C "${openblas_source}" -j "${jobs}" \
      "${openblas_args[@]}" libs
    make -C "${openblas_source}" -j "${jobs}" \
      "${openblas_args[@]}" netlib
    make -C "${openblas_source}" \
      "${openblas_args[@]}" "PREFIX=${openblas_root}" install
  )
  [[ -f "${openblas_root}/lib/libopenblas.a" ]] ||
    die "OpenBLAS static archive was not installed"
  cmake -E touch "${openblas_stamp}"
fi

# Force the static archive's BLAS entry point and its transitive Fortran
# runtime dependencies through the linker before accepting a new or cached
# dependency. This catches incomplete archives at dependency-validation time
# instead of much later while linking REGENIE.
openblas_link_test_dir="${cache_root}/openblas-link-test-${target_label}"
cmake -E remove_directory "${openblas_link_test_dir}"
mkdir -p "${openblas_link_test_dir}"
cat > "${openblas_link_test_dir}/main.cpp" <<'OPENBLAS_LINK_TEST'
#include <cblas.h>
#include <lapacke.h>

int main() {
  double lhs[1] = {2.0};
  double rhs[1] = {3.0};
  double output[1] = {0.0};
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              1, 1, 1, 1.0, lhs, 1, rhs, 1, 0.0, output, 1);
  double coefficient[1] = {2.0};
  double dependent[1] = {6.0};
  lapack_int pivot[1] = {0};
  int solve_status = LAPACKE_dgesv(
      LAPACK_COL_MAJOR, 1, 1, coefficient, 1, pivot, dependent, 1);
  return output[0] == 6.0 && solve_status == 0 && dependent[0] == 3.0
             ? 0
             : 1;
}
OPENBLAS_LINK_TEST
/usr/bin/clang++ \
  "-mmacosx-version-min=${deployment_target}" \
  -I"${openblas_root}/include" \
  "${openblas_link_test_dir}/main.cpp" \
  "${openblas_root}/lib/libopenblas.a" \
  "${gfortran_root}/lib/libgfortran.a" \
  "${gfortran_root}/lib/libquadmath.a" \
  -o "${openblas_link_test_dir}/openblas-link-test"
"${openblas_link_test_dir}/openblas-link-test" ||
  die "managed OpenBLAS static archive failed its link and runtime smoke test"
cmake -E remove_directory "${openblas_link_test_dir}"

llvm_version="22.1.6"
llvm_url="https://github.com/llvm/llvm-project/releases/download/llvmorg-${llvm_version}/llvm-project-${llvm_version}.src.tar.xz"
llvm_sha256="6e0b376a1f6d9873e7dfb09ae6e04b9c7024400f01733fa4c29be69d5c138bc2"
llvm_archive="${cache_root}/llvm-project-${llvm_version}.src.tar.xz"
llvm_source="${cache_root}/llvm-project-${llvm_version}.src"
libomp_root="${cache_root}/libomp-${llvm_version}-macos-${target_label}"
libomp_stamp="${libomp_root}/.regenie-${llvm_sha256}-macos-${target_label}"
ensure_archive "${llvm_url}" "${llvm_archive}" "${llvm_sha256}"
libomp_provenance="verified-managed-cache"
if [[ ! -f "${libomp_root}/lib/libomp.dylib" || ! -f "${libomp_stamp}" ]]; then
  libomp_provenance="verified-fresh-build"
  libomp_build="${cache_root}/libomp-build-${llvm_version}-${target_label}"
  cmake -E remove_directory "${llvm_source}"
  cmake -E remove_directory "${libomp_root}"
  cmake -E remove_directory "${libomp_build}"
  tar -xJf "${llvm_archive}" -C "${cache_root}"
  [[ -d "${llvm_source}/runtimes" ]] ||
    die "verified LLVM archive has an unexpected top-level directory"
  cmake -S "${llvm_source}/runtimes" -B "${libomp_build}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${libomp_root}" \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="${deployment_target}" \
    -DLIBOMP_INSTALL_ALIASES=OFF \
    -DLLVM_ENABLE_RUNTIMES=openmp \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DOPENMP_ENABLE_OMPT_TOOLS=OFF
  cmake --build "${libomp_build}" --parallel "${jobs}"
  cmake --install "${libomp_build}"
  [[ -f "${libomp_root}/lib/libomp.dylib" ]] ||
    die "LLVM OpenMP runtime was not installed"
  cmake -E touch "${libomp_stamp}"
fi
[[ "$(binary_minos "${libomp_root}/lib/libomp.dylib")" == "${deployment_target}" ]] ||
  die "built libomp does not declare macOS ${deployment_target} as its minimum"

source_dir="${build_dir}/source"
cmake_build_dir="${build_dir}/cmake"
cmake -E remove_directory "${source_dir}"
cmake -E remove_directory "${cmake_build_dir}"
mkdir -p "${source_dir}" "${cmake_build_dir}"
git -C "${repo_root}" archive --format=tar HEAD | tar -xf - -C "${source_dir}"

echo "MACOS_ARM64_RELEASE_BUILD_BEGIN"
echo "MACOS_ARM64_RELEASE_BUILD_SOURCE=${repo_root}"
echo "MACOS_ARM64_RELEASE_BUILD_COMMIT=${git_commit}"
echo "MACOS_ARM64_RELEASE_BUILD_DEPLOYMENT_TARGET=${deployment_target}"
echo "MACOS_ARM64_RELEASE_BUILD_CPU_TARGET=${cpu_target}"
echo "MACOS_ARM64_RELEASE_BUILD_BGEN_PROVENANCE=${bgen_provenance}"
echo "MACOS_ARM64_RELEASE_BUILD_BGEN_SOURCE_SHA256=${bgen_sha256}"
echo "MACOS_ARM64_RELEASE_BUILD_GFORTRAN_PROVENANCE=${gfortran_provenance}"
echo "MACOS_ARM64_RELEASE_BUILD_GFORTRAN_SOURCE_SHA256=${gfortran_sha256}"
echo "MACOS_ARM64_RELEASE_BUILD_OPENBLAS_PROVENANCE=${openblas_provenance}"
echo "MACOS_ARM64_RELEASE_BUILD_OPENBLAS_SOURCE_SHA256=${openblas_sha256}"
echo "MACOS_ARM64_RELEASE_BUILD_OPENBLAS_TARGET=${openblas_target}"
echo "MACOS_ARM64_RELEASE_BUILD_LIBOMP_PROVENANCE=${libomp_provenance}"
echo "MACOS_ARM64_RELEASE_BUILD_LIBOMP_SOURCE_SHA256=${llvm_sha256}"

export BGEN_PATH="${bgen_root}"
export OPENBLAS_ROOT="${openblas_root}"
export MACOSX_DEPLOYMENT_TARGET="${deployment_target}"
export STATIC=1
if (( run_tests == 1 )); then
  cmake_build_testing=ON
else
  cmake_build_testing=OFF
fi
cmake -S "${source_dir}" -B "${cmake_build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  "-DBUILD_TESTING=${cmake_build_testing}" \
  -DREGENIE_WITH_CUDA=OFF \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="${deployment_target}" \
  -DCMAKE_PREFIX_PATH="${libomp_root}" \
  -DOpenMP_ROOT="${libomp_root}" \
  "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -mcpu=${cpu_target}"

build_targets=(regenie)
if (( run_tests == 1 )); then
  build_targets+=(step1_compute_test cox_firth_test)
fi
cmake --build "${cmake_build_dir}" \
  --target "${build_targets[@]}" \
  --parallel "${jobs}"

binary_path="${cmake_build_dir}/regenie"
[[ -x "${binary_path}" ]] || die "built executable is missing: ${binary_path}"
[[ "$(binary_minos "${binary_path}")" == "${deployment_target}" ]] ||
  die "REGENIE does not declare macOS ${deployment_target} as its minimum"

if (( run_tests == 1 )); then
  ctest --test-dir "${cmake_build_dir}" --output-on-failure
  regression_dir="${build_dir}/regression-build"
  cmake -E remove_directory "${regression_dir}"
  mkdir -p "${regression_dir}/src"
  cp -R "${source_dir}/example" "${regression_dir}/example"
  cp -R "${source_dir}/test" "${regression_dir}/test"
  ln -s "${binary_path}" "${regression_dir}/regenie"
  bash "${regression_dir}/test/test_bash.sh" --path "${regression_dir}"
  echo "MACOS_ARM64_RELEASE_BUILD_REGRESSION=PASS"
else
  echo "MACOS_ARM64_RELEASE_BUILD_TESTS=SKIPPED"
fi

artifact_base="regenie-${version}-g${git_short}-macos-arm64-macos${target_label}-${cpu_label}-openblas"
stage_root="${build_dir}/package"
stage_dir="${stage_root}/${artifact_base}"
archive_path="${output_dir}/${artifact_base}.tar.gz"
cmake -E remove_directory "${stage_root}"
mkdir -p "${stage_dir}/bin" "${stage_dir}/lib" \
  "${stage_dir}/THIRD-PARTY-LICENSES"
cp "${binary_path}" "${stage_dir}/bin/regenie"
cp "${libomp_root}/lib/libomp.dylib" "${stage_dir}/lib/libomp.dylib"
cp "${source_dir}/LICENSE" "${source_dir}/README.md" "${source_dir}/VERSION" \
  "${stage_dir}/"
cp "${openblas_source}/LICENSE" \
  "${stage_dir}/THIRD-PARTY-LICENSES/OpenBLAS.txt"
cp "${llvm_source}/openmp/LICENSE.TXT" \
  "${stage_dir}/THIRD-PARTY-LICENSES/LLVM-OpenMP.txt"
cp "${bgen_root}/LICENSE_1_0.txt" \
  "${stage_dir}/THIRD-PARTY-LICENSES/BGEN-Boost-1.0.txt"

old_libomp="$(otool -L "${stage_dir}/bin/regenie" |
  awk '/libomp[^[:space:]]*[.]dylib/ { print $1; exit }')"
[[ -n "${old_libomp}" ]] || die "built REGENIE binary has no libomp dependency"
install_name_tool -change "${old_libomp}" \
  '@loader_path/../lib/libomp.dylib' "${stage_dir}/bin/regenie"
install_name_tool -id '@rpath/libomp.dylib' "${stage_dir}/lib/libomp.dylib"
codesign --force --sign - "${stage_dir}/lib/libomp.dylib"
codesign --force --sign - "${stage_dir}/bin/regenie"

{
  printf 'REGENIE_VERSION=%s\n' "${version}"
  printf 'GIT_COMMIT=%s\n' "${git_commit}"
  printf 'GIT_DESCRIBE=%s\n' "${git_describe}"
  printf 'BUILD_KIND=macos-arm64-openblas\n'
  printf 'MACOSX_DEPLOYMENT_TARGET=%s\n' "${deployment_target}"
  printf 'CPU_TARGET=%s\n' "${cpu_target}"
  printf 'CXX=%s\n' "$(/usr/bin/clang++ --version | head -n 1)"
  printf 'GFORTRAN=%s\n' "$(gfortran --version | head -n 1)"
  printf 'CMAKE=%s\n' "$(cmake --version | head -n 1)"
  printf 'BGEN_VERSION=%s\n' "${bgen_version}"
  printf 'BGEN_SOURCE_URL=%s\n' "${bgen_url}"
  printf 'BGEN_SOURCE_SHA256=%s\n' "${bgen_sha256}"
  printf 'BGEN_PROVENANCE=%s\n' "${bgen_provenance}"
  printf 'GFORTRAN_VERSION=%s\n' "${gfortran_version}"
  printf 'GFORTRAN_SOURCE_URL=%s\n' "${gfortran_url}"
  printf 'GFORTRAN_SOURCE_SHA256=%s\n' "${gfortran_sha256}"
  printf 'GFORTRAN_PROVENANCE=%s\n' "${gfortran_provenance}"
  printf 'OPENBLAS_VERSION=%s\n' "${openblas_version}"
  printf 'OPENBLAS_SOURCE_URL=%s\n' "${openblas_url}"
  printf 'OPENBLAS_SOURCE_SHA256=%s\n' "${openblas_sha256}"
  printf 'OPENBLAS_PROVENANCE=%s\n' "${openblas_provenance}"
  printf 'OPENBLAS_TARGET=%s\n' "${openblas_target}"
  printf 'OPENBLAS_NUM_THREADS=%s\n' "${openblas_threads}"
  printf 'LIBOMP_VERSION=%s\n' "${llvm_version}"
  printf 'LIBOMP_SOURCE_URL=%s\n' "${llvm_url}"
  printf 'LIBOMP_SOURCE_SHA256=%s\n' "${llvm_sha256}"
  printf 'LIBOMP_PROVENANCE=%s\n' "${libomp_provenance}"
  printf 'SOURCE_DATE_EPOCH=%s\n' "${source_date_epoch}"
  printf 'CODE_SIGNATURE=ad-hoc\n'
} > "${stage_dir}/BUILD-METADATA.txt"

runtime_links="$(otool -L "${stage_dir}/bin/regenie")"
grep -q '@loader_path/../lib/libomp.dylib' <<<"${runtime_links}" ||
  die "packaged REGENIE does not use its bundled libomp"
if grep -Eq '/opt/homebrew|/usr/local|libgfortran|libquadmath|libopenblas|liblapack' \
     <<<"${runtime_links}"; then
  echo "${runtime_links}" >&2
  die "packaged REGENIE has a non-portable runtime dependency"
fi
libomp_links="$(otool -L "${stage_dir}/lib/libomp.dylib")"
if grep -Eq '/opt/homebrew|/usr/local' <<<"${libomp_links}"; then
  echo "${libomp_links}" >&2
  die "packaged libomp has a non-portable runtime dependency"
fi
[[ "$(binary_minos "${stage_dir}/bin/regenie")" == "${deployment_target}" ]] ||
  die "packaged REGENIE has the wrong deployment target"
[[ "$(binary_minos "${stage_dir}/lib/libomp.dylib")" == "${deployment_target}" ]] ||
  die "packaged libomp has the wrong deployment target"
codesign --verify --strict "${stage_dir}/lib/libomp.dylib"
codesign --verify --strict "${stage_dir}/bin/regenie"
"${stage_dir}/bin/regenie" --version >/dev/null

if (( run_tests == 1 )); then
  package_regression_dir="${build_dir}/regression-package"
  cmake -E remove_directory "${package_regression_dir}"
  mkdir -p "${package_regression_dir}/src"
  cp -R "${source_dir}/example" "${package_regression_dir}/example"
  cp -R "${source_dir}/test" "${package_regression_dir}/test"
  ln -s "${stage_dir}/bin/regenie" "${package_regression_dir}/regenie"
  bash "${package_regression_dir}/test/test_bash.sh" \
    --path "${package_regression_dir}"
  echo "MACOS_ARM64_RELEASE_BUILD_PACKAGED_REGRESSION=PASS"
fi

archive_tmp="${archive_path}.tmp"
python3 - "${stage_root}" "${artifact_base}" "${archive_tmp}" \
  "${source_date_epoch}" <<'PY'
import gzip
import pathlib
import sys
import tarfile

stage_root = pathlib.Path(sys.argv[1])
artifact_base = sys.argv[2]
archive_path = pathlib.Path(sys.argv[3])
mtime = int(sys.argv[4])
artifact_root = stage_root / artifact_base
paths = [artifact_root] + sorted(artifact_root.rglob("*"))

with archive_path.open("wb") as raw:
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tf:
            for path in paths:
                arcname = path.relative_to(stage_root).as_posix()
                info = tf.gettarinfo(str(path), arcname)
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = mtime
                info.pax_headers = {}
                if info.isfile():
                    with path.open("rb") as source:
                        tf.addfile(info, source)
                else:
                    tf.addfile(info)
PY
mv "${archive_tmp}" "${archive_path}"
archive_sha256="$(sha256_file "${archive_path}")"
printf '%s  %s\n' "${archive_sha256}" "$(basename "${archive_path}")" \
  > "${archive_path}.sha256"

archive_validation="${build_dir}/archive-validation"
cmake -E remove_directory "${archive_validation}"
mkdir -p "${archive_validation}"
tar -xzf "${archive_path}" -C "${archive_validation}"
validated_dir="${archive_validation}/${artifact_base}"
codesign --verify --strict "${validated_dir}/lib/libomp.dylib"
codesign --verify --strict "${validated_dir}/bin/regenie"
"${validated_dir}/bin/regenie" --version >/dev/null
[[ "$(sha256_file "${archive_path}")" == "${archive_sha256}" ]] ||
  die "archive checksum changed during validation"

echo "MACOS_ARM64_RELEASE_BUILD_STATUS=PASS"
echo "MACOS_ARM64_RELEASE_BUILD_BINARY=${binary_path}"
echo "MACOS_ARM64_RELEASE_BUILD_ARCHIVE=${archive_path}"
echo "MACOS_ARM64_RELEASE_BUILD_CHECKSUM=${archive_path}.sha256"
echo "MACOS_ARM64_RELEASE_BUILD_END"
