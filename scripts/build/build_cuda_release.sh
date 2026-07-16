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
clean_dependencies=0
allow_external_bgen=0

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
  --build-dir PATH               Release work directory.
  --output-dir PATH              Package output directory (default: ./dist).
  --jobs N                       Parallel build jobs (default: nproc).
  --cuda-validation MODE         auto, always, or never (default: auto).
  --skip-tests                   Skip CTest, backend tests, CUDA validation,
                                 and both regression-suite passes.
  --clean                        Remove the selected release work directory first.
  --clean-dependencies           Rebuild the managed BGEN dependency.
  --allow-external-bgen          Permit an explicitly supplied BGEN_PATH and
                                 record it as unverified in package metadata.
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
    --clean-dependencies)
      clean_dependencies=1
      shift
      ;;
    --allow-external-bgen)
      allow_external_bgen=1
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
if [[ -n "${BGEN_PATH:-}" ]] && (( allow_external_bgen == 0 )); then
  die "BGEN_PATH is set; unset it for a verified managed dependency or pass --allow-external-bgen"
fi

for command_name in awk cmake cmp ctest g++ gfortran git gzip make nproc nvcc \
  python3 readelf sed sort tar; do
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
  if (( clean_dependencies == 1 )); then
    cmake -E remove_directory "${BGEN_PATH}"
  fi
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
  # The managed-cache stamp is specific to this compatibility correction, so
  # a cache hit carries the same provenance as a fresh managed build.
  bgen_compatibility_patch=1
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

source_dir="${build_dir}/source"
cmake_build_dir="${build_dir}/cmake"
cmake -E remove_directory "${source_dir}"
mkdir -p "${source_dir}"
git -C "${repo_root}" archive --format=tar HEAD |
  tar -xf - -C "${source_dir}"

version="$(tr -d '[:space:]' < "${source_dir}/VERSION")"
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
echo "CUDA_RELEASE_BUILD_SOURCE_SNAPSHOT=${source_dir}"
echo "CUDA_RELEASE_BUILD_COMMIT=${git_commit}"
echo "CUDA_RELEASE_BUILD_PROFILE=${profile}"
echo "CUDA_RELEASE_BUILD_ARCHITECTURES=${cuda_architectures}"
echo "CUDA_RELEASE_BUILD_CPU architecture=${cpu_architecture} tune=${cpu_tune}"
echo "CUDA_RELEASE_BUILD_TOOLKIT=$(nvcc --version | tail -n 1)"
echo "CUDA_RELEASE_BUILD_COMPILER=$(g++ --version | head -n 1)"
echo "CUDA_RELEASE_BUILD_BGEN=${BGEN_PATH}"
echo "CUDA_RELEASE_BUILD_BGEN_PROVENANCE=${bgen_provenance}"
echo "CUDA_RELEASE_BUILD_BGEN_SOURCE_SHA256=${bgen_source_sha256}"
echo "CUDA_RELEASE_BUILD_BGEN_COMPATIBILITY_PATCH=${bgen_compatibility_patch}"
echo "CUDA_RELEASE_BUILD_MKL=${MKLROOT}"

export STATIC=1
if (( run_tests == 1 )); then
  cmake_build_testing=ON
else
  cmake_build_testing=OFF
fi
cmake -S "${source_dir}" -B "${cmake_build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  "-DBUILD_TESTING=${cmake_build_testing}" \
  -DREGENIE_WITH_CUDA=ON \
  "-DREGENIE_CUDA_ARCHITECTURES=${cuda_architectures}" \
  "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -march=${cpu_architecture} -mtune=${cpu_tune}" \
  "-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG -Xcompiler=-march=${cpu_architecture},-mtune=${cpu_tune}"
build_targets=(regenie)
if (( run_tests == 1 )); then
  build_targets+=(step1_compute_test cox_firth_test)
fi
cmake --build "${cmake_build_dir}" \
  --target "${build_targets[@]}" \
  --parallel "${jobs}"

if (( run_tests == 1 )); then
  (
    cd "${cmake_build_dir}"
    ctest --output-on-failure
  )
  "${cmake_build_dir}/step1_compute_test" --backend cpu

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
    BUILD_DIR="${cmake_build_dir}" \
    JOBS="${jobs}" \
      "${source_dir}/scripts/test_step1_cuda.sh"
  elif [[ "${cuda_validation}" == "never" ]]; then
    echo "CUDA_RELEASE_BUILD_CUDA_VALIDATION=SKIPPED reason=disabled"
  else
    echo "CUDA_RELEASE_BUILD_CUDA_VALIDATION=SKIPPED reason=no_usable_gpu"
  fi

  # Exercise the complete upstream regression suite through the CUDA binary's
  # CPU fallback. GPU-specific behavior is covered separately above by the
  # focused Step 1 CUDA validation suite.
  regression_dir="${build_dir}/regression"
  cmake -E remove_directory "${regression_dir}"
  mkdir -p "${regression_dir}/src"
  cp -R "${source_dir}/example" "${regression_dir}/example"
  cp -R "${source_dir}/test" "${regression_dir}/test"
  ln -s "${cmake_build_dir}/regenie" "${regression_dir}/regenie"
  CUDA_VISIBLE_DEVICES="" \
    "${regression_dir}/test/test_bash.sh" --path "${regression_dir}"
  echo "CUDA_RELEASE_BUILD_REGRESSION=PASS"
else
  echo "CUDA_RELEASE_BUILD_TESTS=SKIPPED"
fi

binary_path="${cmake_build_dir}/regenie"
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
dynamic_needed="$(awk '
  /Shared library:/ {
    value = $NF
    gsub(/^\[|\]$/, "", value)
    libraries = libraries (libraries == "" ? "" : ",") value
  }
  END { print libraries }
' <<<"${dynamic_entries}")"
glibc_required="$(
  readelf --version-info "${binary_path}" 2>/dev/null |
    grep -o 'GLIBC_[0-9][0-9.]*' |
    sort -Vu |
    tail -n 1 || true
)"
[[ -n "${glibc_required}" ]] || glibc_required="unknown"
cuda_toolkit_version="$(nvcc --version | awk '
  /release/ {
    value = $0
    sub(/^.*release[[:space:]]+/, "", value)
    sub(/,.*/, "", value)
    print value
    exit
  }
')"
[[ -n "${cuda_toolkit_version}" ]] || cuda_toolkit_version="unknown"

cmake -E remove_directory "${stage_root}"
mkdir -p "${stage_dir}/bin"
cp "${binary_path}" "${stage_dir}/bin/regenie"
cp "${source_dir}/LICENSE" "${source_dir}/README.md" "${source_dir}/VERSION" \
  "${stage_dir}/"
{
  printf 'REGENIE_VERSION=%s\n' "${version}"
  printf 'GIT_COMMIT=%s\n' "${git_commit}"
  printf 'GIT_DESCRIBE=%s\n' "${git_describe}"
  printf 'BUILD_KIND=cuda\n'
  printf 'CUDA_PROFILE=%s\n' "${profile}"
  printf 'CUDA_ARCHITECTURES=%s\n' "${cuda_architectures}"
  printf 'CUDA_TOOLKIT_VERSION=%s\n' "${cuda_toolkit_version}"
  printf 'CPU_ARCHITECTURE=%s\n' "${cpu_architecture}"
  printf 'CPU_TUNE=%s\n' "${cpu_tune}"
  printf 'NVCC=%s\n' "$(nvcc --version | tail -n 1)"
  printf 'CXX=%s\n' "$(g++ --version | head -n 1)"
  printf 'CMAKE=%s\n' "$(cmake --version | head -n 1)"
  printf 'MKLROOT=%s\n' "${MKLROOT}"
  printf 'BGEN_PATH=%s\n' "${BGEN_PATH}"
  printf 'BGEN_PROVENANCE=%s\n' "${bgen_provenance}"
  printf 'BGEN_SOURCE_SHA256=%s\n' "${bgen_source_sha256}"
  printf 'BGEN_COMPATIBILITY_PATCH=%s\n' "${bgen_compatibility_patch}"
  printf 'GLIBC_REQUIRED=%s\n' "${glibc_required}"
  printf 'DYNAMIC_NEEDED=%s\n' "${dynamic_needed}"
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

archive_validation="${build_dir}/archive-validation"
cmake -E remove_directory "${archive_validation}"
mkdir -p "${archive_validation}"
tar -xzf "${archive_path}" -C "${archive_validation}"
validated_dir="${archive_validation}/${artifact_base}"
validated_binary="${validated_dir}/bin/regenie"
[[ -x "${validated_binary}" ]] ||
  die "packaged CUDA executable is missing after archive extraction"
cmp -s "${binary_path}" "${validated_binary}" ||
  die "packaged CUDA executable differs from the validated build output"
"${validated_binary}" --help >/dev/null
if command -v ldd >/dev/null 2>&1 &&
   ldd "${validated_binary}" | grep -q 'not found'; then
  ldd "${validated_binary}" >&2
  die "packaged CUDA executable has unresolved shared-library dependencies"
fi
archive_actual_sha256="$("${sha256_command[@]}" "${archive_path}" | awk '{print $1}')"
archive_recorded_sha256="$(awk 'NR == 1 { print $1 }' "${archive_path}.sha256")"
[[ "${archive_actual_sha256}" == "${archive_recorded_sha256}" ]] ||
  die "packaged CUDA archive does not match its SHA-256 file"

if (( run_tests == 1 )); then
  packaged_regression_dir="${build_dir}/packaged-regression"
  cmake -E remove_directory "${packaged_regression_dir}"
  mkdir -p "${packaged_regression_dir}/src"
  cp -R "${source_dir}/example" "${packaged_regression_dir}/example"
  cp -R "${source_dir}/test" "${packaged_regression_dir}/test"
  ln -s "${validated_binary}" "${packaged_regression_dir}/regenie"
  CUDA_VISIBLE_DEVICES="" \
    "${packaged_regression_dir}/test/test_bash.sh" \
      --path "${packaged_regression_dir}"
  echo "CUDA_RELEASE_BUILD_PACKAGED_REGRESSION=PASS"
fi

echo "CUDA_RELEASE_BUILD_STATUS=PASS"
echo "CUDA_RELEASE_BUILD_BINARY=${binary_path}"
echo "CUDA_RELEASE_BUILD_ARCHIVE=${archive_path}"
echo "CUDA_RELEASE_BUILD_CHECKSUM=${archive_path}.sha256"
echo "CUDA_RELEASE_BUILD_END"
