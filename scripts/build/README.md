# Release build scripts

The sibling release builders produce performance-oriented Linux x86-64 and
Apple Silicon artifacts:

- `build_cpu_release.sh` creates a CPU-only binary that has no CUDA runtime
  dependency.
- `build_cuda_release.sh` creates a CUDA-enabled binary containing both the
  CUDA Step 1 backend and the CPU fallback backend.
- `build_macos_arm64_release.sh` creates a CPU-only Apple Silicon binary with
  static OpenBLAS and GNU Fortran runtimes plus a bundled LLVM OpenMP runtime.
- `build_linux_release_bundle.sh` drives both Linux builders from one commit and
  produces an upload-ready release directory.
- `verify_release_assets.py` validates archive checksums, build metadata,
  dependency provenance, and cross-artifact commit consistency.

Each platform builder validates the executable and creates a versioned tarball
plus SHA-256 checksum under `dist/`.

Artifact names contain the source commit, and the builder refuses tracked source
changes that have not been committed. Untracked compiler products are allowed so
an existing remote build checkout can be reused.

The Linux builders manage BGEN v1.1.7 when `BGEN_PATH` is unset. They verify
the official source archive's pinned SHA-256 checksum and require a matching
build stamp before reusing that cache. For a clean build they apply the narrow
`std::ios::streampos` to `std::streampos` compatibility correction needed by
current compilers before invoking Waf. An explicit `BGEN_PATH` is rejected by
default; passing `--allow-external-bgen` permits it as an unverified external
dependency and records that fact in package metadata. The Apple Silicon builder
always uses its own platform- and deployment-target-specific managed dependency
cache.

## Upload-ready Linux release set

On a Linux x86-64 machine with a supported CUDA toolkit and static oneMKL
development files, one command builds both public Linux variants from the same
committed source snapshot:

```bash
scripts/build/build_linux_release_bundle.sh \
  --profile datacenter \
  --cuda-validation always \
  --clean
```

The command ignores an inherited `BGEN_PATH` and uses the checksum-verified
managed dependency. It runs the CPU tests and regression suite, focused CUDA
validation, CPU-fallback regression through the CUDA binary, and repeats the
regression against both extracted packages. The output directory contains only
the CPU and CUDA archives, their individual checksum files, `SHA256SUMS`, and
`release-manifest.json`; these are ready to upload as GitHub release assets.
Build logs and intermediate files remain under the separate build directory.

To rebuild managed BGEN rather than reuse a verified cache, add
`--clean-dependencies`. Run
`scripts/build/build_linux_release_bundle.sh --help` for CPU-floor, CUDA-target,
work-directory, and validation controls.

## CPU-only release

The CPU builder defaults to `-march=x86-64-v3 -mtune=generic`, statically links
oneMKL, and verifies that the resulting executable has no CUDA runtime
dependencies. Its validation includes CTest, the Step 1 CPU backend test, and
the repository regression suite before and after packaging. Compilation uses a
`git archive HEAD` source snapshot so untracked checkout contents cannot enter
the artifact:

```bash
# Portable optimized binary for modern x86-64 hosts.
scripts/build/build_cpu_release.sh --clean

# Machine-specific optimization; do not redistribute to dissimilar CPUs.
scripts/build/build_cpu_release.sh --native-cpu --clean

# Select a different portable CPU floor explicitly.
scripts/build/build_cpu_release.sh \
  --cpu-architecture x86-64-v2 \
  --cpu-tune generic \
  --clean
```

Run `scripts/build/build_cpu_release.sh --help` for dependency expectations,
environment overrides, and validation controls.

## Apple Silicon release

The macOS builder defaults to an Apple M1 CPU floor and macOS 13.0 deployment
target, so one arm64 artifact can run on M1 and newer Apple Silicon Macs. It
builds BGEN, a deterministic Apple M1-floor OpenBLAS target, and LLVM OpenMP
from pinned, checksum-verified inputs and extracts a pinned Ventura-compatible
Apple Silicon gfortran toolchain. OpenBLAS and the GNU Fortran runtimes are
linked statically. `libomp.dylib` is bundled under `lib/`, and the packaged
executable uses a relative loader path rather than requiring Homebrew on the
destination machine.

The build itself is made from `git archive HEAD` in an isolated directory, so
untracked objects in the repository cannot enter the artifact. Validation
includes CTest, the full regression suite before and after relocation, Mach-O
deployment-target and dependency checks, ad-hoc code-signature verification,
archive extraction, and a final runtime check:

```bash
# Install build-time tools; Xcode command-line tools must also be available.
brew install cmake sevenzip

# Portable optimized binary for Apple Silicon Macs running macOS 13 or newer.
scripts/build/build_macos_arm64_release.sh --clean

# Rebuild every managed dependency from its verified archive.
scripts/build/build_macos_arm64_release.sh \
  --clean \
  --clean-dependencies

# Machine-specific host optimization; do not redistribute indiscriminately.
scripts/build/build_macos_arm64_release.sh --native-cpu --clean
```

The binary and bundled `libomp.dylib` are ad-hoc signed, not Developer-ID
signed or Apple-notarized. A browser that adds a quarantine attribute may cause
Gatekeeper to request explicit approval for the downloaded command-line tool.
Run `scripts/build/build_macos_arm64_release.sh --help` for deployment-target,
CPU-target, dependency-cache, and validation controls.

## CUDA release

The default `datacenter` profile produces one CUDA fat binary for these compute
capabilities:

| Compute capability | Representative GPUs |
| --- | --- |
| 7.5 | T4 |
| 8.0 | A100, A30 |
| 8.6 | A10, A40 |
| 8.9 | L4, L40 |
| 9.0 | H100, H200 |

The CUDA toolkit used on the build machine must support every requested target.
CUDA 13 no longer supports offline compilation for Volta, so V100 (`7.0`) is
kept in the separate `datacenter-v100` profile, which must be built with CUDA
12.x. The script checks `nvcc --list-gpu-arch` before starting the expensive
build.

The packaged host code defaults to `-march=x86-64-v3 -mtune=generic` so that a
single artifact can run across modern x86-64 GPU hosts. Use `--native-cpu` for a
machine-specific build, or select another CPU floor explicitly. GPU targets can
also be supplied directly, which is useful for newer architectures supported by
the installed toolkit. The builder passes the same CPU target to the ordinary
C++ compiler and NVCC's host compiler so Eigen retains one ABI across C++ and
CUDA translation units:

```bash
# Portable fat binary for common NVIDIA data-center GPU families.
scripts/build/build_cuda_release.sh --profile datacenter --clean

# Include V100 when building with a CUDA 12.x toolkit.
scripts/build/build_cuda_release.sh --profile datacenter-v100 --clean

# Target only the locally tested T4 and A100 architectures.
scripts/build/build_cuda_release.sh --profile tested --clean

# Explicit target set and machine-specific CPU optimization.
scripts/build/build_cuda_release.sh \
  --cuda-architectures '75;80;89;90' \
  --native-cpu \
  --clean
```

Run `scripts/build/build_cuda_release.sh --help` for dependency expectations,
environment overrides, and validation controls. CUDA release metadata records
the build toolkit version and dynamic runtime libraries; the destination host
must provide compatible CUDA libraries in addition to an NVIDIA driver.

## Verifying downloaded or cross-platform assets

The verifier can check a directory containing Linux and/or macOS archives. Each
archive must have its sibling `.sha256` file and `BUILD-METADATA.txt`:

```bash
python3 scripts/build/verify_release_assets.py ./release-assets \
  --expected-commit COMMIT_SHA \
  --require-kind cpu-mkl \
  --require-kind cuda \
  --require-kind macos-arm64-openblas \
  --require-verified-dependencies \
  --manifest ./release-assets/release-manifest.json \
  --sha256sums ./release-assets/SHA256SUMS
```

It rejects checksum failures, malformed metadata, mixed source commits or
versions, missing build kinds, and unverified dependency provenance.
