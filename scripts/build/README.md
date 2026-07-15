# Release build scripts

The sibling release builders produce performance-oriented Linux x86-64
artifacts with static oneMKL and GNU C++ runtime linkage:

- `build_cpu_release.sh` creates a CPU-only binary that has no CUDA runtime
  dependency.
- `build_cuda_release.sh` creates a CUDA-enabled binary containing both the
  CUDA Step 1 backend and the CPU fallback backend.

Both builders validate the executable and create a versioned tarball plus
SHA-256 checksum under `dist/`.

Artifact names contain the source commit, and the builder refuses tracked source
changes that have not been committed. Untracked compiler products are allowed so
an existing remote build checkout can be reused.

When `BGEN_PATH` is unset, the builder manages BGEN v1.1.7 in its dependency
cache. It verifies the official source archive's pinned SHA-256 checksum and
requires a matching build stamp before reusing that cache. For a clean build it
applies the narrow `std::ios::streampos` to `std::streampos` compatibility
correction needed by current libstdc++ releases before invoking Waf. Supplying
`BGEN_PATH` explicitly is treated as an unverified external dependency and is
recorded as such in the package metadata.

## CPU-only release

The CPU builder defaults to `-march=x86-64-v3 -mtune=generic`, statically links
oneMKL, and verifies that the resulting executable has no CUDA runtime
dependencies. Its validation includes CTest, the Step 1 CPU backend test, and
the repository regression suite running against an isolated copy of the
committed fixtures:

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
environment overrides, and validation controls.
