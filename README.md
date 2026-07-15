<!-- BEGIN FORK README PREAMBLE -->
# REGENIE acceleration fork

This repository is an ***unofficial***, performance-oriented fork of the [official
REGENIE project](https://github.com/rgcgithub/regenie). This is ***not*** an
official Regeneron Genetics Center distribution. The goal is to preserve the
upstream command-line interface and statistical behavior while improving
performance and adding GPU acceleration for Step 1.

## What this fork adds

- An optional NVIDIA CUDA backend for the compute-intensive parts of Step 1.
  (Note that Step 2 does not use CUDA.)
- Faster Step 1 host I/O, PGEN ingestion, profiling, and LOCO prediction output,
  even for CPU-only runs.
- Step 2 CPU fast paths for PGEN hardcalls, BGEN dosages, and dense
  quantitative-trait scoring, even for CPU-only runs.
- Faster saddlepoint approximation and Cox Firth correction paths, even for
  CPU-only runs.
- Structured performance profiles and CPU/CUDA conformance tests used to
  validate optimized paths against reference results.

Normal CPU-only builds will build and run without CUDA. To enable CUDA, build
with `-DREGENIE_WITH_CUDA=ON`. CUDA-enabled builds use `--compute-backend auto`
by default, selecting an available GPU for Step 1 and falling back to the CPU
when necessary; see the [CUDA build and runtime
documentation](docs/docs/install.md#experimental-cuda-step-1-backend) for
details. The CUDA path has been exercised on NVIDIA T4 and A100 GPUs.
Floating-point implementations can differ in their final printed digits, so
validation includes numerical comparisons in addition to regression tests.

Please report fork-specific problems to the [carbocation/regenie issue
tracker](https://github.com/carbocation/regenie/issues).

## Example optimized builds for x86-64 Ubuntu

> [!IMPORTANT] These example commands are specifically for **x86-64 Ubuntu Linux
> systems** using the APT package manager. Other systems require
> platform-appropriate dependencies and package-manager commands.

Both examples install `regenie` in `$HOME/.local/bin` and use Intel oneMKL. The
resulting binary is optimized for the machine that builds it.

### Performance-optimized CPU build

This example installs the build dependencies and oneMKL, fetches this fork and
the required BGEN library, builds a CPU-only binary, and installs it for the
current user:

```bash
set -eo pipefail
export PATH="$HOME/.local/bin:$PATH"

sudo apt-get update
sudo apt-get install -y \
  ca-certificates cmake g++ gfortran git gnupg make python3 wget zlib1g-dev

# Add Intel's official oneAPI APT repository and install oneMKL.
wget -O- \
  https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor \
  | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg >/dev/null
echo \
  "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list >/dev/null
sudo apt-get update
sudo apt-get install -y intel-oneapi-mkl-devel

source /opt/intel/oneapi/setvars.sh
export MKL_THREADING_LAYER=GNU

git clone --branch master --single-branch \
  https://github.com/carbocation/regenie.git regenie-acceleration
cd regenie-acceleration

# Build REGENIE's BGEN dependency once in a reusable cache directory.
deps_dir="${XDG_CACHE_HOME:-$HOME/.cache}/regenie-build-deps"
bgen_dir="$deps_dir/v1.1.7"
mkdir -p "$deps_dir"
if [ ! -f "$bgen_dir/build/libbgen.a" ]; then
  wget -O "$deps_dir/v1.1.7.tgz" \
    http://code.enkre.net/bgen/tarball/release/v1.1.7
  tar -xzf "$deps_dir/v1.1.7.tgz" -C "$deps_dir"
  (cd "$bgen_dir" && python3 waf configure && python3 waf)
fi

BGEN_PATH="$bgen_dir" \
MKLROOT="$MKLROOT" \
STATIC=1 \
cmake -S . -B build-cpu-mkl \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -mtune=native"

cmake --build build-cpu-mkl --target regenie -j "$(nproc)"
cmake --install build-cpu-mkl --prefix "$HOME/.local"

regenie --version
```

The CPU recipe uses static oneMKL linkage so the installed executable does not
require the oneAPI environment to be sourced at runtime.

### Performance-optimized CUDA build

This example assumes that the NVIDIA driver and CUDA toolkit are already
installed. It checks both, reports each GPU's compute capability, and builds
only for architectures that are present. See NVIDIA's [CUDA installation
guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) if either
check fails.

The resulting executable includes both CUDA and CPU backends. This recipe uses
REGENIE's existing `STATIC=1` build mode, which statically links oneMKL and the
supported host dependencies. CUDA remains dynamically linked. The executable
can use its CPU backend on a machine where the required CUDA shared libraries
are installed but no usable GPU is available. It will not start on a genuinely
CUDA-free machine where those libraries are absent; use the CPU-only build
above in that situation. The CUDA-enabled binary retains the same oneMKL and
native-host optimizations for Step 2 and for Step 1 CPU fallbacks.

```bash
set -eo pipefail
export PATH="$HOME/.local/bin:$PATH"

command -v nvidia-smi >/dev/null || {
  echo "The NVIDIA driver and nvidia-smi are required." >&2
  exit 1
}
command -v nvcc >/dev/null || {
  echo "The CUDA toolkit and nvcc are required." >&2
  exit 1
}

nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
nvcc --version

cuda_architectures="$(
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
    | sed 's/[[:space:].]//g' \
    | sort -u \
    | paste -sd';' -
)"
if [[ -z "$cuda_architectures" || "$cuda_architectures" == *[!0-9\;]* ]]; then
  echo "Could not derive CUDA architectures from nvidia-smi." >&2
  exit 1
fi
echo "Building for CUDA architectures: $cuda_architectures"

sudo apt-get update
sudo apt-get install -y \
  ca-certificates cmake g++ gfortran git gnupg make python3 wget zlib1g-dev

# Install oneMKL for maximum performance in CPU code paths.
wget -O- \
  https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor \
  | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg >/dev/null
echo \
  "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list >/dev/null
sudo apt-get update
sudo apt-get install -y intel-oneapi-mkl-devel

source /opt/intel/oneapi/setvars.sh
export MKL_THREADING_LAYER=GNU

git clone --branch master --single-branch \
  https://github.com/carbocation/regenie.git regenie-acceleration-cuda
cd regenie-acceleration-cuda

deps_dir="${XDG_CACHE_HOME:-$HOME/.cache}/regenie-build-deps"
bgen_dir="$deps_dir/v1.1.7"
mkdir -p "$deps_dir"
if [ ! -f "$bgen_dir/build/libbgen.a" ]; then
  wget -O "$deps_dir/v1.1.7.tgz" \
    http://code.enkre.net/bgen/tarball/release/v1.1.7
  tar -xzf "$deps_dir/v1.1.7.tgz" -C "$deps_dir"
  (cd "$bgen_dir" && python3 waf configure && python3 waf)
fi

BGEN_PATH="$bgen_dir" \
MKLROOT="$MKLROOT" \
STATIC=1 \
cmake -S . -B build-cuda-mkl \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DREGENIE_WITH_CUDA=ON \
  "-DREGENIE_CUDA_ARCHITECTURES=${cuda_architectures}" \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -mtune=native"

cmake --build build-cuda-mkl --target regenie -j "$(nproc)"
cmake --install build-cuda-mkl --prefix "$HOME/.local"

if ldd "$(command -v regenie)" | grep -q 'libmkl'; then
  echo "Unexpected dynamic oneMKL dependency." >&2
  exit 1
fi
regenie --version
```

The oneAPI environment is needed while configuring and linking this build, but
not at runtime because oneMKL is linked statically. CUDA runtime libraries are
still required.

### CUDA Usage: Selecting the Step 1 backend when running regenie

Note that CUDA is only relevant for Step 1. If regenie is compiled with CUDA
support and run on a machine with CUDA, the default backend and device options
will attempt to use CUDA and will fall back to CPU if not successful. I.e., in
the most common one-GPU case, the standard REGENIE commands can be used without
requiring any additional arguments:

```bash
mkdir -p results
regenie \
  --step 1 \
  --pgen data/cohort \
  --phenoFile data/phenotypes.tsv \
  --covarFile data/covariates.tsv \
  --qt \
  --bsize 1000 \
  --threads "$(nproc)" \
  --out results/step1-auto
```

CUDA-enabled builds default to `--compute-backend auto`, in which Step 1 tries
the selected GPU device (default 0) and falls back to the CPU backend when the
executable starts successfully but no usable CUDA device is available. Use
`--gpu-device` to select a different visible device, `--compute-backend cpu` to
prevent GPU use, or `--compute-backend cuda` when the run should fail instead of
falling back. For example, this quantitative-trait run explicitly requires CUDA
and specifies GPU 0:

```bash
mkdir -p results
regenie \
  --step 1 \
  --pgen data/cohort \
  --phenoFile data/phenotypes.tsv \
  --covarFile data/covariates.tsv \
  --qt \
  --bsize 1000 \
  --threads "$(nproc)" \
  --compute-backend cuda \
  --gpu-device 0 \
  --out results/step1-cuda
```

The paths and `--qt` option are illustrative; replace them with the normal
Step 1 arguments for the dataset and trait type being analyzed. Add
`--compute-backend cpu` to such a command when GPU use is not wanted.

The upstream README is preserved below.

---

<!-- END FORK README PREAMBLE -->
[![build](https://github.com/rgcgithub/regenie/actions/workflows/test.yml/badge.svg)](https://github.com/rgcgithub/regenie/actions/workflows/test.yml)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/rgcgithub/regenie?logo=Github)
[![install with conda](https://img.shields.io/badge/install%20with-conda-brightgreen.svg)](https://anaconda.org/bioconda/regenie)
[![Github All Releases](https://img.shields.io/github/downloads/rgcgithub/regenie/total.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**regenie** is a C++ program for whole genome regression modelling of large [genome-wide association studies](https://en.wikipedia.org/wiki/Genome-wide_association_study).

It is developed and supported by a team of scientists at the Regeneron Genetics Center.

The method has the following properties

- It works on quantitative, binary, and time-to-event traits, including binary traits with unbalanced case-control ratios and time-to-event traits with low event rates
- It can handle population structure and relatedness
- It can process multiple phenotypes at once efficiently
- It is fast and memory efficient 🔥
- For binary traits, it supports Firth logistic regression and an SPA test
- For time-to-event traits, it supports Firth cox regression
- It can perform gene/region-based tests, interaction tests and conditional analyses
- It supports the [BGEN](https://www.well.ox.ac.uk/~gav/bgen_format/), [PLINK](https://www.cog-genomics.org/plink/1.9/formats#bed) bed/bim/fam and [PLINK2](https://www.cog-genomics.org/plink/2.0/formats#pgen) pgen/pvar/psam genetic data formats
- It is ideally suited for implementation in [Apache Spark](https://spark.apache.org/) (see [GLOW](https://projectglow.io/))
- It can be installed with [Conda](https://anaconda.org/bioconda/regenie)

Full documentation for the **regenie** can be found [here](https://rgcgithub.github.io/regenie/).

## Citation 
Mbatchou, J., Barnard, L., Backman, J. et al. Computationally efficient whole-genome regression for quantitative and binary traits. Nat Genet 53, 1097–1103 (2021). https://doi.org/10.1038/s41588-021-00870-7

## License

**regenie** is distributed under an [MIT license](https://github.com/rgcgithub/regenie/blob/master/LICENSE).

## Contact
If you have any questions about regenie please contact

- <jonathan.marchini@regeneron.com>
- <joelle.mbatchou@regeneron.com>

If you want to submit a issue concerning the software please do so
using the **regenie** [Github repository](https://github.com/rgcgithub/regenie/issues).


## Version history
[Version 4.1](https://github.com/rgcgithub/regenie/releases/tag/v4.1) (Timing reduction for single variant association tests; New option --htp to output summary statistics in the [HTP](https://rgcgithub.github.io/remeta/file_formats/#-htp) format; New option --skip-dosage-comp to skip dosage compensation for males in non-PAR chrX regions; Various bug fixes)

[Version 4.0](https://github.com/rgcgithub/regenie/releases/tag/v4.0) (New options `--t2e` and `--eventColList` for time-to-event analysis to specify time-to-event analysis and the event phenotype name, respectively; Fix algorithm used to fit logistic Firth model when using `--write-null-firth` to match closer to the approach used in step 2)

[Version 3.6](https://github.com/rgcgithub/regenie/releases/tag/v3.6) (Bug fix for the approximate Firth test when ultra-rare variants [MAC below 50] are being tested; Address convergence failures & speed-up exact Firth by using warm starts based on null model with just covariates)

[Version 3.5](https://github.com/rgcgithub/regenie/releases/tag/v3.5) (Added CHR/POS columns to snplist output file when using `--write-mask-snplist`; Genotype counts are now reported in the sumstats file when using `--no-split`; Improved efficiency of LOOCV scheme in ridge level 0; Detect carriage return in fam/psam/bim/pvar/sample files; Minor bug fixes)

[Version 3.4.1](https://github.com/rgcgithub/regenie/releases/tag/v3.4.1) (Reduction in memory usage for LD computation when writing to text files; Fix bug rejecting valid PVAR files)

[Version 3.4](https://github.com/rgcgithub/regenie/releases/tag/v3.4) (Reduction in memory usage for LD computation with dosages; Minor bug fixes for LD computation; Bug fix for when carriage returns are in optional input files)

[Version 3.3](https://github.com/rgcgithub/regenie/releases/tag/v3.3) (Faster implementation of approximate Firth LRT; New strategy for approximate Firth LRT with ultra-rare variants; Relaxed convergence criterion of Firth LRT from 1E-4 to 2.5E-4)

[Version 3.2.9](https://github.com/rgcgithub/regenie/releases/tag/v3.2.9) (Switch to robust version of ACAT to handle very small p-values; Bug fix for Step1 when sex chromosome was included in the analysis; Allow for 64 domains when using the 4-column annotation file)

[Version 3.2.8](https://github.com/rgcgithub/regenie/releases/tag/v3.2.8) (New option `--bgi` to specify custom index bgi file accompagnying BGEN file; Relax matching criteria between BGEN and index bgi files to use CPRA instead of variant ID)

[Version 3.2.7](https://github.com/rgcgithub/regenie/releases/tag/v3.2.7) (New option `--force-mac-filter` to apply different MAC filter to subset of SNPs; Extend maximum number of domains to 32 for 4-column anno-file; Update PGEN library)

[Version 3.2.6](https://github.com/rgcgithub/regenie/releases/tag/v3.2.6) (Relax tolerance parameter for null unpenalized logistic regression from 1e-8 to 1e-6; Minor bug fixes)

[Version 3.2.5.3](https://github.com/rgcgithub/regenie/releases/tag/v3.2.5.3) (Fix inflation issue when testing main effect of SNP in GxE model; Minor bug fixes)

[Version 3.2.5](https://github.com/rgcgithub/regenie/releases/tag/v3.2.5) (Use pseudo-data representation algorithm as default in step 2 single variant tests; Use ACAT to get SBAT p-value across POS/NEG models; Bug fix for ACATV when set has a single variant with zero weight)

[Version 3.2.4](https://github.com/rgcgithub/regenie/releases/tag/v3.2.4) (Relaxed the requirement on the minimum number of unique values for QTs to 3; Various bug fixes)

[Version 3.2.3](https://github.com/rgcgithub/regenie/releases/tag/v3.2.3) (Address convergence issues in Firth regression; Various bug fixes)

[Version 3.2.2](https://github.com/rgcgithub/regenie/releases/tag/v3.2.2) (New columns in sumstats file (N_CASES/N_CONTROLS) to output the number of cases/controls when using `--af-cc`; Various bug fixes)

[Version 3.2.1](https://github.com/rgcgithub/regenie/releases/tag/v3.2.1) (New option `--lovo-snplist` to only consider a subset of LOVO masks; Improve efficiency of LOVO for large sets to reduce memory usage; Bug fix for SPA with numerical overflow; For SKAT/ACAT tests with Firth correction, don't include SKAT weights when running Firth on single variants)

[Version 3.2](https://github.com/rgcgithub/regenie/releases/tag/v3.2) (Bug fix for SKAT/SKATO when testing on binary traits using Firth/SPA; Switched name of NNLS joint test to SBAT test altering name of corresponding options and applied Bonferroni correction before reporting its p-value [correcting for minP of 2 tests])

[Version 3.1.4](https://github.com/rgcgithub/regenie/releases/tag/v3.1.4) (New option `--par-region` to specify build to determine bounds for chrX PAR regions; new option `--force-qt` to force QT runs for traits with fewer than 10 values [otherwise will throw an error]; phenotype imputation for missing values is now applied after RINTing when using `--apply-rint`; several bug fixes)

[Version 3.1.2](https://github.com/rgcgithub/regenie/releases/tag/v3.1.2) (Reduction in memory usage for SKAT/SKATO tests; Bug fix for LOVO with SKAT/ACAT tests; Improvements for null Firth logistic algorithm to address reported convergence issues)

[Version 3.1.1](https://github.com/rgcgithub/regenie/releases/tag/v3.1.1) (Reduction in memory usage for SKAT/SKATO tests; Improvements for logistic regressions algorithms to address reported convergence issues)

[Version 3.1](https://github.com/rgcgithub/regenie/releases/tag/v3.1) (Fixed bug in SKAT/SKATO tests when applying Firth/SPA correction; Improved SPA implementation by computing both tail probabilities; New option `--set-singletons` to specify variants to consider as singletons for burden masks; New option `--l1-phenoList` to run level 1 models in Step 1 in parallel across phenotypes; Several bug fixes)

[Version 3.0.3](https://github.com/rgcgithub/regenie/releases/tag/v3.0.3) (Skip BTs where null model fit failed; Bug fix for BURDEN-ACAT; Bug fix when nan/inf values are in phenotype/covariate file)

[Version 3.0.1](https://github.com/rgcgithub/regenie/releases/tag/v3.0.1) (Improve ridge logistic regression in Step 1; Add compilation with Cmake)

[Version 3.0](https://github.com/rgcgithub/regenie/releases/tag/v3.0) (New gene-based tests: SKAT, SKATO, ACATV, ACATO and NNLS [Non-Negative Least Square test]; New GxE and GxG interaction testing functionality; New conditional analysis functionality; see [release page](https://github.com/rgcgithub/regenie/releases/tag/v3.0) for minor additions)

For past releases, see [here](RELEASE_LOG.md).
