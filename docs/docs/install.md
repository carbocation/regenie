##Download

The **regenie** source code is hosted on
[Github](https://github.com/rgcgithub/regenie).

##Installation

<div class="bs-callout bs-callout-default">
  <h4>Pre-requisites</h4>
    <b>regenie</b> requires compilation with 
<a href="https://gcc.gnu.org">GCC</a> version >= 5.1 (on Linux) 
or Clang version >=3.3 (on Mac OSX). 
It also requires having GFortran library installed.
    </div>

### Pre-compiled binaries

Pre-compiled binaries are available in the 
[Github repository](https://github.com/rgcgithub/regenie/releases).
These are provided for Linux (including Centos7) and Mac OSX 
computing environments and are statically linked. 
For the Linux binaries, users should have GLIBC version >= 2.22 installed.
Additionally, they are provided compiled with Intel MKL library which
will provide speedups for many of the operations done in **regenie**. 


### Standard installation
1. **regenie** requires the
  [BGEN library](https://enkre.net/cgi-bin/code/bgen/dir?ci=trunk) so
  you will need to download and install that library.
2. Edit the BGEN_PATH variable in the `Makefile`
   to the BGEN library path.
3. On the command line type `make` while in the main source code directory.
4. This should produce the executable called `regenie`.

**regenie** has been enhanced to allow for gzip compressed input 
(for phenotype/covariate files) and output (for association results files)
 using the Boost Iostream library. 
If this library is installed on the system, you should compile using 
`make HAS_BOOST_IOSTREAM=1`. 

Furthermore, we have enabled compilation of **regenie** with
the Intel Math Kernel (MKL) library. You first need to have it installed 
on your system and modify the MKLROOT variable in the `Makefile`
to the installed MKL library path.

### With CMake
You can compile the binary using CMake version >=3.13 (instead of `make` as above).
```
mkdir -p build
cd build
BGEN_PATH=<path_to_bgen_lib> cmake ..
make
```
This will generate the binary in the `build/` subdirectory. 
To use with Boost Iostreams and/or Intel MKL library,
add the corresponding flags before the `cmake` command on line 3
(e.g. `BGEN_PATH=<path_to_bgen_lib> HAS_BOOST_IOSTREAM=1 cmake ..`).

### Experimental CUDA Step 1 backend

The CUDA backend accelerates the FP64 matrix work in Step 1, including Gram
and phenotype crossproducts, symmetric eigensystems, batched ridge prediction,
weighted logistic/Poisson/Cox updates, diagonal-penalty Cholesky solves, and
chunked final-model influence solves. Level 0 products, design Gram matrices,
weighted Hessians, score crossproducts, linear predictors, and
chromosome-grouped LOCO predictions are streamed through bounded device
buffers; reusable eigensystems and Cholesky factorizations remain
device-resident across prediction chunks. The backend is disabled by default,
requires CMake 3.18 or newer and the CUDA toolkit, and currently requires a
dynamic build.
Build for an NVIDIA A100 (compute capability 8.0) with:

```
BGEN_PATH=<path_to_bgen_lib> cmake -S . -B build-cuda \
  -DREGENIE_WITH_CUDA=ON \
  -DREGENIE_CUDA_ARCHITECTURES=80
cmake --build build-cuda -j
```

For an NVIDIA T4, use `-DREGENIE_CUDA_ARCHITECTURES=75`. Multiple
architectures may be supplied as a semicolon-separated CMake list when a
portable binary is required.

Select it in Step 1 with `--compute-backend cuda`; use `--gpu-device` when
more than one CUDA device is visible. `--compute-backend auto` uses CUDA when
the binary contains the backend and the requested device is available,
otherwise it uses the CPU backend.

Sample-major device buffers are limited to approximately 1 GB by default.
Set `REGENIE_CUDA_CHUNK_MB` to a positive integer to use a smaller per-buffer
streaming limit; this is primarily useful for validation or sharing a GPU with
other jobs.

CUDA genotype preprocessing uses a resident full-block buffer when the block
fits within `REGENIE_CUDA_RESIDENT_MB` (1,024 MB by default). CUDA operations
reuse the resident full block and its full-row column slices instead of
uploading the same genotype values again. Ordinary k-fold Step 1 runs leave the
normalized block device-only; LOOCV and `--test-l0` also receive a normalized
host copy because those paths read genotype values on the CPU. Larger blocks
fall back to CPU preprocessing. Set the limit to `0` to disable GPU
preprocessing, or raise it deliberately when a larger block fits comfortably
in device memory. The validation harness exposes the same setting as
`CUDA_RESIDENT_MB`.

The CUDA backend uses FP64 throughout by default. On devices with weak FP64
throughput, the opt-in setting `REGENIE_CUDA_GRAM_PRECISION=fp32` converts each
bounded genotype chunk to FP32 for its Gram product and accumulates the chunk
results into an FP64 matrix. Phenotype crossproducts, eigendecompositions,
ridge solves, and predictions remain FP64. This mode trades numerical accuracy
for speed and must be validated against the default FP64 output for each target
workload; unset the variable or set it to `fp64` to retain the default path.
Each outer streaming chunk is subdivided into 128-sample Gram products by
default to limit FP32 accumulation drift without shrinking transfers or other
CUDA operations. Override this positive sample count with
`REGENIE_CUDA_FP32_GRAM_CHUNK_SAMPLES`; smaller values favor accuracy and
larger values favor throughput. The validation harness exposes these settings
as `CUDA_GRAM_PRECISION` and `CUDA_FP32_GRAM_CHUNK_SAMPLES`.

For development, the repository includes a hardware-parameterized GPU
validation command. It builds both backends, checks matrix shapes and failure
paths, benchmarks both the Level 0 eigensystem and nonlinear Level 1 workloads,
runs quantitative, binary, count, time-to-event, top-SNP, k-fold, and LOOCV
Step 1 jobs, and compares the CPU and CUDA LOCO files. It also records peak
device-memory use for the CUDA benchmark and each end-to-end case. The defaults
target an A100 and write to `build-cuda/a100-validation`:

```
BGEN_PATH=<path_to_bgen_lib> scripts/test_step1_cuda.sh
```

Validate a T4 in a separate build and results directory with:

```
BGEN_PATH=<path_to_bgen_lib> \
BUILD_DIR=build-cuda-t4 \
CUDA_ARCHITECTURES=75 \
GPU_VALIDATION_LABEL=t4 \
CUDA_STREAM_CHUNK_MB=64 \
scripts/test_step1_cuda.sh
```

`VALIDATION_DIR` can override the results directory, and `GPU_DEVICE` selects
the CUDA device. Benchmark dimensions remain configurable with
`BENCHMARK_BLOCKS`, `BENCHMARK_SAMPLES`, `BENCHMARK_PHENOTYPES`, and
`BENCHMARK_REPEATS`. One warmup iteration is run by default before steady-state
measurements; set `BENCHMARK_WARMUP_REPEATS` to a different positive integer.
Benchmark records report first-warmup wall time, mean warmup wall time,
steady-state wall time, backend-accounted time, and the remaining unaccounted
host/orchestration time. The validation requires `compute-sanitizer` by
default; set `RUN_COMPUTE_SANITIZER=0` only when running it separately or when
performing a deliberately reduced smoke test.

Application runs using `--step1-profile` report both additive stage totals and
scope-level breakdowns for genotype preprocessing, cross-validation matrix
construction, and Level 0 ridge prediction. A final end-to-end record also
separates initialization, Level 0 wall time, Level 1 preparation, Level 1
fitting, and output. The scope records separate backend compute, transfers, and
host/orchestration time, making them useful for distinguishing accelerated
linear algebra from data packing and result assembly. The preprocessing scope
also records how many blocks used the CUDA path versus the bounded CPU
fallback.

`scripts/compare_numeric_files.py` compares large whitespace-delimited outputs
one line at a time and automatically uses a vectorized NumPy engine when NumPy
is installed, with a dependency-free Python fallback. Its default tolerances
remain strict for full-precision validation. For REGENIE text written with the
default six significant digits, pass `--output-significant-digits 6` to
additionally tolerate one unit in the last serialized digit without weakening
comparisons beyond the precision present in the files. `--engine` can force
`numpy` or `python` when comparing implementations.

An opt-in end-to-end benchmark generates a deterministic PLINK BED dataset in
the validation results directory, runs matched CPU and CUDA Step 1 jobs,
compares every LOCO output, and reports elapsed time, speedup, and peak device
memory. The generated data is not stored in the repository. This A100 example
creates 20,000 samples by 20,000 variants (a BED file of approximately 100 MB):

```
BGEN_PATH=<path_to_bgen_lib> \
BUILD_DIR=build-cuda-a100 \
CUDA_ARCHITECTURES=80 \
GPU_VALIDATION_LABEL=a100-large \
RUN_SYNTHETIC_BENCHMARK=1 \
SYNTHETIC_SAMPLES=20000 \
SYNTHETIC_VARIANTS=20000 \
SYNTHETIC_PHENOTYPES=4 \
SYNTHETIC_BSIZE=512 \
SYNTHETIC_THREADS=8 \
scripts/test_step1_cuda.sh
```

The synthetic dimensions, phenotype count, chromosome count, block size,
thread count, and random seed can be changed with `SYNTHETIC_SAMPLES`,
`SYNTHETIC_VARIANTS`, `SYNTHETIC_PHENOTYPES`, `SYNTHETIC_CHROMOSOMES`,
`SYNTHETIC_BSIZE`, `SYNTHETIC_THREADS`, and `SYNTHETIC_SEED`. The BED payload
uses approximately `samples * variants / 4` bytes. Generation fails before
writing when the result would exceed `SYNTHETIC_MAX_BED_GB` (4 GiB by
default). The generated files and a JSON manifest containing dimensions,
seed, size, and BED SHA-256 remain under the validation directory so the run
can be archived. Set `RUN_COMPUTE_SANITIZER=0` for a benchmark-only rerun only
if the same binary has already passed the sanitizer validation separately.

The normal build remains CUDA-free, so CPU development and regression testing
can continue on macOS. CUDA compilation is also checked in CI without running
GPU code.

### With Docker
Alternatively, you can use a Docker image to run **regenie**. 
A guide to using docker is available on 
the [Github page](https://github.com/rgcgithub/regenie/wiki/Using-docker).

### With conda
To install with [conda](https://anaconda.org/bioconda/regenie), you can use the following commands:
```
# create new environment
conda create -n regenie_env -c conda-forge -c bioconda regenie
# load it
conda activate regenie_env
```



##Computing requirements

We have tested **regenie** on 64-bit Linux and 64-bit Mac OSX computing environments.
 
Note that for Mac OSX computing environments, compiling is done without OpenMP, as the library is not built-in by default and has to be installed separately. 

### Memory usage
In both Step 1 and Step 2 of a **regenie** run the genetic data file is
read once, in blocks of SNPs, so at no point is the full dataset ever stored in
memory.

**regenie** uses a dimension reduction approach using ridge regression
  to produce a relatively small set of genetic predictors, that are
  then used to fit a whole-genome regression model. These genetic
  predictors are stored in memory by default, and can be relatively
  large if many phenotypes are stored at once.

For example, if there are \(P\) phenotypes, \(M\) SNPs and \(N\) samples, and a
block size of \(B\) SNPs is used with \(R\) ridge parameters,
 then **regenie** needs to store roughly \(N\times M/B\times R\)
doubles per phenotype, which is 8Gb per phenotype when \(M=500,000,
N=400,000, B =1,000,R=5\) and 200Gb in total when \(P=25\).

However, the `--lowmem` option can be used to avoid that memory usage,
at negligible extra computational cost, by writing temporary files to disk.

### Threading

**regenie** can take advantage of multiple cores using threading. The
number of threads can be specified using the `--threads` option.

**regenie** uses the [Eigen library](http://eigen.tuxfamily.org/index.php?title=Main_Page) for 
efficient linear algebra operations and this uses threading where possible.

For PLINK bed/bim/fam files, PLINK2 pgen/pvar/psam files, as well as BGEN v1.2 files with 8-bit encoding (format used for UK Biobank
500K imputed data), step 2 of **regenie** has been optimized by 
using multithreading through [OpenMP](https://www.openmp.org).

When running the SKAT/ACAT gene-based tests, we recommend to use at most 2 threads and 
instead parallelize the runs over partitions of the genome (e.g. groups of genes).

### For Windows platforms

If you are on a Windows machine, we recommend to use [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install) (WSL)
to install a Ubuntu distribution so that you will be able to run REGENIE
from a Linux terminal.
You can download pre-compiled REGENIE binaries from the [Github repository](https://github.com/rgcgithub/regenie/releases) 
(note that you will need to install the `libgomp1` library).

Note: from your Windows command prompt, you can run REGENIE using `wsl regenie`.
