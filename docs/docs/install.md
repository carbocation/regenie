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
device-resident across prediction chunks. The backend is disabled by default
and requires CMake 3.18 or newer and the CUDA toolkit.
Build for an NVIDIA A100 (compute capability 8.0) with:

```
BGEN_PATH=<path_to_bgen_lib> cmake -S . -B build-cuda \
  -DREGENIE_WITH_CUDA=ON \
  -DREGENIE_CUDA_ARCHITECTURES=80
cmake --build build-cuda -j
```

The existing `STATIC=1` mode remains available for CUDA-enabled builds. It
retains its usual behavior for oneMKL and the supported host dependencies,
while the CUDA libraries remain dynamically linked:

```
BGEN_PATH=<path_to_bgen_lib> MKLROOT=<path_to_oneMKL> STATIC=1 \
cmake -S . -B build-cuda-mkl \
  -DREGENIE_WITH_CUDA=ON \
  -DREGENIE_CUDA_ARCHITECTURES=80
cmake --build build-cuda-mkl -j
```

CUDA-enabled builds default to `--compute-backend auto`: Step 1 uses device 0
when it is available and otherwise falls back to the CPU backend. Use
`--gpu-device` to select a different visible device, `--compute-backend cpu` to
prevent GPU use, or `--compute-backend cuda` to require CUDA rather than fall
back.

Sample-major device buffers are limited to approximately 1 GB by default.
Set `REGENIE_CUDA_CHUNK_MB` to a positive integer to use a smaller per-buffer
streaming limit; this is primarily useful for validation or sharing a GPU with
other jobs.

CUDA genotype preprocessing uses a resident full-block buffer when the block
fits within the available device-memory budget. By default, the budget is the
smaller of 6,000 MB and 60% of currently free device memory. Set
`REGENIE_CUDA_RESIDENT_MB` to a non-negative integer to override that automatic
budget. CUDA operations reuse the resident full block and its full-row column
slices instead of uploading the same genotype values again. Ordinary k-fold
Step 1 runs leave the normalized block device-only; LOOCV and `--test-l0` also
receive a normalized host copy because those paths read genotype values on the
CPU. Larger blocks fall back to CPU preprocessing. Set the limit to `0` to
disable GPU preprocessing. The validation harness exposes the same setting as
`CUDA_RESIDENT_MB`.

For k-fold binary-trait Level 1 fitting, CUDA can keep the complete Level 1
design resident across folds, ridge parameters, and IRLS iterations. This
eliminates repeated uploads of the same predictions and accumulates each
weighted Gram matrix and score crossproduct on the device. The automatic
limit allows one design copy to use at most 40% of currently available device
memory (capped at 16 GB), because IRLS also needs an equally sized weighted
design workspace. If the two buffers plus a 512 MB reserve do not fit, REGENIE
uses the existing streamed fold path. Set `REGENIE_CUDA_LEVEL1_RESIDENT_MB` to
a non-negative integer to override the maximum size of one design copy; `0`
disables this specialization. The validation harness exposes the override as
`CUDA_LEVEL1_RESIDENT_MB`.

Ordinary k-fold PGEN hardcall runs can avoid materializing and uploading the
host FP64 genotype matrix entirely. When the resident block fits, the reader
passes PGEN's native two-bit hardcalls to CUDA; the device then expands,
masks, mean-imputes, residualizes, and scales the block directly into its
resident FP64 buffer. A 1,000-variant by 500,000-sample block therefore crosses
the host/device boundary as approximately 125 MB of hardcalls instead of 4 GB
of doubles. Dosage mode, LOOCV, `--test-l0`, MAF-dependent priors, CPU runs,
and blocks above the resident limit retain the general host-matrix path. Set
`REGENIE_STEP1_PGEN_PACKED=0` to disable this specialization for matched A/B
validation; the validation harness exposes it as `STEP1_PGEN_PACKED`.

Packed-resident k-fold runs solve each Level 0 ridge system directly with one
Cholesky factorization per ridge parameter. This avoids computing a complete
eigendecomposition when only the configured ridge solutions are needed, and
the resulting coefficients are multiplied by the already resident genotype
block. The existing eigendecomposition implementation remains the fallback for
other input paths and for any ridge grid containing zero. Set
`REGENIE_CUDA_LEVEL0_CHOLESKY=0` to force the previous implementation for a
matched A/B comparison; the validation harness exposes the same switch as
`CUDA_LEVEL0_CHOLESKY`.

The independent k-fold systems are submitted to reusable CUDA streams so their
Cholesky factorizations, solves, and held-out predictions can overlap. The
number of streams follows the configured fold count; it is not tied to a
particular phenotype count. Set `REGENIE_CUDA_LEVEL0_FOLD_BATCH=0` to retain
the sequential Cholesky implementation for a matched A/B comparison. The
validation harness exposes this switch as `CUDA_LEVEL0_FOLD_BATCH`.

Chromosome-grouped prediction uploads copy row slices directly from the
column-major Level 1 design into bounded device buffers. This avoids building
a full temporary host matrix for every prediction chunk. Set
`REGENIE_CUDA_DIRECT_GROUPED_UPLOAD=0` to restore the materialized upload path
for a matched A/B comparison.

Large resident uploads use two reusable pinned host chunks so host packing can
overlap transfer to the device. `REGENIE_CUDA_PINNED_STAGING_MB` controls the
size of each chunk (64 MB by default); set it to `0` to restore direct pageable
uploads. Pinned staging adds only twice the configured chunk size to host
memory, independent of the genotype block size.

For CUDA Step 1 runs on PGEN input, the next genotype block is decoded into a
reusable second host buffer while the current block is processed on the GPU.
The buffer holds native packed hardcalls when the specialization above is
active and an FP64 matrix otherwise. Prefetching is enabled when it is no larger than
`REGENIE_STEP1_PGEN_PREFETCH_MB` (4,096 MB by default). Set the value to `0` to
disable prefetching or lower it to cap the additional host-memory allowance.
The PGEN decoder itself reuses one sample tile per worker and fuses
validation, masking, allele-frequency accumulation, and missing-value
imputation into a single scatter pass. It materializes eight variants at a
time by default so each sample-major destination write is contiguous; set
`REGENIE_STEP1_PGEN_TILE_VARIANTS` to an integer from 1 through 64 to tune the
memory-bandwidth tradeoff. Each worker retains one tile buffer, so its host
memory cost is `tile variants * samples * 8` bytes per worker.

For development, the repository includes a single A100 validation command.
It builds both backends, checks matrix shapes and failure paths, benchmarks
both the Level 0 eigensystem and nonlinear Level 1 workloads, runs quantitative,
binary, count, time-to-event, top-SNP, k-fold, and LOOCV Step 1 jobs, and
compares the CPU and CUDA LOCO files. It also records peak device-memory use
for the CUDA benchmark and each end-to-end case:

```
BGEN_PATH=<path_to_bgen_lib> scripts/test_step1_cuda.sh
```

Application runs using `--step1-profile` report how many blocks used CUDA
genotype preprocessing, along with pinned-staging upload counts and bytes.
Packed-hardcall runs additionally report their block count, transfer bytes,
and device expansion time.
PGEN-prefetched runs add a `pgen_pipeline` scope containing decoder service,
foreground wait, and estimated overlap time. A `pgen_ingest` scope separates
aggregate worker time inside pgenlib from fused validation, masking,
allele-frequency calculation, missing-value imputation, and matrix
materialization, and reports packed variant and byte counts when host
materialization is skipped. On Linux it also reports process I/O deltas sampled around
each PGEN block; these distinguish logical reads, which may be page-cached,
from storage-backed reads reported by `/proc/self/io`.

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
