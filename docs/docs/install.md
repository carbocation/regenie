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

For development, the repository includes a single A100 validation command.
It builds both backends, checks matrix shapes and failure paths, benchmarks
both the Level 0 eigensystem and nonlinear Level 1 workloads, runs quantitative,
binary, count, time-to-event, top-SNP, k-fold, and LOOCV Step 1 jobs, and
compares the CPU and CUDA LOCO files. It also records peak device-memory use
for the CUDA benchmark and each end-to-end case:

```
BGEN_PATH=<path_to_bgen_lib> scripts/test_step1_cuda.sh
```

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
