# Stage 1 production benchmark

Updated 2026-07-22 after profiling large, multi-trait Level 1 workloads.

## Result

The production baseline is upstream REGENIE v4.1.2 (`5f924b9`). Its directly
measured N=500,000, M=700,000, P=1 quantitative run on the eight-core N2 takes
8,207.13 seconds. The best measured branch A100 pipeline completes that same
workload in 110.34 seconds, a direct 74.38x placement speedup. The best
measured branch CPU result on the same N2 is 6,182.49 seconds, a direct 1.33x
software speedup over upstream.

Matched upstream P=8 and P=32 Stage 1 runs were not completed. The measured
upstream P=1 quantitative time can still serve as a conservative floor for the
larger multi-trait workloads: adding outcomes does not remove the genotype,
Level 0, or single-outcome work already present at P=1. On that basis,
`b5f86e9` on A100 is at least 6.44x faster for P=32 quantitative, 12.20x for
P=8 binary, and 11.38x for P=8 survival. These are lower bounds, not estimates
of the exact upstream multi-trait runtime.

| Model | N | M | P | Fork placement and full runtime | Upstream v4.1.2 comparator | Upstream basis | Minimum speedup over upstream |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: |
| Quantitative, 0-10% input missingness | 500,000 | 700,000 | 32 | `b5f86e9` A100: 21.2 min estimated | at least 136.79 min | Direct P=1 quantitative measurement used as a floor | at least 6.44x |
| Binary, 0-10% input missingness | 500,000 | 700,000 | 8 | `b5f86e9` A100: 11.21 min estimated | at least 136.79 min | Direct P=1 quantitative measurement used as a conservative Level 0 floor | at least 12.20x |
| Survival, 0-10% input missingness | 500,000 | 700,000 | 8 | `b5f86e9` A100: 12.01 min estimated | at least 136.79 min | Direct P=1 quantitative measurement used as a conservative Level 0 floor | at least 11.38x |
| Quantitative, 0-10% input missingness | 500,000 | 700,000 | 8 | `b5f86e9` N2: 107.08 min estimated | at least 136.79 min | Direct P=1 quantitative measurement used as a floor | at least 1.28x |

The fork runtimes in this table are estimated full runs: measured `b5f86e9`
Level 1 and output times are combined with matched measured Level 0 times. The
upstream floor is measured, but it has fewer outcomes; consequently only the
minimum speedup is reported. No upstream number is shown for the N=50,000
binary and survival checks later in this report. The only Stage 1 upstream
measurement has N=500,000 and P=1 quantitative, and scaling it across both N
and model would not be defensible.

## Workloads

All N=500,000 headline rows use the same M=700,000 real-LD-derived
hard-call PGEN, five-fold cross-validation, a block size of 1,000, and the
default Level 0 and Level 1 ridge grids. The expanded cohort is synthetic, but
the fixture retains local LD, allele-frequency variation, genotype
missingness, and population structure from the source genotypes.

| Model | Jointly fitted outcomes | Outcome construction | Missingness |
| --- | ---: | --- | --- |
| Quantitative | 32 | Correlated quantitative traits | Input missingness increases by trait from 0% to 10%; Step 1 applies its normal mean-imputation behavior |
| Binary | 8 | Prevalences of 1%, 2%, 5%, 10%, 20%, 30%, 40%, and 50% | Input missingness increases by trait from 0% to 10% |
| Survival | 8 | Event fractions from 10% through 80% | Input missingness increases by trait from 0% to 10% |

The A100 system is an `a2-highgpu-1g` with one 40 GB A100 and six physical CPU
cores exposed as 12 vCPUs. The CPU system is an `n2-standard-16` configured to
use eight physical cores without SMT. Both x86 builds use oneMKL 2026.1. No
measurement in this report runs multiple REGENIE processes concurrently on a
GPU.

## Current implementation

The current validated Stage 1 implementation is `b5f86e9`. It includes four
important execution changes:

1. Level 1 now computes the selected chromosome-group predictions while the
   design is still resident. Output reads a compact temporary prediction
   matrix instead of rereading the much larger Level 0 design and repeating
   the model calculation.
2. Multi-trait runs prefetch the next phenotype's Level 0 matrix while fitting
   the preceding phenotype. The buffer is bounded to two phenotype slots, and
   each file can be read in parallel by up to four threads by default.
3. Weighted binary and survival Gram matrices compute only one triangle. CUDA
   uses double-precision GEMM tiles and mirrors the result; oneMKL uses its
   symmetric matrix routines on CPU.
4. Cox fitting reuses the resident design for lambda selection and validation,
   and avoids a design multiplication when the coefficient vector is exactly
   zero.

These are execution changes, not new model hyperparameters. The prediction
cache and Level 0 prefetch have environment-variable off switches for
diagnosis, but are enabled by default. The implementation does not use
`float32`.

## Current multi-trait performance

### Quantitative: N=500,000, M=700,000, P=32

Against upstream v4.1.2, the estimated `b5f86e9` A100 full run is at least
6.44x faster. The comparator is the measured 136.79-minute upstream P=1
runtime used as a floor; upstream P=32 was not run.

| Component | Upstream v4.1.2 (`5f924b9`) | Current branch `b5f86e9` | Current speedup vs upstream |
| --- | ---: | ---: | ---: |
| Level 0 | not measured at P=32 | 608.17 s | -- |
| Level 1 | not measured at P=32 | 623.70 s | -- |
| Prediction output | not measured at P=32 | 41.61 s | -- |
| Level 1 plus output | not measured at P=32 | 665.31 s | -- |
| Full run | at least 8,207.13 s; measured P=1 floor | about 1,273.5 s | at least 6.44x |

In the `b5f86e9` Level 1 run, the retained Level 0 reads represented by the
profile total 619.96 seconds. Prefetch overlaps 117.28 seconds of that work,
but the foreground still waits 502.68 seconds. More GPU arithmetic would not
materially improve this workload until the input path changes.

The storage path is already close to its hardware allowance. These 32 Level 0
files contain 455.04 GB of double-precision predictions. The A100 system uses
a 2 TB balanced Persistent Disk, whose documented size-based allowance is
about 700 MiB/s; the measured reader rate is approximately 700 MiB/s. Reading
larger staged chunks increased the four-trait Level 1 wall time from 89.47 to
107.43 seconds, and increasing kernel readahead produced a similar regression,
so neither change was retained. Lossless zstd reduced a representative 1 GiB
slice by only 4.2%. A material improvement now requires faster scratch storage
such as Local SSD, or a redesign that avoids writing 455 GB between levels;
changing the reader again is not a high-confidence optimization.

All 32 current-branch LOCO files passed exact-output regression checks.

### Binary: N=500,000, M=700,000, P=8

Against upstream v4.1.2, the estimated `b5f86e9` A100 full run is at least
12.20x faster. The comparator is the measured 136.79-minute upstream P=1
quantitative runtime used as a conservative Level 0 floor; upstream P=8
binary was not run.

| Component | Upstream v4.1.2 (`5f924b9`) | Current branch `b5f86e9` | Current speedup vs upstream |
| --- | ---: | ---: | ---: |
| Level 0 | not measured for P=8 binary | 356.80 s | -- |
| Level 1 | not measured for P=8 binary | 300.40 s | -- |
| Weighted Gram construction | not instrumented in upstream | 209.23 s | -- |
| Prediction output | not measured for P=8 binary | 9.49 s | -- |
| Full run | at least 8,207.13 s; measured P=1 quantitative floor | about 672.48 s | at least 12.20x |

The current branch performs 508 IRLS iterations and passes the prediction
regression check with a maximum absolute printed difference of `1e-6` and
identical missingness masks.

Mean Level 1 GPU utilization is 84.7%. During the accelerated Gram kernels,
the A100 reaches 100% utilization and approximately 350-407 W.

### Survival: N=500,000, M=700,000, P=8

Against upstream v4.1.2, the estimated `b5f86e9` A100 full run is at least
11.38x faster. The comparator is the measured 136.79-minute upstream P=1
quantitative runtime used as a conservative Level 0 floor; upstream P=8
survival was not run.

| Component | Upstream v4.1.2 (`5f924b9`) | Current branch `b5f86e9` | Current speedup vs upstream |
| --- | ---: | ---: | ---: |
| Level 0 | not measured for P=8 survival | 461.25 s | -- |
| Level 1 | not measured for P=8 survival | 243.12 s | -- |
| Weighted Gram construction | not instrumented in upstream | 138.17 s | -- |
| Lambda selection | not instrumented in upstream | 0.50 s | -- |
| Validation | not instrumented in upstream | 11.61 s | -- |
| Prediction output | not measured for P=8 survival | 9.77 s | -- |
| Full run | at least 8,207.13 s; measured P=1 quantitative floor | about 720.88 s | at least 11.38x |

Mean Level 1 GPU utilization is 65.8% at 274 W. All eight current-branch LOCO
files, including missing values, passed exact-output regression checks.

## Current-branch checks across N and CPU hardware

Revision `b5f86e9` was also tested at N=50,000, M=700,000 without changing its
defaults. These rows characterize `b5f86e9`; they do not use an intermediate
fork revision as a performance baseline.

No upstream estimate is reported because the available upstream Stage 1 run
has N=500,000, P=1, and a quantitative model. Scaling across both N and model
would be speculative.

| Model | N | M | P | Upstream v4.1.2 full run | `b5f86e9` full run | Level 0 | Level 1 | Output |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Binary, complete | 50,000 | 700,000 | 8 | not available; upstream data are N=500,000, P=1 quantitative | 62.00 s | 22.72 s | 35.03 s | 2.68 s |
| Survival, complete | 50,000 | 700,000 | 8 | not available; upstream data are N=500,000, P=1 quantitative | 57.81 s | 26.67 s | 26.85 s | 2.71 s |

These complete end-to-end runs show that the implementation works at a tenfold
smaller N and for two iterative models. They are not used to claim a speedup
over upstream.

The eight-core N2 comparison uses N=500,000, M=700,000, and P=8 quantitative
traits:

| Metric | Upstream v4.1.2 (`5f924b9`) | Current branch `b5f86e9` symmetric oneMKL | Current speedup vs upstream |
| --- | ---: | ---: | ---: |
| Level 1 | not measured at P=8 | 227.25 s | -- |
| Full run | at least 8,207.13 s; measured P=1 floor | 6,424.80 s estimated | at least 1.28x |

The committed Level 1 includes 131.84 seconds for Gram construction and
produces eight LOCO files that pass exact-output regression checks.

CPU Level 0 remains the larger problem. It took 6,195.76 seconds in the matched
N2 run; residualization accounted for 3,563.67 seconds and Gram construction
for 1,471.21 seconds. The process averaged roughly four busy cores during Level
0 despite having eight available. Improving that path is more valuable for a
CPU-only Stage 1 run than further tuning the now-four-minute Level 1 phase.

## Direct P=1 upstream anchor

The independent single-quantitative-trait production anchors below remain
useful for machine placement. Every row uses N=500,000, M=700,000, and P=1;
these are not the multi-trait `b5f86e9` comparisons above.

| Implementation | System | Full run | Speedup over upstream v4.1.2 |
| --- | --- | ---: | ---: |
| Best branch A100 pipeline | A100 | 110.34 s | 74.38x |
| Best branch CPU pipeline | 8-core N2 | 6,182.49 s | 1.33x |
| Upstream REGENIE v4.1.2 | 8-core N2 | 8,207.13 s | 1.00x |
| Best branch T4 pipeline | T4 | 4,692.19 s | 1.75x |

The CPU rows are a direct same-system comparison. The other rows compare
hardware placements, not software revisions. The T4 is saturated but slow and
is not a useful Step 1 target. Upstream is CPU-only, so there is no reason to
benchmark it on either GPU system. Exact branch revisions remain in the raw
TSVs.

## Validation and raw data

The final revision is `b5f86e9`. Its A100 CUDA+oneMKL build passes the CPU,
CUDA-auto, Step 2 CPU, and Cox test targets. Its N2 oneMKL build passes the CPU,
auto-backend, and Cox targets.

The new measurements are in
[`2026-07-22-step1-level1.tsv`](2026-07-22-step1-level1.tsv). Earlier Stage 1
runs, including the upstream, T4, single-trait, and first multi-trait anchors,
remain in [`2026-07-19-production.tsv`](2026-07-19-production.tsv). Direct and
lower-bound upstream comparisons are collected in
[`2026-07-22-step1-upstream.tsv`](2026-07-22-step1-upstream.tsv).
