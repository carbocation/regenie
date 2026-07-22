# Stage 1 production benchmark

Updated 2026-07-22 after profiling large, multi-trait Level 1 workloads.

## Result

The production baseline is upstream REGENIE v4.1.2 (`5f924b9`). Its directly
measured N=500,000, M=700,000, P=1 quantitative run on the eight-core N2 takes
8,207.13 seconds. The best measured branch A100 pipeline completes that same
workload in 110.34 seconds, a direct 74.38x placement speedup. The best
measured branch CPU result on the same N2 is 6,182.49 seconds, a direct 1.33x
software speedup over upstream.

The matched upstream P=32 quantitative run takes 14,585.02 seconds on the
eight-core N2. The `b5f86e9` A100 pipeline is therefore 11.45x faster for the
same N, M, P, model, and missingness pattern. For P=8 binary traits, a measured
upstream Level 0 and four measured phenotype fits support a 6.1-6.5 hour full
runtime projection, making the current A100 pipeline 32.7-34.7x faster. The
matched P=8 survival evidence supports a 230.19-239.78 minute upstream
projection, making the current A100 pipeline 19.16-19.96x faster. On the same
eight-core N2, the matched P=8 quantitative evidence supports a
156.75-163.17 minute upstream projection, making the current CPU
implementation 1.46-1.52x faster.

| Model | N | M | P | Fork placement and full runtime | Upstream v4.1.2 comparator | Upstream basis | Speedup over upstream |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: |
| Quantitative, 0-10% input missingness | 500,000 | 700,000 | 32 | `b5f86e9` A100: 21.2 min estimated | 243.08 min | Matched P=32 quantitative full run | 11.45x |
| Binary, 0-10% input missingness | 500,000 | 700,000 | 8 | `b5f86e9` A100: 11.21 min estimated | 366.56-388.74 min projected | Measured Level 0 and first four phenotype fits; remaining fits extrapolated | 32.70-34.68x |
| Survival, 0-10% input missingness | 500,000 | 700,000 | 8 | `b5f86e9` A100: 12.01 min estimated | 230.19-239.78 min projected | Measured Level 0 and first four phenotype fits; remaining fits bounded by the complete-outcome survival trend and TIME4 cost | 19.16-19.96x |
| Quantitative, 0-10% input missingness | 500,000 | 700,000 | 8 | `b5f86e9` N2: 107.08 min estimated | 156.75-163.17 min projected | 152 measured upstream Level 0 blocks; exact first-eight Level 1 and output timings reused from the matched P=32 run | 1.46-1.52x |

The fork runtimes in this table are estimated full runs: measured `b5f86e9`
Level 1 and output times are combined with matched measured Level 0 times. The
upstream P=32 quantitative comparator is a matched full run. The P=8 binary
range is a projection from the partially completed matched run, as described
below. The P=8 quantitative CPU range combines a bounded matched Level 0 run
with exact per-trait timings from the completed P=32 run, whose first eight
phenotype columns are identical. The survival row is a range from a matched
run, as described below. Direct N=50,000 binary and survival comparisons are
also shown later in this report. Cross-model scaling is not used as a
substitute.

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

Against upstream v4.1.2, the estimated `b5f86e9` A100 full run is 11.45x
faster. The comparator is a matched 243.08-minute upstream P=32 run on the
eight-core N2, not a single-trait floor.

| Component | Upstream v4.1.2 (`5f924b9`) | Current branch `b5f86e9` | Current speedup vs upstream |
| --- | ---: | ---: | ---: |
| Level 0 | at least 9,208.83 s of internally timed work | 608.17 s | at least 15.14x |
| Level 1 | 2,928.23 s | 623.70 s | 4.69x |
| Prediction output | 2,019.60 s | 41.61 s | 48.54x |
| Level 1 plus output | 4,947.83 s | 665.31 s | 7.44x |
| Full run | 14,585.02 s measured | about 1,273.5 s | 11.45x |

Upstream reports per-operation timings rather than phase wall-clock
boundaries. Its Level 0 entry is the sum of the 711 block reads,
residualizations, working-matrix calculations, and ridge fits, so it excludes
untimed setup and transition overhead. The Level 1 and output rows sum the 32
per-phenotype timings printed by upstream. Full process wall time is the
headline comparison.

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

Against upstream v4.1.2, the estimated `b5f86e9` A100 full run is 32.7-34.7x
faster. The upstream range is based on the matched P=8 binary run, not a
single-trait quantitative floor.

| Component | Upstream v4.1.2 (`5f924b9`) | Current branch `b5f86e9` | Current speedup vs upstream |
| --- | ---: | ---: | ---: |
| Level 0 | about 8,404.55 s measured | 356.80 s | 23.56x |
| Level 1 | 13,084.05-14,415.22 s projected | 300.40 s | 43.56-47.99x |
| Weighted Gram construction | not instrumented in upstream | 209.23 s | -- |
| Prediction output | about 504.90 s projected | 9.49 s | 53.20x |
| Full run | 21,993.50-23,324.67 s projected | about 672.48 s | 32.70-34.68x |

Upstream completed all 711 Level 0 blocks and four of eight Level 1 fits. The
1%, 2%, 5%, and 10% prevalence traits took 2,297.47, 2,019.58, 1,970.71, and
1,625.49 seconds. The lower full-run projection fits a logit-prevalence trend
through those points; the upper assigns every remaining trait the full
observed 10%-prevalence time. Prediction output uses 8/32 of the directly
measured upstream quantitative P=32 output time. The range avoids assigning a
single observed rate to all four more-common traits.

The current branch performs 508 IRLS iterations and passes the prediction
regression check with a maximum absolute printed difference of `1e-6` and
identical missingness masks.

Mean Level 1 GPU utilization is 84.7%. During the accelerated Gram kernels,
the A100 reaches 100% utilization and approximately 350-407 W.

### Survival: N=500,000, M=700,000, P=8

Against upstream v4.1.2, the estimated `b5f86e9` A100 full run is
19.16-19.96x faster. The comparator is a matched P=8 survival projection, not
a quantitative single-trait floor.

| Component | Upstream v4.1.2 (`5f924b9`) | Current branch `b5f86e9` | Current speedup vs upstream |
| --- | ---: | ---: | ---: |
| Level 0 | about 8,468.46 s measured | 461.25 s | 18.36x |
| Level 1 | 4,839.31-5,415.12 s projected | 243.12 s | 19.90-22.27x |
| Weighted Gram construction | not instrumented in upstream | 138.17 s | -- |
| Lambda selection | not instrumented in upstream | 0.50 s | -- |
| Validation | not instrumented in upstream | 11.61 s | -- |
| Prediction output | about 503.40 s projected | 9.77 s | 51.51x |
| Full run | 13,811.17-14,386.98 s projected | about 720.88 s | 19.16-19.96x |

Upstream completed all 711 Level 0 blocks and the first four Level 1 fits. The
10%, 20%, 30%, and 40% event-fraction traits took 816.85, 704.84, 700.15, and
638.66 seconds. The lower remaining-fit projection preserves the TIME5-TIME8
versus TIME4 shape measured in the complete N=50,000 survival run. The upper
assigns each remaining trait the full observed TIME4 cost. Prediction output
uses the 503.40 seconds directly measured for the same first eight columns in
the upstream quantitative P=32 run.

Mean Level 1 GPU utilization is 65.8% at 274 W. All eight current-branch LOCO
files, including missing values, passed exact-output regression checks.

## Current-branch checks across N and CPU hardware

Revision `b5f86e9` was also tested at N=50,000, M=700,000 without changing its
defaults. These rows characterize `b5f86e9`; they do not use an intermediate
fork revision as a performance baseline.

Both matched upstream runs are measured directly.

| Model | N | M | P | Upstream v4.1.2 full run | `b5f86e9` full run | Current speedup vs upstream | Current Level 0 | Current Level 1 | Current output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Binary, complete | 50,000 | 700,000 | 8 | 2,591.97 s measured | 62.00 s | 41.80x | 22.72 s | 35.03 s | 2.68 s |
| Survival, complete | 50,000 | 700,000 | 8 | 1,938.50 s measured | 57.81 s | 33.53x | 26.67 s | 26.85 s | 2.71 s |

The upstream binary run directly matches N, M, P, prevalences, complete
outcomes, and oneMKL host library; the systems differ because upstream has no
CUDA backend. Upstream spent 1,418.24 seconds in the eight phenotype fits and
27.52 seconds in prediction output. Seven LOCO files are byte-identical to the
current branch; the eighth is identical after predictions are rounded to five
significant digits.

The upstream survival run spent 763.02 seconds in the eight phenotype fits
and 5.54 seconds in prediction output. A paired check of the first LOCO file
covered 1.15 million predictions with a maximum absolute difference of
`1.065e-4`, mean absolute difference of `1.41e-5`, RMS difference of
`1.82e-5`, and no difference above `1e-3`.

The eight-core N2 comparison uses N=500,000, M=700,000, and P=8 quantitative
traits:

| Metric | Upstream v4.1.2 (`5f924b9`) | Current branch `b5f86e9` symmetric oneMKL | Current speedup vs upstream |
| --- | ---: | ---: | ---: |
| Level 0 | 8,167.91-8,553.05 s projected | 6,195.76 s | 1.32-1.38x |
| Level 1 | 733.71 s measured | 227.25 s | 3.23x |
| Prediction output | 503.40 s measured | 5.33 s | 94.46x |
| Full run | 9,405.02-9,790.16 s projected | 6,424.80 s estimated | 1.46-1.52x |

The upstream Level 1 and output values are not scaled from the P=32 totals.
They sum the directly reported timings for PHENO1 through PHENO8 in the
completed P=32 run; those are exactly the eight columns used here. The Level 0
run was stopped after 152 complete blocks covering 151,347 variants. Its lower
projection models the fixture's 689 full and 22 chromosome-end blocks
separately and retains the observed fixed overhead. The upper projection
scales the observed partial process wall by the 711 total blocks. This bounded
run replaces the former P=1 floor with a matched P=8 range.

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
projected upstream comparisons are collected in
[`2026-07-22-step1-upstream.tsv`](2026-07-22-step1-upstream.tsv).
