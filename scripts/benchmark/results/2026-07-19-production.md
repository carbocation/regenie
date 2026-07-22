# Stage 1 production benchmark

Updated 2026-07-22 after profiling large, multi-trait Level 1 workloads.

## Result

The A100 remains the right machine for Stage 1. The latest work cuts Level 1
time by 1.47x for eight binary traits and 1.96x for eight survival models at
N=500,000 and M=700,000. A 32-trait quantitative run improves less in Level 1
because reading the retained Level 0 files is now the critical path, but its
prediction-output time falls from 17.3 minutes to 42 seconds.

The CPU path also improves: an eight-trait quantitative Level 1 run on eight
physical N2 cores falls from 289 to 227 seconds. The result is not specific to
one A100 workload or to a statistical tuning change. All calculations remain
double precision, and the ridge grids, convergence criteria, and fitted models
are unchanged.

| Model | N | M | Traits | System | Level 1 before | Level 1 now | Level 1 speedup | Full run before | Full run now |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Quantitative, 0-10% input missingness | 500,000 | 700,000 | 32 | A100 | 1,049.96 s | 623.70 s | 1.68x | 45.0 min measured | 21.2 min estimated |
| Binary, 0-10% input missingness | 500,000 | 700,000 | 8 | A100 | 442.26 s | 300.40 s | 1.47x | 13.60 min measured | 11.21 min estimated |
| Survival, 0-10% input missingness | 500,000 | 700,000 | 8 | A100 | 475.54 s | 243.12 s | 1.96x | 15.89 min measured | 12.01 min estimated |
| Quantitative, 0-10% input missingness | 500,000 | 700,000 | 8 | 8-core N2 | 289.04 s | 227.25 s | 1.27x | 108.21 min measured | 107.08 min estimated |

The current A100 and N2 Level 1 measurements use retained Level 0 files. The
estimated full-run totals replace the matched control's Level 1 and output
times with the new measurements while keeping its measured Level 0 time. This
avoids rerunning an unchanged Level 0 merely to obtain a new total. The smaller
N=50,000 checks below are complete end-to-end runs.

## Workloads

All current headline rows use the same N=500,000, M=700,000 real-LD-derived
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

## What changed

Four changes account for most of the improvement:

1. Level 1 now computes the selected chromosome-group predictions while the
   design is still resident. Output reads a compact temporary prediction
   matrix instead of rereading the much larger Level 0 design and repeating
   the model calculation.
2. Multi-trait runs prefetch one phenotype's Level 0 matrix while fitting the
   current phenotype. The buffer is bounded to two phenotype slots, and each
   file can be read in parallel by up to four threads by default.
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

## A100 results by model

### Quantitative: 32 traits

| Component | Before | Now | Change |
| --- | ---: | ---: | ---: |
| Level 0 | 608.17 s | unchanged | -- |
| Level 1 | 1,049.96 s | 623.70 s | 1.68x faster |
| Prediction output | 1,039.60 s | 41.61 s | 25.0x faster |
| Level 1 plus output | 2,089.56 s | 665.31 s | 3.14x faster |
| Full run | about 2,697.7 s | about 1,273.5 s | 2.12x faster |

In the current Level 1 run, the retained Level 0 reads represented by the
profile total 619.96 seconds. Prefetch overlaps 117.28 seconds of that work,
but the foreground still waits 502.68 seconds. More GPU arithmetic would not
materially improve this workload until the input path changes.

All 32 LOCO files are byte-for-byte identical to the control.

### Binary: eight traits

| Component | Before | Now | Change |
| --- | ---: | ---: | ---: |
| Level 0 | 356.80 s | unchanged | -- |
| Level 1 | 442.26 s | 300.40 s | 1.47x faster |
| Weighted Gram construction | 346.75 s | 209.23 s | 1.66x faster |
| Prediction output | 11.11 s | 9.49 s | 1.17x faster |
| Full run | 815.96 s | about 672.48 s | 1.21x faster |

Both runs perform 508 IRLS iterations and select the same models. Of roughly
90.5 million nonmissing printed prediction values, 526 differ in their decimal
rendering. That is 0.00058%; the maximum absolute difference is `1e-6`, with
the same missingness masks.

Mean Level 1 GPU utilization falls slightly, from 88.6% to 84.7%, while elapsed
time improves by 142 seconds. This is a useful reminder that utilization is a
diagnostic, not the objective. During the accelerated Gram kernels, the A100
reaches 100% utilization and approximately 350-407 W.

### Survival: eight models

| Component | Before | Now | Change |
| --- | ---: | ---: | ---: |
| Level 0 | 461.25 s | unchanged | -- |
| Level 1 | 475.54 s | 243.12 s | 1.96x faster |
| Weighted Gram construction | 229.18 s | 138.17 s | 1.66x faster |
| Lambda selection | 105.88 s | 0.50 s | 214x faster |
| Validation | 53.24 s | 11.61 s | 4.58x faster |
| Prediction output | 9.69 s | 9.77 s | unchanged |
| Full run | 953.23 s | about 720.88 s | 1.32x faster |

Mean Level 1 GPU utilization rises from 55.7% at 230 W to 65.8% at 274 W, but
the important result is that elapsed time falls by 49%. All eight LOCO files,
including missing values, are text-identical to the control.

## Generalization checks

The same code was tested at N=50,000 without changing its defaults.

| Model | N | M | Traits | Earlier full run | Current full run | Earlier Level 1 | Current Level 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Binary, complete | 50,000 | 700,000 | 8 | 95.34 s | 62.00 s | 63.68 s | 35.03 s |
| Survival, complete | 50,000 | 700,000 | 8 | 133.90 s | 57.81 s | 97.87 s | 26.85 s |

Those complete end-to-end runs improve by 1.54x and 2.32x respectively. The
benefit therefore survives a tenfold change in sample count and applies to two
different iterative models.

On the eight-core N2, the N=500,000, M=700,000, eight-trait quantitative Level
1 control took 289.04 seconds. The symmetric oneMKL path takes 227.25 seconds,
including 131.84 seconds for Gram construction, and produces eight
byte-identical LOCO files. A second optimized implementation routed through
Eigen took 231.27 seconds, so the small difference between those two candidates
should be treated as run noise rather than a tuned-machine advantage.

CPU Level 0 remains the larger problem. It took 6,195.76 seconds in the matched
N2 run; residualization accounted for 3,563.67 seconds and Gram construction
for 1,471.21 seconds. The process averaged roughly four busy cores during Level
0 despite having eight available. Improving that path is more valuable for a
CPU-only Stage 1 run than further tuning the now-four-minute Level 1 phase.

## Overall Stage 1 placement

The earlier single-quantitative-trait production anchor remains useful for
machine placement. It uses N=500,000 and M=700,000.

| Implementation | System | Full run |
| --- | --- | ---: |
| Current retained A100 Level 0 path | A100 | 110.34 s |
| Current CPU reference | 8-core N2 | 6,182.49 s |
| Upstream v4.1.2 | 8-core N2 | 8,207.13 s |
| CUDA reference | T4 | 4,692.19 s |

The T4 is saturated but slow and is not a useful Step 1 target. Upstream is
CPU-only, so there is no reason to benchmark it on either GPU system. The A100
result is the production anchor for Step 1; CPU work is still worthwhile for
portability and for understanding which optimizations generalize.

## Validation and raw data

The final revision is `b5f86e9`. Its A100 CUDA+oneMKL build passes the CPU,
CUDA-auto, Step 2 CPU, and Cox test targets. Its N2 oneMKL build passes the CPU,
auto-backend, and Cox targets.

The new measurements are in
[`2026-07-22-step1-level1.tsv`](2026-07-22-step1-level1.tsv). Earlier Stage 1
runs, including the upstream, T4, single-trait, and first multi-trait anchors,
remain in [`2026-07-19-production.tsv`](2026-07-19-production.tsv).
