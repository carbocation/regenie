# Stage 1 production benchmark

Updated 2026-07-22 after profiling large, multi-trait Level 1 workloads.

## Result

The A100 remains the right machine for Stage 1. Compared with parent revision
`c312f41`, revision `b5f86e9` cuts Level 1 time by 1.47x for eight binary traits
and 1.96x for eight survival models at N=500,000 and M=700,000. This is a
comparison with the immediately preceding accelerated branch, **not with
upstream REGENIE**. In the 32-trait quantitative workload, prediction-output
time falls from 17.3 minutes at `c312f41` to 42 seconds at `b5f86e9`.

The CPU path also improves: relative to a matched full-matrix Gram diagnostic
build, the committed `b5f86e9` symmetric oneMKL path reduces an eight-trait
quantitative Level 1 run on eight physical N2 cores from 289 to 227 seconds.
That CPU comparison is described separately below. All calculations remain
double precision, and the ridge grids, convergence criteria, and fitted models
are unchanged.

| Model | N | M | Traits | System | `c312f41` Level 1 | `b5f86e9` Level 1 | `b5f86e9` / `c312f41` speedup | `c312f41` full run | `b5f86e9` full run |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Quantitative, 0-10% input missingness | 500,000 | 700,000 | 32 | A100 | 1,049.96 s | 623.70 s | 1.68x | 45.0 min measured | 21.2 min estimated |
| Binary, 0-10% input missingness | 500,000 | 700,000 | 8 | A100 | 442.26 s | 300.40 s | 1.47x | 13.60 min measured | 11.21 min estimated |
| Survival, 0-10% input missingness | 500,000 | 700,000 | 8 | A100 | 475.54 s | 243.12 s | 1.96x | 15.89 min measured | 12.01 min estimated |

The `b5f86e9` A100 measurements use retained Level 0 files. Its estimated
full-run totals replace the matched `c312f41` run's Level 1 and output times
with the `b5f86e9` measurements while keeping the measured `c312f41` Level 0
time. This avoids rerunning an unchanged Level 0 merely to obtain a new total.
The N2 comparison uses a separate full-matrix Gram control described below.
The N=50,000 checks are complete end-to-end runs.

### Comparison map

| Question | Workload | Compared implementation | Baseline | Upstream comparison? |
| --- | --- | --- | --- | --- |
| Did the new Level 1 work help the large multi-trait A100 workloads? | N=500,000, M=700,000; P=32 quantitative or P=8 binary/survival | `b5f86e9` | Parent revision `c312f41` | No |
| Does the result generalize to a smaller N? | N=50,000, M=700,000, P=8 binary/survival | `b5f86e9` | Earlier multi-trait revision `fa7506f` | No |
| Does the symmetric CPU Gram calculation help? | N=500,000, M=700,000, P=8 quantitative on N2 | Committed `b5f86e9` oneMKL path | Matched `b5f86e9` diagnostic build using the original full-matrix Gram | No |
| How much faster is the accelerated CPU branch than released REGENIE? | N=500,000, M=700,000, P=1 quantitative on N2 | `114ef81` | Upstream v4.1.2 (`5f924b9`) | Yes |

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

## What changed

Four changes distinguish `b5f86e9` from `c312f41`:

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

## A100 comparison: `b5f86e9` versus `c312f41`

### Quantitative: N=500,000, M=700,000, P=32

| Component | `c312f41` | `b5f86e9` | Ratio |
| --- | ---: | ---: | ---: |
| Level 0 | 608.17 s measured | 608.17 s reused; not rerun | -- |
| Level 1 | 1,049.96 s | 623.70 s | 1.68x faster |
| Prediction output | 1,039.60 s | 41.61 s | 25.0x faster |
| Level 1 plus output | 2,089.56 s | 665.31 s | 3.14x faster |
| Full run | about 2,697.7 s | about 1,273.5 s | 2.12x faster |

In the `b5f86e9` Level 1 run, the retained Level 0 reads represented by the
profile total 619.96 seconds. Prefetch overlaps 117.28 seconds of that work,
but the foreground still waits 502.68 seconds. More GPU arithmetic would not
materially improve this workload until the input path changes.

All 32 `b5f86e9` LOCO files are byte-for-byte identical to the matched
`c312f41` files.

### Binary: N=500,000, M=700,000, P=8

| Component | `c312f41` | `b5f86e9` | Ratio |
| --- | ---: | ---: | ---: |
| Level 0 | 356.80 s measured | 356.80 s reused; not rerun | -- |
| Level 1 | 442.26 s | 300.40 s | 1.47x faster |
| Weighted Gram construction | 346.75 s | 209.23 s | 1.66x faster |
| Prediction output | 11.11 s | 9.49 s | 1.17x faster |
| Full run | 815.96 s | about 672.48 s | 1.21x faster |

Both revisions perform 508 IRLS iterations and select the same models. Of roughly
90.5 million nonmissing printed prediction values, 526 differ in their decimal
rendering. That is 0.00058%; the maximum absolute difference is `1e-6`, with
the same missingness masks.

Mean Level 1 GPU utilization falls slightly, from 88.6% to 84.7%, while elapsed
time improves by 142 seconds. This is a useful reminder that utilization is a
diagnostic, not the objective. During the accelerated Gram kernels, the A100
reaches 100% utilization and approximately 350-407 W.

### Survival: N=500,000, M=700,000, P=8

| Component | `c312f41` | `b5f86e9` | Ratio |
| --- | ---: | ---: | ---: |
| Level 0 | 461.25 s measured | 461.25 s reused; not rerun | -- |
| Level 1 | 475.54 s | 243.12 s | 1.96x faster |
| Weighted Gram construction | 229.18 s | 138.17 s | 1.66x faster |
| Lambda selection | 105.88 s | 0.50 s | 214x faster |
| Validation | 53.24 s | 11.61 s | 4.58x faster |
| Prediction output | 9.69 s | 9.77 s | unchanged |
| Full run | 953.23 s | about 720.88 s | 1.32x faster |

Mean Level 1 GPU utilization rises from 55.7% at 230 W to 65.8% at 274 W, but
the important result is that elapsed time falls by 49%. All eight `b5f86e9`
LOCO files, including missing values, are text-identical to the matched
`c312f41` files.

## Generalization checks across N and CPU hardware

Revision `b5f86e9` was also tested at N=50,000, M=700,000 without changing its
defaults. These rows compare it with revision `fa7506f`, an earlier retained
multi-trait A100 implementation; they are not upstream comparisons.

| Model | N | M | Traits | `fa7506f` full run | `b5f86e9` full run | `fa7506f` Level 1 | `b5f86e9` Level 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Binary, complete | 50,000 | 700,000 | 8 | 95.34 s | 62.00 s | 63.68 s | 35.03 s |
| Survival, complete | 50,000 | 700,000 | 8 | 133.90 s | 57.81 s | 97.87 s | 26.85 s |

Relative to `fa7506f`, the complete `b5f86e9` end-to-end runs are 1.54x faster
for binary traits and 2.32x faster for survival traits. The benefit therefore
survives a tenfold change in sample count and applies to two different
iterative models.

On the eight-core N2, a matched `b5f86e9` diagnostic build using the original
full-matrix CPU Gram calculation took 289.04 seconds for Level 1 at N=500,000,
M=700,000, and P=8 quantitative traits. The committed symmetric oneMKL path in
`b5f86e9` takes 227.25 seconds, including 131.84 seconds for Gram construction,
and produces eight byte-identical LOCO files. This isolates the CPU Gram
change; it is neither an upstream comparison nor a comparison with
`c312f41`. A second diagnostic build routed the symmetric operation through
Eigen and took 231.27 seconds, so its small difference from explicit oneMKL
should be treated as run noise.

CPU Level 0 remains the larger problem. It took 6,195.76 seconds in the matched
N2 run; residualization accounted for 3,563.67 seconds and Gram construction
for 1,471.21 seconds. The process averaged roughly four busy cores during Level
0 despite having eight available. Improving that path is more valuable for a
CPU-only Stage 1 run than further tuning the now-four-minute Level 1 phase.

## Overall Stage 1 placement

The independent single-quantitative-trait production anchors below remain
useful for machine placement. Every row uses N=500,000, M=700,000, and P=1;
these are not the multi-trait `b5f86e9` comparisons above.

| Implementation | Revision | System | Full run |
| --- | --- | --- | ---: |
| Retained A100 Level 0 pipeline | `dc64b54` | A100 | 110.34 s |
| Accelerated CPU branch | `114ef81` | 8-core N2 | 6,182.49 s |
| Upstream REGENIE v4.1.2 | `5f924b9` | 8-core N2 | 8,207.13 s |
| CUDA branch | `114ef81` | T4 | 4,692.19 s |

The direct upstream comparison in this table is `114ef81` versus upstream
v4.1.2 (`5f924b9`) on the same N2. The other rows compare hardware placements,
not software revisions. The T4 is saturated but slow and is not a useful Step
1 target. Upstream is CPU-only, so there is no reason to benchmark it on either
GPU system.

## Validation and raw data

The final revision is `b5f86e9`. Its A100 CUDA+oneMKL build passes the CPU,
CUDA-auto, Step 2 CPU, and Cox test targets. Its N2 oneMKL build passes the CPU,
auto-backend, and Cox targets.

The new measurements are in
[`2026-07-22-step1-level1.tsv`](2026-07-22-step1-level1.tsv). Earlier Stage 1
runs, including the upstream, T4, single-trait, and first multi-trait anchors,
remain in [`2026-07-19-production.tsv`](2026-07-19-production.tsv).
