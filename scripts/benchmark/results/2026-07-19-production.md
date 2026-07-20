# Production performance checkpoint: 2026-07-19

This report began as a performance snapshot of `114ef81` at production-like
Step 1 scale, with upstream v4.1.2 (`5f924b9`) as a CPU baseline. It now also
includes the Cox design-residency change in `50ca3c1` and the resident weighted
ridge solve in `87270e6`, followed by the small-block Level 0 pipeline in
`3b285ad`, the production-width GPU pipeline in `dc64b54`, and the
multi-phenotype and non-Gaussian extension in `fa7506f`. Unless noted otherwise,
timings are external wall time. Step 1 runs use five-fold cross-validation,
`--bsize 1000`, `--lowmem`, and the default Level 0 and Level 1 ridge grids.

## Test systems and workloads

| System | Compute | Guest-visible CPU and RAM | Build |
| --- | --- | --- | --- |
| `a2-highgpu-1g` | NVIDIA A100-SXM4 40 GB, 400 W | 6 physical Xeon 2.20 GHz cores / 12 vCPU, 83 GiB | CUDA sm80, oneMKL 2026.1 |
| `n1-standard-8-t4` | NVIDIA Tesla T4 15 GB, 70 W | 4 physical Xeon 2.20 GHz cores / 8 vCPU, 29 GiB | CUDA sm75, oneMKL 2026.1 |
| `n2-standard-16-smt-off` | CPU only | 8 physical Xeon 2.80 GHz cores / 8 vCPU, 62 GiB | native x86-64, oneMKL 2026.1 |

The headline input is a hard-call PGEN with 500,000 synthetic samples and
700,000 variants. Synthetic genomes were produced by resampling 1000 Genomes
donors within super-population groups at 32-variant LD-block boundaries. The
fixture therefore retains local LD, allele-frequency variation, missingness,
and population structure, but the expanded samples are not real people. It
has one quantitative phenotype with h2=0.3 and `SUPERPOP` as a categorical
covariate. The PGEN, PVAR, PSAM, phenotype, and covariate files were copied to
all three systems and their SHA-256 digests matched before the CPU comparison.
Every retained run below uses that one phenotype without phenotype or
covariate missingness unless the multi-phenotype section says otherwise.

## Current cold-cache A100 result

The production-width optimization was measured on the full 500,000-sample,
700,000-variant fixture after clearing the page cache. The baseline is the
previous retained implementation at `3b285ad`; the optimized run uses
`dc64b54`. Both use the same input, model, 12 host threads, and A100 system.

| Cold run | Wall | REGENIE total | Level 0 | Level 1 | Level 0 + 1 | Level 0 GPU util. | Level 0 power |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `3b285ad` | 188.29 s | 185.507 s | 153.885 s | 17.864 s | 171.749 s | 58.1% | 214.8 W |
| `dc64b54` | 110.34 s | 107.709 s | 90.312 s | 0.533 s | 90.845 s | 93.5% | 308.0 W |

Level 0 plus Level 1 is 47.1% faster, while internal total time falls by
41.9%. This is a throughput improvement, not a utilization-only result. During
the optimized Level 0, median utilization is 95%, p10/p90 are 95%/97%, and
94.2% of samples are at or above 90%. Mean power is 77.0% of the 400 W limit.
Whole-run utilization is 78.9% because setup, prediction output, and other
mostly host-side work remain in the trace.

The retained path keeps fold systems and the assembled Level 1 design on the
device, normalizes each Level 0 prediction block on the GPU, overlaps its
low-memory file write with the next block, reuses static phenotype and
preprocessing inputs, and uses registered packed-PGEN buffers. Static,
contiguous PGEN worker partitions were also important for cold input: relative
to the immediately preceding candidate, aggregate PGEN service time fell from
102.10 to 64.87 seconds and foreground wait from 14.34 to 0.81 seconds.

Commit `fa7506f` retains this single-trait path and extends its Level 0 solve,
normalization, and asynchronous low-memory output to multiple dense outcomes,
binary traits, and survival models. Multi-QT input missingness handled by the
usual per-trait mean imputation is included because the fitted outcomes are
dense. Phenotype-specific sparse masks still use the established fallback.
The optimized build passes the CPU, CUDA-auto, and Cox test targets.

Small-fixture single-trait output is byte-identical to the control. At full
scale, 2 of 11,500,023 printed numeric values differ from the earlier
CPU-normalized path; the maximum absolute difference is `1e-9` and the maximum
relative difference is `2.61e-6`. A 500,000-sample, 20,000-variant regression
check completed in 7.83 seconds, confirming that the original one-trait path
still takes the optimized branch.

Each cold row is one run. The per-block timing and utilization trace support
the mechanism, but another cold repetition is warranted before treating the
exact 110.34-second wall time as a stable estimate.

## Original cross-system Step 1 snapshot

| Run | Revision | Backend | N | M | Host threads | Wall | REGENIE total | Level 0 | Peak host RSS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A100, repetition 1 | `114ef81` | CUDA | 500,000 | 700,000 | 12 | 188.14 s | 186.63 s | 152.82 s | 14.97 GiB |
| A100, repetition 2 | `114ef81` | CUDA | 500,000 | 700,000 | 12 | 186.43 s | 184.95 s | 153.58 s | 14.97 GiB |
| T4 | `114ef81` | CUDA | 500,000 | 700,000 | 4 | 4,692.19 s | 4,690.15 s | 4,567.98 s | 14.88 GiB |
| Eight-core N2 | `114ef81` | CPU | 500,000 | 700,000 | 8 | 6,182.49 s | 6,182.06 s | 6,133.79 s | 14.98 GiB |
| Eight-core N2 | `5f924b9` | CPU | 500,000 | 700,000 | 8 | 8,207.13 s | 8,206.77 s | 7,693.37 s | 14.75 GiB |

The A100 result is a warm-cache checkpoint: REGENIE reported 78.3 GB of
logical PGEN reads but only 96.6 MB of storage-backed reads. The asynchronous
PGEN pipeline nevertheless shows that input is not on the critical path in
this run: 23.50 s of decoder service overlapped computation, with only 55.7 ms
of foreground wait across 711 blocks.

The A100 repetition differs by only 0.9% in wall time and 1.0% in measured GPU
energy. Using the median of the two runs, A100 is 25.1x faster end-to-end and
29.8x faster in Level 0 than T4. The gap reaches 97.9x for Gram construction,
while residualization and ridge are 8.6x and 8.2x faster, respectively. A100
also uses a median 9.66 Wh of measured GPU energy versus 85.25 Wh on T4, an
8.8x energy reduction despite A100's much higher instantaneous power envelope.

The T4 is 1.32x faster than the eight-core N2 CPU run and 2.87x faster than its
own four-physical-core host. That is real acceleration, but modest compared
with its 25.1x gap from A100. The production-width CPU result also confirms why
the earlier independent-marker fixture was not a sound anchor: it exercised a
materially different genotype workload.

## Multi-phenotype Step 1 at 50,000 samples

The multi-phenotype check uses a deterministic, equal-stratum subset of the
same real-LD fixture: 50,000 samples (10,000 from each of five population
groups), all 700,000 variants, and 32 standardized quantitative traits. Traits
2 through 32 have a target correlation of 0.4 with trait 1; their observed
correlations range from 0.395 to 0.406. The physical PGEN subset and its
phenotype and covariate files had matching SHA-256 digests on A100 and N2.

The complete phenotype file has no missing values. In the second file,
per-trait input missingness increases linearly from 0% to 10%, for 80,000
missing values in total. Of the 50,000 samples, 40,463 have at least one
missing trait and 9,537 are complete across all 32. Multi-QT Step 1 uses its
default per-trait mean imputation, so both runs fit dense 50,000-sample traits;
the second case measures the cost of realistic missing input under that
default behavior, not phenotype-specific sparse fitting.

| System and revision | Input missingness | Wall | Level 0 | Level 1 | Output | Peak host RSS | Average process CPU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A100 CUDA, `114ef81` | none | 227.51 s | 120.806 s | 72.085 s | 30.915 s | 2.87 GiB | 124% |
| A100 CUDA, `fa7506f` | none | 113.80 s | 54.457 s | 24.387 s | 30.983 s | 2.13 GiB | 161% |
| A100 CUDA, `114ef81` | 0-10% by trait | 227.44 s | 120.953 s | 71.911 s | 30.888 s | 2.87 GiB | 124% |
| A100 CUDA, `fa7506f` | 0-10% by trait | 113.30 s | 54.346 s | 24.321 s | 30.943 s | 2.13 GiB | 162% |
| Eight-core N2, `114ef81` | none | 1,423.27 s | 986.913 s | 410.919 s | 24.080 s | 2.80 GiB | 599% |
| Eight-core N2, `114ef81` | 0-10% by trait | 1,407.77 s | 981.663 s | 406.551 s | 18.205 s | 2.80 GiB | 617% |

For complete phenotypes, `fa7506f` cuts A100 wall time by 50.0%, Level 0 by
54.9%, and Level 1 by 66.2% relative to the original checkpoint. The
missing-input result is effectively identical: wall time falls by 50.2% and
Level 0 by 55.1%. The retained path solves all outcomes together, normalizes
and reorders their predictions on the device, transfers them into registered
host storage, and writes phenotype-major blocks asynchronously. Level 1 also
uses the resident-design path without rereading and transforming the same
low-memory data twice.

The complete and missing `fa7506f` runs differ by only 0.4% in wall time. The
original paired A100 runs were similarly close, and the N2 missing-input run
was 1.1% faster. These measurements provide no evidence that default
multi-QT mean imputation has a material performance cost at this scale.

The optimized A100 is now 12.5x faster than the eight-core N2 end-to-end,
18.1x in Level 0, and 16.9x in Level 1. Prediction output remains host-bound:
it takes 30.98 seconds on A100 versus 24.08 seconds on N2. The grouped output
path still uploads 45.5 GB of design data for only about 64 ms of device
compute, so output-side design reuse remains a concrete next opportunity.

All four A100 rows above were started after clearing the page cache. An earlier
pair mixed cache states and varied mainly in Level 1 and output despite nearly
identical Level 0 time; those runs are not used here. This is also a reminder
that one low-memory run is not enough to characterize post-Level-0 timing when
the generated prediction files remain cacheable in host memory.

| A100 revision and input | Whole-run GPU util. | Level 0 GPU util. | Level 1/output GPU util. | Whole-run power | Level 0 power | Peak memory | GPU energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `114ef81`, complete | 33.1% | 27.0% | 41.6% | 88.3 W | 69.7 W | 2,138 MiB | 5.58 Wh |
| `fa7506f`, complete | 31.0% | 44.3% | 19.6% | 104.0 W | 133.3 W | 5,020 MiB | 3.28 Wh |
| `114ef81`, 0-10% missing | 32.9% | 26.4% | 41.4% | 87.5 W | 68.8 W | 2,138 MiB | 5.53 Wh |
| `fa7506f`, 0-10% missing | 31.6% | 45.8% | 19.7% | 95.9 W | 121.0 W | 5,020 MiB | 3.02 Wh |

The optimized path raises complete-case Level 0 utilization from 27.0% to
44.3% while more than halving its elapsed time. Whole-run utilization is
roughly flat because Level 0 and Level 1 finish much sooner and the unchanged
31-second output phase occupies a larger share of the run. Measured GPU energy
falls by 41% for the complete case. This is a throughput gain accompanied by
better Level 0 use of the device, not an attempt to maximize utilization in
isolation.

Level 0 ridge transfer is now the largest cost inside the generalized solve:
23.85 of 48.83 seconds for the complete run, versus 21.17 seconds of device
compute and 3.81 seconds of host work. Registering the destination buffer per
block reduced that transfer time by 34% and cut full-run wall time by 9.6%
relative to the first generalized candidate. A reusable pinned staging buffer
was also tested on a matched 20,000-variant slice, but its extra host copy
raised transfer time from 0.640 to 1.078 seconds, so it was not retained.

These are single production-thread measurements on the current branch, not a
thread sweep. No T4 Step 1 or A100 CPU run was added for this workload.

## Binary and time-to-event Step 1 at 50,000 samples

The non-Gaussian checks use the same physical 50,000-sample, 700,000-variant
PGEN and covariates as the quantitative-trait runs. The binary fixture has
eight traits with exact prevalences of 1%, 2%, 5%, 10%, 20%, 30%, 40%, and
50%. The survival fixture has eight TIME/EVENT pairs with exact observed event
fractions from 10% through 80%. Neither fixture has phenotype missingness, and
the phenotype files matched by checksum across systems.

| Model and system | Revision | Models | Wall | Level 0 | Level 1 | Output | Peak host RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Binary, A100 CUDA | `114ef81` | 8 | 152.55 s | 64.370 s | 76.910 s | 7.724 s | 2.26 GiB |
| Binary, eight-core N2 | `114ef81` | 8 | 3,308.72 s | 937.320 s | 2,365.552 s | 4.553 s | 2.50 GiB |
| Survival, eight-core N2 | `114ef81` | 8 | 2,723.69 s | 948.247 s | 1,769.544 s | 4.523 s | 3.32 GiB |
| Survival, A100 CUDA before residency | `114ef81` | 8 | 1,278.30 s | 70.270 s | 1,196.565 s | 7.705 s | 3.13 GiB |
| Survival, A100 CUDA with residency | `50ca3c1` | 8 | 232.08 s | 84.424 s | 137.478 s | 7.859 s | 3.23 GiB |

Binary Level 1 makes substantially better use of A100 than the quantitative
traits: it averages 67.2% GPU utilization and 248 W during the logistic fit.
A100 is 21.69x faster than N2 end-to-end, 14.56x in Level 0, and 30.76x in
logistic Level 1. N2 averages 82.6% busy across its eight cores. Output again
favors the CPU host, at 4.55 seconds versus 7.72 seconds on A100. Within N2
Level 1, 1,035.82 seconds are measured Gram work and 1,311.62 seconds remain
in host orchestration, so the CPU opportunity is not just a BLAS substitution.

The original Cox path is the opposite. It repeatedly uploads the same dense
Level 1 design and averages only 17.4% utilization and 63 W while fitting.

Commit `50ca3c1` keeps that design resident for prediction, weighted products,
and score calculation. Across eight models it uploads 11.38 GB of resident
design data and reuses it 1,364 times. Cox Level 1 becomes 8.70x faster, the
whole run becomes 5.51x faster, and measured GPU energy falls from 22.82 Wh to
7.02 Wh. All eight LOCO files are byte-for-byte identical before and after the
change.

The optimized run followed other A100 work and read its PGEN from cache,
whereas the baseline was storage-backed. That does not account for the gain:
optimized Level 0 was actually 14.15 seconds slower, and the 1,059-second
reduction occurs in Cox Level 1.

The original A100 path is 2.13x faster than N2 for survival; residency raises
that to 11.74x end-to-end and 12.87x in Cox Level 1. N2 averages 82.2% busy
across eight cores. As in the other multi-model runs, prediction output is
faster on N2 (4.52 seconds) than on A100 (7.86 seconds). The CPU calculation is
unchanged by `50ca3c1`, so the `114ef81` N2 result is the matching CPU anchor
for both A100 rows.

### Keeping the weighted ridge solve on the GPU

Once the design matrix was resident, each IRLS iteration still copied its
weighted Gram matrix and right-hand side to the host, then immediately copied
them back for the penalized Cholesky solve. Commit `87270e6` adds an optional
backend operation that carries the weighted products directly into the solve
on the device. The CPU fallback and the older two-call backend interface remain
available.

The following controls and committed runs were paired on the same warm input.
The control includes Cox design residency but not the resident solve.

| Model | Revision | Wall | Level 0 | Level 1 | Output | GPU energy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Binary control | `50ca3c1` | 170.09 s | 72.006 s | 88.020 s | 7.835 s | 7.64 Wh |
| Binary resident solve | `87270e6` | 146.43 s | 72.725 s | 63.603 s | 7.843 s | 7.32 Wh |
| Binary generalized Level 0 | `fa7506f` | 95.34 s | 21.602 s | 63.678 s | 7.843 s | 6.33 Wh |
| Survival control | `50ca3c1` | 232.15 s | 84.748 s | 137.239 s | 7.841 s | 7.05 Wh |
| Survival resident solve | `87270e6` | 189.98 s | 84.188 s | 95.587 s | 7.868 s | 6.31 Wh |
| Survival generalized Level 0 | `fa7506f` | 133.90 s | 25.802 s | 97.869 s | 7.849 s | 6.25 Wh |

The committed binary run is 13.9% faster end-to-end and 27.7% faster in Level
1. Its measured Level 1 transfers fall from 27.74 seconds to 2.86 seconds.
The committed survival run is 18.2% faster end-to-end and 30.4% faster in Cox
Level 1, with Level 1 transfers falling from 41.49 seconds to 2.46 seconds.
Level 0 and output time are effectively controls here; the gain is where the
round trip was removed.

Commit `fa7506f` then applies the multi-outcome device-normalization and
asynchronous-output path to both models. Relative to `87270e6`, binary wall
time falls by 34.9% and Level 0 by 70.3%; survival wall time falls by 29.5% and
Level 0 by 69.4%. Level 1 and output are effectively unchanged, which isolates
the improvement to the intended shared front half of Step 1. The binary Level
0 averages 80.9% GPU utilization, and survival Level 0 averages 76.3%, up from
36.8% and 33.4% in the corresponding earlier runs.

All eight binary LOCO files are byte-identical to `87270e6`. Four of eight
survival files are byte-identical; the other four contain five changed printed
values among 9.6 million, with a maximum absolute difference of `1e-8`. The
32-trait quantitative comparisons are similarly tight: four changed values
among 38.4 million for the complete input and three among 38.4 million for the
missing-input fixture, again with maximum absolute difference at most `1e-8`.

Two additional candidate repetitions took 147.77 and 148.23 seconds for
binary traits, with Level 1 at 63.559 and 63.536 seconds. The survival
repetitions took 191.89 and 191.64 seconds, with Level 1 at 96.361 and 96.110
seconds. The final committed runs fall within the same narrow range. All eight
binary and all eight survival LOCO files are byte-for-byte identical between
the paired control and `87270e6`.

Binary energy is noisier than timing: the three resident-solve measurements
range from 7.14 to 8.17 Wh, so the 7.32-versus-7.64 Wh committed pair is not
evidence of a repeatable energy reduction. Survival is more consistent at
6.31-6.49 Wh versus 7.05 Wh for the paired control.

| A100 run | Whole-run GPU util. | Level 0 GPU util. | Level 1 GPU util. | Level 1 power | Peak device memory | GPU energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Binary | 52.6% | 36.3% | 67.2% | 248.0 W | 3,920 MiB | 7.86 Wh |
| Binary with resident solve | 50.0% | 36.8% | 64.7% | 286.2 W | 3,920 MiB | 7.32 Wh |
| Binary with generalized Level 0 | 68.1% | 80.9% | 66.4% | 257.3 W | 3,960 MiB | 6.33 Wh |
| Survival before residency | 18.5% | 38.6% | 17.4% | 62.9 W | 3,244 MiB | 22.82 Wh |
| Survival with residency | 29.4% | 34.1% | 27.3% | 130.5 W | 4,624 MiB | 7.02 Wh |
| Survival with resident solve | 32.0% | 33.4% | 31.3% | 157.7 W | 4,624 MiB | 6.31 Wh |
| Survival with generalized Level 0 | 39.0% | 76.3% | 30.5% | 166.9 W | 4,700 MiB | 6.25 Wh |

The resident solve improves throughput without making utilization the target.
In the final Cox run, Level 1 still spends 63.77 of 95.59 seconds in host
orchestration. Mean Level 1 utilization is 31.3%, with a 16.5% median and 76%
p90. Moving Cox risk-set and IRLS vector work to the device is now the clearest
Level 1 opportunity. Level 0 batching and prediction-output design reuse remain
the broader Step 1 opportunities described below.

## GPU saturation

GPU telemetry was sampled every 0.5 seconds on A100 and every second on T4.

| Run and phase | Mean GPU util. | Median | p10 / p90 | Samples >=90% | Mean power | Mean of limit | Peak device memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A100 700k original, whole run | 50.6% | 58% | 0% / 64% | 0.3% | 185 W | 46.3% | 6,432 MiB |
| A100 700k original, Level 0 | 58.8% | 58% | 56% / 64% | 0.0% | 212 W | 53.1% | 4,890 MiB |
| A100 700k original, Level 1/output | 15.9% | 0% | 0% / 39.8% | 0.0% | 71 W | 17.8% | 6,432 MiB |
| A100 700k optimized cold, whole run | 78.9% | 95% | 0% / 97% | 78.1% | 265 W | 66.3% | 19,000 MiB |
| A100 700k optimized cold, Level 0 | 93.5% | 95% | 95% / 97% | 94.2% | 308 W | 77.0% | 18,460 MiB |
| T4 700k, whole run | 97.0% | 100% | 98% / 100% | 95.0% | 65 W | 93.5% | 6,107 MiB |
| T4 700k, Level 0 | 98.7% | 100% | 99% / 100% | 96.8% | 66 W | 94.3% | 4,567 MiB |
| T4 700k, Level 1/output | 35.1% | 0% | 0% / 100% | 28.7% | 44 W | 62.5% | 6,107 MiB |

The T4 is saturated during Level 0. Its full run spends 3,768.12 of 4,567.98
Level 0 seconds (82.5%) in Gram construction. It physically reads 58.1 GB, but
205.82 seconds of decoder service overlap GPU work and foreground wait is only
0.68 seconds. The host averages 13.6% busy across eight vCPUs. This is a
device-compute limit, not a host-thread or input-pipeline limit.

The original A100 path is not saturated. Level 0 uses only about 59% of the GPU
and 12% of its 40 GB memory. Its utilization is also extremely steady (56% p10,
64% p90), which points to a repeated per-block work/transfer/orchestration
ceiling rather than occasional I/O stalls. Level 0 power nevertheless reaches
363 W at p90 and 392 W at its maximum, so individual kernels can drive the
device; the low mean reflects gaps and lower-intensity work between those
bursts, not a low power cap.

The optimized cold run closes that gap during Level 0: mean utilization rises
to 93.5% and Level 0 plus Level 1 falls from 171.75 to 90.85 seconds. Peak
Level 0 device memory rises from 4,890 to 18,460 MiB because the Level 1 design
is retained on the GPU. Level 1 itself is only 0.53 seconds; setup and output,
not the core fit, now account for most remaining low-utilization samples.

## A100 Level 0 profile

| Component | Time | Share of Level 0 |
| --- | ---: | ---: |
| Genotype residualization/scaling | 57.78 s | 37.8% |
| Gram construction | 38.49 s | 25.2% |
| Level 0 ridge | 32.26 s | 21.1% |
| Backend downloads | 9.99 s | 6.5% |
| Backend uploads | 6.79 s | 4.4% |
| Other | 5.29 s | 3.5% |
| Phenotype cross-product | 2.07 s | 1.4% |

The detailed timers show where additional A100 performance is available:

| Scope | Wall | Device compute | Transfer | Host/orchestration |
| --- | ---: | ---: | ---: | ---: |
| Genotype preprocessing | 57.73 s | 28.27 s | 25.40 s upload | 4.04 s |
| CV matrices | 52.93 s | 40.56 s | 7.19 s | 5.19 s |
| Level 0 ridge | 41.85 s | 6.80 s | 9.59 s | 25.47 s |

The profile points toward keeping more Level 0 state resident on the device
and submitting larger units of work. Ridge spends only 16% of its wall time in
device compute; 61% is host orchestration and another 23% is transfer.
Genotype preprocessing uploads 87.5 GB of packed hardcalls and spends almost
as long transferring them as computing on them.

### Overlapping the next Level 0 block

Commit `3b285ad` adds a two-block pipeline for smaller resident blocks. While
one CUDA backend finishes the current block's cross-validation and ridge work,
a second backend expands and preprocesses the already-prefetched packed
hardcalls for the next block. The control below used the same binary with the
pipeline disabled, which follows the `3e07ea3` execution path. Both runs used
the same warm 50,000-sample, 700,000-variant input and eight binary traits.

| Run | Wall | Level 0 | Level 1 | Level 0 GPU util. | Level 0 power | Level 0 device memory | GPU energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pipeline disabled | 146.71 s | 72.727 s | 63.746 s | 36.9% | 77.1 W | 1,220 MiB | 7.34 Wh |
| `3b285ad` | 138.41 s | 64.354 s | 64.016 s | 36.3% | 110.0 W | 2,010 MiB | 7.55 Wh |

This is an 11.5% Level 0 improvement and a 5.7% end-to-end improvement. Two
earlier candidate repetitions took 138.35 and 138.76 seconds, so the committed
result is representative rather than a lucky run. The pipeline serviced 710
next blocks in 6.48 seconds and left only 53 ms on the foreground path. All
eight LOCO files are byte-for-byte identical to the control.

Mean Level 0 utilization does not move at the 0.5-second sampling resolution,
but power rises by 43%. The higher power is consistent with useful concurrent
work that the coarse busy/not-busy utilization sample does not distinguish.
The speedup, rather than the utilization change, is the reason to retain this
code. Whole-run peak device memory is effectively unchanged at 3.92 GiB
because Level 1 already exceeds the pipeline's Level 0 allocation. Measured
energy rises by 0.21 Wh in this single pair, so this is a throughput win, not a
demonstrated energy win.

The same approach does not help the 500,000-sample shape. Forcing it on gives
188.67 seconds wall and 152.756 seconds in Level 0, within the earlier
186.43-188.14-second wall and 152.824-153.582-second Level 0 ranges. It spends
51.28 seconds waiting for the next preprocessed block and raises peak device
memory from 4,890 to 9,350 MiB. The committed default therefore enables the
pipeline only when one expanded resident block is at most 1 GB; the 50,000-
sample block is 0.4 GB, while the 500,000-sample block is 4 GB. The environment
variable `REGENIE_STEP1_LEVEL0_PIPELINE=0` or `1` can override that choice for
profiling.

For the large-sample workload, the next Level 0 target remains persistent
intermediates or more fused/batched work within one backend. Running a second
full preprocessing stream merely competes with the current block.

After Level 0, Level 1 fitting costs 19.34 s and prediction output costs 12.47
s. Grouped prediction uploads 14.22 GB in 3.19 s for only 14.8 ms of measured
device compute, while the prediction-output profile contains another 11.51 s
outside formatting and file writes. This is the second optimization target
after Level 0 residency/batching.

## Eight-core CPU profile

The current CPU backend completes the shared fixture in 103.0 minutes. Its
Level 0 time is distributed as follows:

| Component | Time | Share of Level 0 |
| --- | ---: | ---: |
| Genotype residualization/scaling | 3,724.28 s | 60.7% |
| Gram construction | 1,480.78 s | 24.1% |
| Eigensolve/transform | 402.58 s | 6.6% |
| PGEN decode | 382.55 s | 6.2% |
| Level 0 ridge | 104.81 s | 1.7% |
| Phenotype cross-product | 33.72 s | 0.5% |
| Other | 5.06 s | 0.1% |

Despite `--threads 8` on eight physical cores, the process averages 391% CPU
and the host averages 49.2% busy. I/O wait averages only 0.1%, so this is not a
storage stall. Residualization/scaling is the clearest CPU optimization target,
both because it dominates elapsed time and because the host has unused cores.
The run logically reads 63.7 GB from PGEN and physically reads 28.2 GB.

### Upstream v4.1.2 anchor

On the same N2, upstream v4.1.2 takes 8,207.13 seconds (136.8 minutes).
The current branch is 1.33x faster and saves 2,024.64 seconds, a 24.7% wall-time
reduction. Both binaries use oneMKL 2026.1 and the same fixture, seed, folds,
block size, covariates, and low-memory mode.

Upstream predates the structured Step 1 profile, but its per-block log timers
can be summed and compared with the same human-readable timers from the current
run. Those timers are rounded to the nearest millisecond.

| Level 0 component | Current | v4.1.2 | Current speedup |
| --- | ---: | ---: | ---: |
| Genotype read/decode | 381.62 s | 1,995.40 s | 5.23x |
| Residualization/scaling | 3,723.89 s | 3,574.58 s | 0.96x |
| Working matrices | 1,519.12 s | 1,505.69 s | 0.99x |
| Level 0 ridge | 507.00 s | 617.70 s | 1.22x |
| Timed Level 0 total | 6,131.62 s | 7,693.37 s | 1.25x |

The current PGEN path accounts for most of the gain, saving 1,613.78 seconds.
Ridge saves another 110.70 seconds. Residualization and working-matrix time are
essentially flat, so the headline improvement should not be attributed to
oneMKL. After subtracting the logged block, Level 1, and prediction timers,
v4.1.2 also has about 396 seconds of uninstrumented setup; this is an inferred
remainder, not a named upstream phase. Upstream physically reads 36.8 GB and
averages only 0.3% I/O wait, so storage stalls do not explain its slower
genotype path.

The two runs select identical ridge scores. Of 11.5 million printed LOCO
values, 99.99945% are text-identical; the remaining 63 differ by at most
`1e-6`, with RMSE `6.65e-10`. The performance gain therefore preserves the
numerical result to printed precision apart from last-digit floating-point
variation.

### Full host-CPU controls

The current CPU backend was also run to completion on the A100 and T4 hosts.
These are CPU measurements; the attached GPUs were idle.

| Host configuration | Physical cores | Threads | Wall | Level 0 | Average process CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eight-core N2 | 8 | 8 | 6,182.49 s | 6,133.79 s | 391% |
| A100 host | 6 | 12 | 9,103.16 s | 9,033.69 s | 325% |
| T4 host | 4 | 4 | 13,470.33 s | 13,280.60 s | 265% |

N2 is the fastest CPU host, as expected from its eight physical cores and
higher clock. The median A100 CUDA run is 48.6x faster than its host-CPU
control; T4 CUDA is 2.87x faster than its host-CPU control. The process uses
only 2.7 to 3.9 cores on average in all three full CPU runs, reinforcing the
parallelism opportunity visible in the N2 profile.

## Short matched slices

The same first 16,000 variants were used for short backend and host-thread
checks. These runs are useful for relative phase costs, not as whole-genome
runtime estimates.

| GPU | Host threads | Wall | Level 0 | Result |
| --- | ---: | ---: | ---: | --- |
| A100 | 6 physical | 8.47 s | 3.693 s | baseline |
| A100 | 12 including SMT | 8.27 s | 3.495 s | 2.4% wall reduction; selected for headline |

On A100, SMT provides a small benefit by accelerating the already-overlapped
PGEN decoder.

| System host | Backend | Host threads | Wall | Level 0 |
| --- | --- | ---: | ---: | ---: |
| A100 host | CUDA | 12 | 8.27 s | 3.495 s |
| A100 host | CPU | 12 | 208.57 s | 204.237 s |
| T4 host | CPU | 4 | 302.75 s | 297.591 s |

On the A100 host, CUDA is 25.2x faster end-to-end than the oneMKL CPU backend
on this slice. The production-width controls above show that the full-fixture
same-host speedup is larger.

## Step 2 scaling

Step 2 is CPU-only in this library. The same first 16,000 variants from the
shared fixture were processed on all three hosts with one quantitative
phenotype. This is long enough to expose setup, PGEN, and scoring costs without
presenting a short subset as a whole-genome elapsed-time estimate.

| Host | Threads | Wall | Setup | Genotype I/O | Variant compute | Wall speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A100 host | 1 | 48.67 s | 2.90 s | 3.73 s | 41.58 s | 1.00x |
| A100 host | 6 | 10.48 s | 2.83 s | 0.51 s | 6.70 s | 4.64x |
| A100 host | 12 | 8.73 s | 2.80 s | 0.42 s | 5.06 s | 5.58x |
| Eight-core N2 | 1 | 54.73 s | 2.29 s | 6.04 s | 46.12 s | 1.00x |
| Eight-core N2 | 2 | 26.04 s | 2.28 s | 1.25 s | 22.22 s | 2.10x |
| Eight-core N2 | 4 | 13.51 s | 2.22 s | 0.62 s | 10.40 s | 4.05x |
| Eight-core N2 | 8 | 7.58 s | 2.22 s | 0.32 s | 4.76 s | 7.22x |
| T4 host | 1 | 52.69 s | 3.21 s | 6.45 s | 42.49 s | 1.00x |
| T4 host | 4 | 14.96 s | 3.24 s | 0.80 s | 10.38 s | 3.52x |
| T4 host | 8 | 11.93 s | 3.20 s | 0.62 s | 7.57 s | 4.42x |

N2 reaches a 7.22x wall speedup on eight physical cores and is the fastest
host at 7.58 seconds. The scoring phase itself falls from 46.12 to 4.76
seconds. SMT still helps on the GPU hosts, but moving from physical cores to
all visible threads improves wall time by only 1.20x on A100 and 1.25x on T4.
The fixed 2.2–3.2-second setup cost is already 27–33% of the fastest runs;
it will matter less when more Step 2 variants are processed. Results are
bit-for-bit stable across thread counts within each system.

### Upstream v4.1.2 on N2

The N2 sweep was repeated with upstream v4.1.2 using the same oneMKL 2026.1
runtime, exact variant subset, and current prediction file. Holding the
prediction input fixed isolates Step 2 implementation performance.

| Threads | Current wall | v4.1.2 wall | Current speedup | Current peak RSS | v4.1.2 peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 54.73 s | 194.32 s | 3.55x | 0.33 GiB | 3.89 GiB |
| 2 | 26.04 s | 99.69 s | 3.83x | 0.33 GiB | 3.91 GiB |
| 4 | 13.51 s | 52.33 s | 3.87x | 0.33 GiB | 3.94 GiB |
| 8 | 7.58 s | 27.99 s | 3.69x | 0.33 GiB | 4.01 GiB |

The current branch is 3.55–3.87x faster across the sweep and uses about
one-twelfth of the memory. It also scales slightly better from one to eight
threads: 7.22x versus 6.94x upstream. All four upstream result files are
byte-identical to each other and to the current N2 output. Because v4.1.2 does
not emit structured Step 2 phase records, its checked-in rows retain wall time,
CPU use, and peak RSS but do not invent phase-level values.

## Reproduction and interpretation notes

- Every retained run exited successfully. The original current-branch snapshot
  uses `114ef81`; the retained GPU optimization checkpoints are `50ca3c1`,
  `87270e6`, `3b285ad`, and `dc64b54`. The upstream CPU anchor is v4.1.2 at
  `5f924b9bf54c1c7597174345def6eb2f1dee712c`.
- The original branch binaries passed all three CTest targets on each system.
  The `87270e6`, `3b285ad`, and `dc64b54` builds passed the CPU, CUDA-auto, and
  Cox tests on the A100.
  Dynamic linkage was checked before measurement: all x86 builds use oneMKL
  2026.1, and the GPU builds also link to the expected CUDA libraries. The
  full upstream run reproduced the current ridge scores and numerically
  equivalent LOCO predictions.
- The upstream Level 0 value is the sum of its legacy rounded per-block timers;
  v4.1.2 does not emit the structured phase records used by the current build.
- GPU telemetry is based on `nvidia-smi`, so sub-second kernel gaps are averaged
  into the sampling windows. The long headline traces are authoritative; the
  very short A100 thread-selection run has too few samples for phase power
  comparisons.
- The benchmark harness and result parser are in the parent directory. Each
  run retains the exact command, binary checksum, system inventory, raw
  console/profile records, resource measurements, and raw telemetry.
- Failed fixture-preparation attempts and interrupted preliminary runs are not
  included in any table.
