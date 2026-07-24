# Stage 1 performance at biobank scale

Last refreshed: 2026-07-23.

This report compares the current byte-exact default with upstream REGENIE
v4.1.2. The primary workload has **500,000 samples**, **700,000 variants used
to fit the Stage 1 model**, and multiple outcomes fitted jointly. The traits
have 0-10% outcome-specific missingness unless the table says otherwise.

Level 1 path-Newton continuation is not part of the default. Its explicitly
enabled sensitivity is reported separately below.

`P` is the number of outcomes. “Stage 1 variants” is the number of markers
used for model fitting; it is not the number of variants later tested in
Stage 2.

## Results

The GPU placement uses one 40 GB A100 with six physical host cores. Upstream
v4.1.2 has no CUDA backend, so its placement uses an eight-physical-core N2.
Both builds use oneMKL 2026.1 for host linear algebra. Each measurement runs
one REGENIE process.

| Model | Missingness | N | Stage 1 variants | P | Current A100 wall / compute cost | Upstream v4.1.2 N2 wall / compute cost | Speedup | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Quantitative | none | 500,000 | 700,000 | 1 | **1.74 min / $0.11** | 136.79 min / $1.77 | **78.73x** | Both full runs measured directly; current Level 0 intermediates on SSD Persistent Disk |
| Quantitative | 0-10% | 500,000 | 700,000 | 8 | **4.92 min / $0.30** | 155.51 min / $2.01 | **31.64x** | Both full runs measured directly; current Level 0 intermediates on SSD Persistent Disk |
| Quantitative | 0-10% | 500,000 | 700,000 | 32 | **18.73 min / $1.15** | 243.08 min / $3.15 | **12.98x** | Both full runs measured directly; current Level 0 intermediates on SSD Persistent Disk |
| Binary | 0-10% | 500,000 | 700,000 | 8 | **9.85 min / $0.60** | 366.56-388.74 min / $4.75-$5.03 | **37.22-39.47x** | Current full run measured; upstream completed Level 0 and four of eight phenotype fits |
| Binary | 0-10% | 500,000 | 700,000 | 32 | **34.27 min / $2.10** | 1,065.89-1,161.49 min / $13.80-$15.04 | **31.10-33.89x** | Current full run measured; upstream range projected from a matched 210-block Level 0 prefix |
| Survival | 0-10% | 500,000 | 700,000 | 8 | **8.16 min / $0.50** | 230.19-239.78 min / $2.98-$3.10 | **28.22-29.40x** | Current full run measured; upstream completed Level 0 and four of eight phenotype fits |
| Survival | 0-10% | 500,000 | 700,000 | 32 | **30.63 min / $1.87** | 540.91-586.64 min / $7.00-$7.60 | **17.66-19.16x** | Current full run measured; upstream range projected from a matched 192-block Level 0 prefix |

Compute costs use the 2026-07-20 price snapshot: $3.673385/hour for the
on-demand A100 system and $0.776944/hour for the on-demand N2 system. They are
machine costs for the measured runtime and exclude persistent-disk charges.
At the benchmark size, a 2 TB balanced disk adds about $0.28/hour and a 2 TB
SSD disk about $0.48/hour. The SSD experiment attached the latter in addition
to the existing balanced disk; a production deployment can right-size and
delete scratch storage with the job.

The binary panel repeats prevalences of 1%, 2%, 5%, 10%, 20%, 30%, 40%, and
50% across its 32 traits. The survival panel similarly repeats event fractions
from 10% through 80%. Missingness increases by trait from 0% to 10%. The
quantitative traits are correlated. All rows use the same real-LD-derived
hard-call genotype fixture.

The upstream binary and survival totals are projections, not completed full
runs. The P=8 jobs completed all 711 Level 0 blocks and the first four
phenotype fits. The P=32 Level 0 projections use 210 completed binary blocks
(208,742 variants) and 192 completed survival blocks (191,046 variants).
Their Level 1 projections use the observed P=8 per-trait fit times because
upstream fits traits serially. Each range reflects both instrumented phase
rates and observed wall time.

## Where the current default time goes

| Model | N | Stage 1 variants | P | Setup | Level 0 | Level 1 | LOCO output | Full run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Quantitative, complete; SSD Level 0 | 500,000 | 700,000 | 1 | 2.04 s | 98.80 s | 0.66 s | 2.74 s | **104.24 s** |
| Quantitative, 0-10% missing; SSD Level 0 | 500,000 | 700,000 | 8 | 3.35 s | 157.75 s | 124.14 s | 9.68 s | **294.93 s** |
| Quantitative, 0-10% missing; SSD Level 0 | 500,000 | 700,000 | 32 | 5.84 s | 533.54 s | 543.86 s | 40.78 s | **1,124.01 s** |
| Binary, 0-10% missing; SSD Level 0 | 500,000 | 700,000 | 8 | 5.77 s | 277.05 s | 292.20 s | 15.92 s | **590.94 s** |
| Binary, 0-10% missing; SSD Level 0 | 500,000 | 700,000 | 32 | 15.21 s | 843.41 s | 1,143.68 s | 53.85 s | **2,056.15 s** |
| Survival, 0-10% missing; SSD Level 0 | 500,000 | 700,000 | 8 | 6.73 s | 233.53 s | 233.43 s | 15.70 s | **489.38 s** |
| Survival, 0-10% missing; SSD Level 0 | 500,000 | 700,000 | 32 | 21.71 s | 854.42 s | 906.57 s | 54.83 s | **1,837.54 s** |

All current multi-trait rows put only the Level 0 intermediates on a 2 TB,
network-attached SSD Persistent Disk. Input genotypes and final output remain
on the original disk. The intermediates occupy 113.76 GB at P=8 and 455.04 GB
at P=32.

Quantitative Level 1 remains storage-bound: the P=32 run spent 423.15 of
543.86 seconds waiting for foreground reads. Binary and survival Level 1 are
compute-bound; their P=32 foreground read waits were only 15.26 seconds within
1,143.68 and 906.57 seconds, respectively.

The current byte-exact default uses ordinary FP64 IRLS for binary Level 1.
The survival event fields are not Stage 1 outcomes; Level 0 downloads only
the active time-outcome columns.

## Opt-in path-Newton sensitivity

Path-Newton is disabled by default because it changes low-order printed digits
in final Stage 2 results. It can be enabled explicitly with:

```bash
REGENIE_STEP1_LEVEL1_PATH_NEWTON=1 regenie ...
```

The following matched binary runs include the same unconditional Level 0
transport path as the default. P=8 uses seed 20260721 in both rows and P=32
uses seed 20260722 in both rows.

| P | Byte-exact default | Default + path-Newton | Incremental speedup | Runtime reduction |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 589.753 s | **453.624 s** | **1.30x** | **23.1%** |
| 32 | 2,056.154 s | **1,526.362 s** | **1.35x** | **25.8%** |

Path-Newton preserves the model specification, ridge grids, convergence
tolerance, and scientific conclusions tested, but it is not byte-identical.
The direct P=8 and P=32 Stage 2 comparisons covered 5.6 million and 22.4
million association rows. Maximum absolute differences in printed fields
were:

| Field | Maximum absolute difference |
| --- | ---: |
| `BETA` | 0.000001 |
| `SE` | 0.0000001 |
| `CHISQ` | 0.0001 |
| `LOG10P` | 0.00001 |

All numeric-field correlations exceeded 0.9999999999998. Every trait retained
identical top-100 and top-1,000 variants, identical membership at `p <= 1e-5`
and genome-wide significance, and no sign changes among variants with
`abs(z) >= 3`. These rows are a sensitivity analysis, not the default
headline configuration.

## Same-machine CPU comparison

The A100 table is a placement comparison because upstream is CPU-only. The
following row isolates the software improvement by running both versions on
the same eight-core N2 with oneMKL:

| Model | Missingness | N | Stage 1 variants | P | Current branch | Upstream v4.1.2 | Software speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Quantitative | 0-10% | 500,000 | 700,000 | 8 | **106.73 min measured** | 155.51 min measured | **1.46x** |

The current N2 run spends 6,153.02 seconds in Level 0, 238.72 seconds in
Level 1, and 9.17 seconds in output. CPU-only Stage 1 remains dominated by
Level 0, which is why the same-machine software gain is much smaller than the
A100 placement gain.

## Correctness and numerical libraries

The current implementation keeps the model and ridge grids unchanged and
uses FP64 for the accelerated linear algebra. It does not use a float32 model
path. Dense host operations use oneMKL; dense GPU operations use
cuBLAS/cuSOLVER.

The A100 build passes the CPU, CUDA, Stage 2 CPU, and Cox test targets. The N2
build passes the CPU, automatic-backend, and Cox targets.

The default path keeps the original model, ridge grids, initialization,
iteration order, convergence checks, and line-search semantics. Every
production-size Stage 1 LOCO file in the current P=8/P=32 quantitative,
binary, and survival runs is byte-for-byte identical to its matched validated
control. Those files are the inputs to Stage 2; with unchanged Stage 2 code,
the final Stage 2 scientific output is therefore also byte-identical.

The path-Newton sensitivity is the sole exception discussed in this report.
It requires an explicit opt-in and its final Stage 2 differences are
quantified above.

## Raw results

The focused measurements are in
[`2026-07-22-step1-level1.tsv`](2026-07-22-step1-level1.tsv). Direct and
projected upstream comparisons are in
[`2026-07-22-step1-upstream.tsv`](2026-07-22-step1-upstream.tsv). The larger
[`2026-07-19-production.tsv`](2026-07-19-production.tsv) is a historical raw
run ledger retained for reproducibility; it is not used as a current-branch
comparison table. Transport-safety A/B measurements are kept out of this
headline report and are recorded in
[`2026-07-23-step1-pinned-download.md`](2026-07-23-step1-pinned-download.md).
The full opt-in sensitivity is in
[`2026-07-23-step1-path-newton.md`](2026-07-23-step1-path-newton.md).
