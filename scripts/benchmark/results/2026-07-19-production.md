# Stage 1 performance at biobank scale

This report compares the best validated version of this branch with upstream
REGENIE v4.1.2. The primary workload has **500,000 samples**, **700,000
variants used to fit the Stage 1 model**, and multiple outcomes fitted jointly.
The traits have 0-10% outcome-specific missingness unless the table says
otherwise.

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
| Quantitative | 0-10% | 500,000 | 700,000 | 8 | **4.92 min / $0.30** | 155.51 min / $2.01 | **31.62x** | Both full runs measured directly; current Level 0 intermediates on SSD Persistent Disk |
| Quantitative | 0-10% | 500,000 | 700,000 | 32 | **18.71 min / $1.15** | 243.08 min / $3.15 | **12.99x** | Both full runs measured directly; current Level 0 intermediates on SSD Persistent Disk |
| Binary | 0-10% | 500,000 | 700,000 | 8 | **10.41 min / $0.64** | 366.56-388.74 min / $4.75-$5.03 | **35.23-37.36x** | Current full run measured; upstream completed Level 0 and four of eight phenotype fits |
| Binary | 0-10% | 500,000 | 700,000 | 32 | **36.33 min / $2.22** | 1,065.89-1,161.49 min / $13.80-$15.04 | **29.34-31.97x** | Current full run measured; upstream range projected from a matched 210-block Level 0 prefix |
| Survival | 0-10% | 500,000 | 700,000 | 8 | **9.60 min / $0.59** | 230.19-239.78 min / $2.98-$3.10 | **23.99-24.99x** | Current full run measured; upstream completed Level 0 and four of eight phenotype fits |
| Survival | 0-10% | 500,000 | 700,000 | 32 | **33.15 min / $2.03** | 540.91-586.64 min / $7.00-$7.60 | **16.32-17.70x** | Current full run measured; upstream range projected from a matched 192-block Level 0 prefix |

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

## Where the A100 time goes

| Model | N | Stage 1 variants | P | Setup | Level 0 | Level 1 | LOCO output | Full run | Mean GPU utilization, Level 0 / Level 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Quantitative, complete; SSD Level 0 | 500,000 | 700,000 | 1 | 2.04 s | 98.80 s | 0.66 s | 2.74 s | **104.24 s** | 84.9% / not meaningful (0.66 s) |
| Quantitative, 0-10% missing; SSD Level 0 | 500,000 | 700,000 | 8 | 3.42 s | 158.95 s | 122.98 s | 9.74 s | **295.09 s** | 62.7% / 10.2% |
| Quantitative, 0-10% missing; SSD Level 0 | 500,000 | 700,000 | 32 | 5.93 s | 533.34 s | 541.66 s | 41.50 s | **1,122.43 s** | 25.2% / 9.6% |
| Binary, 0-10% missing; SSD Level 0 | 500,000 | 700,000 | 8 | 5.61 s | 309.13 s | 294.04 s | 15.52 s | **624.32 s** | 31.6% / 82.6% |
| Binary, 0-10% missing; SSD Level 0 | 500,000 | 700,000 | 32 | 15.25 s | 966.00 s | 1,144.88 s | 53.80 s | **2,179.92 s** | 14.2% / 85.3% |
| Survival, 0-10% missing; SSD Level 0 | 500,000 | 700,000 | 8 | 6.75 s | 319.93 s | 233.26 s | 15.77 s | **575.71 s** | 33.7% / 67.6% |
| Survival, 0-10% missing; SSD Level 0 | 500,000 | 700,000 | 32 | 21.38 s | 1,008.61 s | 903.57 s | 55.50 s | **1,989.06 s** | 13.9% / 72.8% |

Low utilization does not have one explanation across the run. Level 0 mixes
GPU solves with genotype preprocessing, host orchestration, and writing the
Level 0 predictions. Level 1 quantitative is storage-bound. The P=32 design
occupies 455.04 GB on disk; with its intermediates on balanced Persistent
Disk, 498.61 of 618.73 Level 1 seconds were foreground read wait.

A matched run put only those intermediates on a 2 TB SSD Persistent Disk.
Input genotypes and final output stayed on the original balanced disk. Full
runtime fell from 1,289.32 to 1,122.43 seconds, a 12.9% wall-time reduction.
Level 0 fell from 608.48 to 533.34 seconds and Level 1 from 618.73 to 541.66
seconds. Level 1 still spent 421.50 seconds waiting for reads, and mean GPU
utilization there was 9.6%. The faster disk therefore helps, but does not
remove the cost of materializing and rereading the 455 GB design. This test
uses network-attached Persistent Disk SSD, not Local SSD.

At P=8, where the intermediates occupy 113.76 GB, the same change reduced the
full run from 316.86 to 295.09 seconds (6.9%). Level 0 was effectively
unchanged; Level 1 fell from 139.72 to 122.98 seconds. All 32 P=32 and all
eight P=8 LOCO files are byte-identical to the matched balanced-disk runs.

Binary and survival Level 1 contain much more GPU work. Mean Level 1
utilization is 82.6% and 67.6% at P=8, and 85.3% and 72.8% at P=32. Moving
their Level 0 intermediates from balanced to SSD Persistent Disk changed the
P=32 full runtimes by only 0.5% for binary and 0.6% for survival. In the SSD
runs, Level 1 foreground read wait was 15.4 seconds of 1,144.9 seconds for
binary and 15.0 seconds of 903.6 seconds for survival. These Level 1 workloads
are compute-bound rather than storage-bound.

For survival, the event fields are not Stage 1 outcomes. The current
implementation performs the full FP64 cuBLAS/cuSOLVER calculation but
downloads only the 32 active time-outcome columns. GPU-to-host result transfer
accounts for 398.37 seconds of the 1,008.61-second Level 0 phase in the
production-size run.

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
build passes the CPU, automatic-backend, and Cox targets. Production-size
outputs for all seven current A100 rows are byte-identical to matched
validated controls.

## Raw results

The focused measurements are in
[`2026-07-22-step1-level1.tsv`](2026-07-22-step1-level1.tsv). Direct and
projected upstream comparisons are in
[`2026-07-22-step1-upstream.tsv`](2026-07-22-step1-upstream.tsv). The larger
[`2026-07-19-production.tsv`](2026-07-19-production.tsv) is a historical raw
run ledger retained for reproducibility; it is not used as a current-branch
comparison table.
