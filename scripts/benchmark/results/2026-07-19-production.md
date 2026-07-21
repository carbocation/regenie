# Stage 1 production benchmark

This report describes the retained Stage 1 performance work as of 2026-07-19.
It covers two explicit workload sizes:

- **Production-width single trait:** N=500,000 samples, M=700,000 variants,
  one quantitative trait.
- **Multi-trait model checks:** N=50,000 samples, M=700,000 variants, with 32
  quantitative traits or eight binary/survival models.

All headline genotypes come from the same real-LD-derived hard-call PGEN. Every
x86 build uses oneMKL 2026.1. Stage 2 has since changed substantially and is
documented separately in [`2026-07-20-step2.md`](2026-07-20-step2.md).

## Bottom line

- The cold-cache A100 completes the N=500,000, M=700,000 single-quantitative-
  trait run in 110.34 seconds. Level 0 averages 93.5% GPU utilization and 308 W.
- On the same workload, the retained CPU branch takes 6,182 seconds on eight
  physical N2 cores; upstream v4.1.2 takes 8,207 seconds. The T4 takes 4,692
  seconds and is not a useful Stage 1 target relative to A100.
- At N=50,000 and M=700,000, the A100 completes 32 quantitative traits in about
  113 seconds, eight binary traits in 95 seconds, and eight survival models in
  134 seconds.
- The remaining A100 work differs by model: output dominates more of the
  quantitative run, logistic Level 1 remains substantial, and Cox Level 1 is
  still limited by host orchestration.

## Systems and fixture

| System label | Accelerator | Physical CPU exposed to benchmark | RAM | Build |
| --- | --- | ---: | ---: | --- |
| `a2-highgpu-1g` | A100-SXM4 40 GB, 400 W | 6 Xeon cores / 12 vCPU | 83 GiB | CUDA sm80, oneMKL 2026.1 |
| `n1-standard-8-t4` | T4 15 GB, 70 W | 4 Xeon cores / 8 vCPU | 29 GiB | CUDA sm75, oneMKL 2026.1 |
| `n2-standard-16-smt-off` | none | 8 Xeon cores / 8 benchmark threads | 62 GiB | native x86-64, oneMKL 2026.1 |

The N=500,000 fixture was made by resampling 1000 Genomes donors within
super-population groups at 32-variant LD-block boundaries. It retains local LD,
allele-frequency variation, genotype missingness, and population structure,
but the expanded samples are synthetic. The phenotype is quantitative with
target heritability 0.3; `SUPERPOP` is a categorical covariate.

The N=50,000 fixture is a deterministic equal-stratum physical subset: 10,000
samples from each of five population groups and the same 700,000 variants.
Genotype, phenotype, and covariate checksums matched across benchmark systems.

Unless a table says otherwise, Stage 1 uses five-fold cross-validation,
`--bsize 1000`, `--lowmem`, the default Level 0 and Level 1 ridge grids, and
local genotype files.

## Headline Stage 1 results

Every row states its sample count, variant count, and number of fitted models.
Rows with different revisions are retained reference points, not claims that
every system ran the final A100 implementation.

| Model | N | M | Traits/models | System | Backend | Revision | Wall | Level 0 | Level 1 | Output/cache note |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| Quantitative | 500,000 | 700,000 | 1 | A100 | CUDA | `dc64b54` | 110.34 s | 90.312 s | 0.533 s | Cold page cache |
| Quantitative | 500,000 | 700,000 | 1 | T4 | CUDA | `114ef81` | 4,692.19 s | 4,567.98 s | not separated | Storage-backed reference |
| Quantitative | 500,000 | 700,000 | 1 | 8-core N2 | CPU | `114ef81` | 6,182.49 s | 6,133.79 s | not separated | Mixed-cache CPU reference |
| Quantitative | 500,000 | 700,000 | 1 | 8-core N2 | CPU | upstream v4.1.2 | 8,207.13 s | 7,693.37 s | not separated | Upstream anchor |
| Quantitative | 50,000 | 700,000 | 32 | A100 | CUDA | `fa7506f` | 113.80 s | 54.457 s | 24.387 s | 30.983 s output; cold cache |
| Quantitative, 0-10% input missingness | 50,000 | 700,000 | 32 | A100 | CUDA | `fa7506f` | 113.30 s | 54.346 s | 24.321 s | 30.943 s output; cold cache |
| Quantitative | 50,000 | 700,000 | 32 | 8-core N2 | CPU | `114ef81` | 1,423.27 s | 986.913 s | 410.919 s | 24.080 s output |
| Binary | 50,000 | 700,000 | 8 | A100 | CUDA | `fa7506f` | 95.34 s | 21.602 s | 63.678 s | 7.843 s output |
| Binary | 50,000 | 700,000 | 8 | 8-core N2 | CPU | `114ef81` | 3,308.72 s | 937.320 s | 2,365.552 s | 4.553 s output |
| Survival | 50,000 | 700,000 | 8 | A100 | CUDA | `fa7506f` | 133.90 s | 25.802 s | 97.869 s | 7.849 s output |
| Survival | 50,000 | 700,000 | 8 | 8-core N2 | CPU | `114ef81` | 2,723.69 s | 948.247 s | 1,769.544 s | 4.523 s output |

The N=500,000 A100 and N2 rows are not a clean cost comparison because they
use different execution backends and the A100 is the intended Stage 1 target.
They establish the size of the acceleration and the upstream CPU anchor. The
N=50,000 multi-model rows show that the GPU gains extend beyond a single
quantitative trait.

## N=500,000, M=700,000 quantitative result

The retained production-width change keeps Level 0 state and the assembled
Level 1 design on the GPU, normalizes predictions on the device, overlaps
low-memory output with the next block, reuses static inputs, and uses registered
packed-PGEN buffers.

| Implementation | N | M | Traits | Cache | Wall | REGENIE total | Level 0 | Level 1 | Level 0 GPU util. | Level 0 power |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Earlier retained path, `3b285ad` | 500,000 | 700,000 | 1 | cold | 188.29 s | 185.507 s | 153.885 s | 17.864 s | 58.1% | 214.8 W |
| Current path, `dc64b54` | 500,000 | 700,000 | 1 | cold | 110.34 s | 107.709 s | 90.312 s | 0.533 s | 93.5% | 308.0 W |

Level 0 plus Level 1 is 47.1% faster; internal total time falls by 41.9%.
During current Level 0, median utilization is 95%, p10/p90 are 95%/97%, and
94.2% of telemetry samples are at or above 90%. Peak device memory is about
18.5 GiB. The improvement therefore raises throughput and closes the original
utilization gap without exhausting the 40 GB device.

Static PGEN worker partitions also matter for cold input. Relative to the
immediately preceding candidate, aggregate PGEN service time falls from 102.10
to 64.87 seconds and foreground wait from 14.34 to 0.81 seconds. Storage is no
longer the dominant critical-path cost in this run.

The 110.34-second figure is one cold run. Per-block timings and the utilization
trace support the mechanism, but another cold repetition is needed before
treating the last few seconds as a stable estimate.

## N=50,000, M=700,000 multi-model result

The 32 quantitative traits are correlated with trait 1 at approximately 0.4.
The missing input increases linearly from 0% to 10% by trait. Stage 1 uses its
documented per-trait mean imputation, so the fitted outcome matrix remains
dense. The binary prevalences are 1%, 2%, 5%, 10%, 20%, 30%, 40%, and 50%; the
survival event fractions range from 10% through 80%.

| Model | N | M | Traits/models | Earlier A100 wall | Current A100 wall | Current Level 0 | Current Level 1 | Current output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Quantitative, complete | 50,000 | 700,000 | 32 | 227.51 s | 113.80 s | 54.457 s | 24.387 s | 30.983 s |
| Quantitative, 0-10% input missingness | 50,000 | 700,000 | 32 | 227.44 s | 113.30 s | 54.346 s | 24.321 s | 30.943 s |
| Binary | 50,000 | 700,000 | 8 | 170.09 s | 95.34 s | 21.602 s | 63.678 s | 7.843 s |
| Survival | 50,000 | 700,000 | 8 | 232.15 s | 133.90 s | 25.802 s | 97.869 s | 7.849 s |

The complete and missing quantitative runs differ by less than 0.5%, so the
default mean-imputation path has no material performance penalty here. The
current multi-outcome Level 0 path solves outcomes together, normalizes and
reorders predictions on the GPU, and writes phenotype-major blocks
asynchronously.

For binary and survival models, the weighted ridge solve also remains on the
GPU instead of copying its products to the host and immediately copying them
back. Relative to the preceding resident-design implementation, this reduced
Level 1 by 27.7% for binary traits and 30.4% for survival models. The later
generalized Level 0 path produced the current totals above.

## GPU utilization and energy

The following rows are measured whole-run or phase summaries. They are not
comparable unless `N`, `M`, and model count match.

| Model/run | N | M | Traits/models | Phase | Mean GPU util. | Mean power | Peak device memory | GPU energy |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Quantitative, current production | 500,000 | 700,000 | 1 | Level 0 | 93.5% | 308.0 W | 18,460 MiB | not isolated |
| Quantitative, earlier production | 500,000 | 700,000 | 1 | Level 0 | 58.1% | 214.8 W | 4,890 MiB | not isolated |
| Quantitative, complete | 50,000 | 700,000 | 32 | Level 0 | 44.3% | 133.3 W | 5,020 MiB | 3.28 Wh whole run |
| Binary, current | 50,000 | 700,000 | 8 | Level 0 | 80.9% | 197.0 W | 3,960 MiB | 6.33 Wh whole run |
| Binary, current | 50,000 | 700,000 | 8 | Level 1 | 66.4% | 257.3 W | 3,960 MiB | included above |
| Survival, current | 50,000 | 700,000 | 8 | Level 0 | 76.3% | 179.6 W | 4,700 MiB | 6.25 Wh whole run |
| Survival, current | 50,000 | 700,000 | 8 | Level 1 | 30.5% | 166.9 W | 4,700 MiB | included above |
| Quantitative, T4 reference | 500,000 | 700,000 | 1 | Level 0 | 98.7% | 66 W | 4,567 MiB | 85.25 Wh whole run |

The T4 is saturated but slow: 82.5% of its Level 0 time is Gram construction.
Its host and input pipeline are not the bottleneck, so more scheduling work is
unlikely to make it competitive with A100. Conversely, the production A100
Level 0 is now close to saturated and should be judged by elapsed time rather
than attempts to raise utilization further.

The lower utilization of the N=50,000 quantitative run is not evidence that it
failed to improve. Wall time halves and whole-run energy falls by about 41%
relative to the earlier path. Output now occupies a much larger share of the
shorter run.

## CPU and upstream anchors

The CPU comparison below uses one quantitative trait with **N=500,000** and
**M=700,000** on the same eight-core N2 and oneMKL runtime.

| Implementation | N | M | Traits | Wall | Logged Level 0 | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current branch, `114ef81` | 500,000 | 700,000 | 1 | 6,182.49 s | 6,133.79 s | 14.98 GiB |
| Upstream v4.1.2, `5f924b9` | 500,000 | 700,000 | 1 | 8,207.13 s | 7,693.37 s | 14.75 GiB |

The current branch is 1.33x faster. Most of the gain is PGEN handling:
genotype read/decode falls from 1,995.40 to 381.62 seconds. Residualization and
working-matrix time are essentially unchanged, so this result should not be
described as an MKL gain. The current CPU process averages only 3.91 busy cores
despite eight benchmark threads; residualization/scaling accounts for 60.7% of
Level 0 and remains the clearest CPU Stage 1 opportunity.

The two runs select identical ridge scores. Of 11.5 million printed LOCO
values, 99.99945% are text-identical; the remaining 63 differ by at most
`1e-6`, with RMSE `6.65e-10`.

## Remaining opportunities

1. **Cox Level 1:** the current survival run spends about 64 of 96 Level 1
   seconds in host orchestration. Risk-set and IRLS vector work are the largest
   remaining A100 opportunity for survival models.
2. **Prediction output:** the 32-trait quantitative run spends about 31 seconds
   writing predictions. Output-side design reuse and formatting are now more
   important because the GPU fit is much shorter.
3. **Large-sample Level 0:** the N=500,000 run is already 93.5% utilized.
   Changes should target fewer transfers or more persistent intermediates and
   must improve elapsed time; a second full preprocessing stream was tested and
   did not help this shape.
4. **CPU residualization:** the eight-core N2 uses only about half of its
   available cores during the N=500,000 run, while residualization/scaling
   dominates Level 0.

## Numerical validation

- The N=500,000 small-fixture single-trait output is byte-identical to its
  control. At full scale, two of 11,500,023 printed values differ; maximum
  absolute difference is `1e-9`.
- All eight binary LOCO files are byte-identical across the retained changes.
- Four of eight survival files are byte-identical; the other four contain five
  changed printed values among 9.6 million, each at most `1e-8`.
- The 32-trait complete and missing quantitative comparisons contain four and
  three changed printed values, respectively, among 38.4 million; maximum
  absolute difference is `1e-8`.
- Retained A100 builds passed the CPU, CUDA-auto, and Cox test targets available
  at the time of measurement.

## Data and reproduction

The retained run-level measurements are in
[`2026-07-19-production.tsv`](2026-07-19-production.tsv). That file contains
the exact workload shape, revision, backend, cache state, phase timings,
resource use, GPU telemetry summaries, and validation status for each row.

`run_profiled.sh` records the exact command, binary checksum, linked libraries,
machine configuration, raw console/profile output, GNU `time`, host telemetry,
and GPU telemetry. `prepare_multitrait_fixture.py` and
`prepare_nongaussian_fixture.py` reproduce the N=50,000 phenotype panels.

Historical implementation checkpoints retained in the raw data include the
Cox resident-design path (`50ca3c1`), resident weighted solve (`87270e6`),
small-block Level 0 pipeline (`3b285ad`), production-width pipeline
(`dc64b54`), and generalized multi-outcome path (`fa7506f`). They are listed
for reproducibility, not as a reading order for this report.
