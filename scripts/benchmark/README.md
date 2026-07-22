# REGENIE benchmarks

The reports in this directory answer two practical questions:

| Report | Workload shapes | Main conclusion |
| --- | --- | --- |
| [Stage 1 production benchmark](results/2026-07-19-production.md) | Upstream v4.1.2 (`5f924b9`) direct P=1 comparison and conservative upstream floors for P=8/P=32 at N=500,000, M=700,000; separate engineering diagnostics | The retained A100 P=1 pipeline is 74.38x faster than upstream; `b5f86e9` multi-trait A100 workloads are at least 6.44-12.20x faster than the measured upstream floor |
| [Stage 2 benchmark](results/2026-07-20-step2.md) | Batched CPU revision `8953759` versus upstream v4.1.2 (`5f924b9`) at N=500,000, M=700,000, P=32; production projection at M=100,000,000 | Versus upstream `5f924b9`, `8953759` is 6.6-18.0x faster on the same N2; the recommended placement is one co-located Spot N2 worker per chromosome |

Each report states `N`, `M`, trait count, model, hardware, and whether a number
is measured or projected. The TSV files hold the detailed run records and
historical controls.

## Running a benchmark

`run_profiled.sh` wraps an ordinary REGENIE command. No separate benchmark
binary is required.

```bash
scripts/benchmark/run_profiled.sh \
  --label step1-n500k-m700k-p1 \
  --system-label a2-highgpu-1g \
  --output-root /path/to/results \
  --gpu-device 0 \
  --revision "$(git rev-parse HEAD)" \
  -- \
  /path/to/regenie \
    --step 1 \
    --pgen /path/to/real-ld-n500000-m700000 \
    --phenoFile /path/to/phenotypes.txt \
    --covarFile /path/to/covariates.txt \
    --qt --bsize 1000 --threads 12 --lowmem \
    --lowmem-prefix /path/to/results/l0 \
    --compute-backend cuda --gpu-device 0 --step1-profile \
    --out /path/to/results/fit
```

Use a stable `--system-label` that describes the machine type and relevant
configuration, never a transient cloud instance name.

For optimized x86 measurements, configure with `MKLROOT` and verify that
`binary_libraries.txt` links oneMKL. A BLAS mismatch can invalidate a CPU
comparison.

## What a run records

Each invocation creates `LABEL-YYYYMMDDTHHMMSSZ/` containing:

- exact command, Git revision, binary checksum, and linked libraries;
- CPU topology, memory, disks, and GPU configuration;
- wall time, CPU time, peak RSS, page faults, context switches, and filesystem
  operations from GNU `time`;
- structured `STEP1_PROFILE` or `STEP2_PROFILE` fields;
- host CPU, I/O, and run-queue telemetry when `vmstat`/`iostat` are available;
- GPU utilization, memory-controller activity, memory allocation, power,
  clocks, and temperature; and
- raw console output plus compact `summary.tsv` and `profile_kv.tsv` files.

The wrapper exits with REGENIE's status and can be used from CI or a batch
scheduler.

## Fixture helpers

`prepare_multitrait_fixture.py` creates a quantitative panel from aligned
phenotype and covariate files. By default it selects equal-sized group strata;
`--all-samples` retains the full source cohort and its original group
proportions. Its output contains 32 correlated traits, a complete file, a file
with per-trait missingness rising from 0% to 10%, and a PLINK keep file for
making a matching physical PGEN subset.

`prepare_nongaussian_fixture.py` derives eight binary traits with prevalences
from 1% to 50% and eight survival models with event fractions from 10% to 80%.

`prepare_step2_trait_matrix.py` expands those targets to the 32-trait Stage 2
panels and carries the quantitative missingness masks into the binary and
survival outcomes. Generation is deterministic so fixtures can be verified by
checksum across systems.

## Raw result files

Stage 1:

- [`results/2026-07-19-production.tsv`](results/2026-07-19-production.tsv) — Stage 1 measurements from revisions
  `114ef81`, `dc64b54`, `fa7506f`, and upstream v4.1.2 (`5f924b9`), with N,
  M, trait count, system, and cache state in every row.
- [`results/2026-07-22-step1-level1.tsv`](results/2026-07-22-step1-level1.tsv) — A100 `b5f86e9` versus `c312f41`
  multi-trait Level 1 comparisons at N=500,000 and M=700,000, plus explicitly
  labeled N=50,000 and CPU diagnostic comparisons.
- [`results/2026-07-22-step1-upstream.tsv`](results/2026-07-22-step1-upstream.tsv) — direct upstream v4.1.2
  comparisons, conservative N=500,000 multi-trait lower bounds, and explicit
  records of the N=50,000 workloads for which no defensible upstream inference
  is available.

Stage 2:

- [`results/2026-07-21-step2-upstream.tsv`](results/2026-07-21-step2-upstream.tsv) — upstream v4.1.2 (`5f924b9`)
  measurements at N=500,000, including the full M=700,000 quantitative anchor
  and chromosome-sized model/missingness runs.
- [`results/2026-07-21-step2-upstream-comparison.tsv`](results/2026-07-21-step2-upstream-comparison.tsv) — direct and
  validated upstream `5f924b9` versus batched CPU `8953759` comparison at
  N=500,000, M=700,000, and P=32.
- [`results/2026-07-21-step2-upstream-production.tsv`](results/2026-07-21-step2-upstream-production.tsv) — upstream
  `5f924b9`, batched CPU `8953759`, and CUDA `8953759` wall-time and cost
  projections at N=500,000, M=100,000,000, and P=32.
- [`results/2026-07-20-step2-cpu-block.tsv`](results/2026-07-20-step2-cpu-block.tsv) — oneMKL CPU revision `8953759`
  phase timings, correction-heavy runs, and validation.
- [`results/2026-07-20-step2-trait-matrix.tsv`](results/2026-07-20-step2-trait-matrix.tsv) — measured A100/N2 score-only
  matrix and GPU telemetry.
- [`results/2026-07-20-step2-large-anchor.tsv`](results/2026-07-20-step2-large-anchor.tsv) — cold-input revision
  `8953759` measurements at N=500,000, M=700,000, and P=32, plus A100
  block-size controls; the `--bsize 1000` rows feed the production model.
- [`results/2026-07-20-step2-production-model.tsv`](results/2026-07-20-step2-production-model.tsv) — N=500,000,
  M=100,000,000 wall-time and whole-placement cost estimates.
- [`results/2026-07-20-step2-chromosome-allocation.tsv`](results/2026-07-20-step2-chromosome-allocation.tsv) — 100 million
  variants allocated across chromosomes 1-22, X, and Y by GRCh38 length.
- [`results/2026-07-20-step2-localization.tsv`](results/2026-07-20-step2-localization.tsv) — durable same-region
  localization measurement and machine storage limits.
- [`results/2026-07-20-step2-sample-scaling.tsv`](results/2026-07-20-step2-sample-scaling.tsv) — matched N=50,000 and
  N=500,000 runs at M=16,000. These isolate sample-size effects and are not
  used for production placement estimates.
- [`results/2026-07-20-step2-prices.tsv`](results/2026-07-20-step2-prices.tsv) — cloud price snapshot and source URLs.
- [`results/2026-07-20-step2.tsv`](results/2026-07-20-step2.tsv) and
  [`results/2026-07-20-step2-integrated.tsv`](results/2026-07-20-step2-integrated.tsv) — historical CPU and integrated
  CUDA controls; use the per-row revision rather than treating them as a
  single baseline.
- [`results/2026-07-20-step2-steady-state.tsv`](results/2026-07-20-step2-steady-state.tsv) — historical utilization and
  concurrency diagnostics.
- [`results/2026-07-20-step2-cuda.tsv`](results/2026-07-20-step2-cuda.tsv) — standalone kernel diagnostics; these
  are not end-to-end runtime forecasts.
