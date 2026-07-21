# REGENIE benchmarks

The reports in this directory answer two practical questions:

| Report | Workload shapes | Main conclusion |
| --- | --- | --- |
| [Stage 1 production benchmark](results/2026-07-19-production.md) | N=500,000, M=700,000, one quantitative trait; and N=50,000, M=700,000 multi-trait panels | A100 is the clear Stage 1 target; the retained large-sample Level 0 path is close to saturated |
| [Stage 2 benchmark](results/2026-07-20-step2.md) | Measured N=50,000 versus N=500,000 sample scaling; M=10,000,000 CPU fleet estimates; measured A100 comparison at N=50,000 | Use Spot N2 workers for production Stage 2; worker count is a wall-time choice, not a chromosome-count rule |

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

- [`results/2026-07-19-production.tsv`](results/2026-07-19-production.tsv) — retained run-level measurements for
  the Stage 1 report.

Stage 2:

- [`results/2026-07-20-step2-cpu-block.tsv`](results/2026-07-20-step2-cpu-block.tsv) — current oneMKL CPU measurements,
  phase timings, correction-heavy runs, and validation.
- [`results/2026-07-20-step2-trait-matrix.tsv`](results/2026-07-20-step2-trait-matrix.tsv) — measured A100/N2 score-only
  matrix and GPU telemetry.
- [`results/2026-07-20-step2-sample-scaling.tsv`](results/2026-07-20-step2-sample-scaling.tsv) — matched N=50,000 and
  N=500,000 runs for 32 quantitative, binary, and survival traits.
- [`results/2026-07-20-step2-cost-projection.tsv`](results/2026-07-20-step2-cost-projection.tsv) — M=10,000,000
  worker-count scenarios with aggregate fleet cost stated explicitly.
- [`results/2026-07-20-step2-prices.tsv`](results/2026-07-20-step2-prices.tsv) — cloud price snapshot and source URLs.
- [`results/2026-07-20-step2.tsv`](results/2026-07-20-step2.tsv) and
  [`results/2026-07-20-step2-integrated.tsv`](results/2026-07-20-step2-integrated.tsv) — earlier CPU and integrated CUDA
  controls.
- [`results/2026-07-20-step2-steady-state.tsv`](results/2026-07-20-step2-steady-state.tsv) — historical utilization and
  concurrency diagnostics.
- [`results/2026-07-20-step2-cuda.tsv`](results/2026-07-20-step2-cuda.tsv) — standalone kernel diagnostics; these
  are not end-to-end runtime forecasts.
