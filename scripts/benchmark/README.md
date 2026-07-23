# REGENIE performance benchmarks

These reports compare the best tested version of this branch with upstream
REGENIE v4.1.2. Start with the report for the stage you plan to run:

| Report | Workload shapes | Main conclusion |
| --- | --- | --- |
| [Stage 1 production benchmark](results/2026-07-19-production.md) | N=500,000; 700,000 model-fitting variants; quantitative, binary, and survival traits with 0-10% missingness | Direct current A100 measurements versus upstream v4.1.2 on an eight-core N2, plus a same-N2 software comparison |
| [Stage 2 benchmark](results/2026-07-20-step2.md) | Current CPU revision `3ab5fbb` versus upstream v4.1.2 at N=50,000/N=500,000 and P=8/P=32; quantitative dispatch checks from N=5,000 to N=500,000; best measured CUDA placement evidence; production projection for 100 million Stage 2 variants tested | At N=500,000 and P=32 with 0-10% missingness, current processes 848.2 quantitative, 792.7 binary, and 524.9 survival variants/s; CPU chromosome fan-out remains the recommended production placement |

Every headline table states the sample count, trait count, model, hardware,
and whether a value was measured or projected. “Stage 1 variants” means the
markers used to fit the model. “Stage 2 variants tested” means the variants
read and tested during association analysis; the reports do not call both of
these quantities `M`.

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

- [`results/2026-07-19-production.tsv`](results/2026-07-19-production.tsv) — upstream and best branch Stage 1
  placement measurements, with sample count, Stage 1 variants, trait count,
  system, revision, and cache state in every row.
- [`results/2026-07-22-step1-level1.tsv`](results/2026-07-22-step1-level1.tsv) — focused
  latest or best-applicable branch measurements at N=500,000 and 700,000
  Stage 1 variants. Intermediate branch revisions are not included.
- [`results/2026-07-22-step1-upstream.tsv`](results/2026-07-22-step1-upstream.tsv) — direct upstream v4.1.2
  comparisons and clearly labeled projections for slow multi-trait runs.
- [`results/2026-07-23-step1-path-newton.md`](results/2026-07-23-step1-path-newton.md) and
  [`TSV`](results/2026-07-23-step1-path-newton.tsv) — default-off Level 1
  ridge-path continuation experiment at N=500,000, 700,000 model-fitting
  variants, and P=8/P=32, including full downstream Stage 2 validation.

Stage 2:

- [`results/2026-07-22-step2-optimization.tsv`](results/2026-07-22-step2-optimization.tsv) — current CPU and upstream
  measurements at N=500,000 and P=32, including the 57,821-variant
  quantitative, binary, and survival panels. It contains only upstream and
  the latest branch revision.
- [`results/2026-07-22-step2-p8.tsv`](results/2026-07-22-step2-p8.tsv) — direct current CPU and upstream
  measurements at N=500,000, P=8, and 57,821 Stage 2 variants tested for
  quantitative, binary, and survival traits with variable missingness.
- [`results/2026-07-22-step2-sample-scaling-upstream.tsv`](results/2026-07-22-step2-sample-scaling-upstream.tsv) — direct
  current CPU and upstream measurements at N=50,000 and N=500,000, P=32,
  and 57,821 Stage 2 variants tested, including complete and variable-missingness
  quantitative, binary, and survival panels.
- [`results/2026-07-22-step2-binary-missingness.tsv`](results/2026-07-22-step2-binary-missingness.tsv) — matched cold-input
  upstream and latest-branch measurements for complete and
  variable-missingness binary traits at N=500,000, P=32, and 57,821 Stage 2
  variants tested.
- [`results/2026-07-22-step2-production-current.tsv`](results/2026-07-22-step2-production-current.tsv) — current CPU,
  upstream, and best measured CUDA production projection for 100 million
  Stage 2 variants tested.
- [`results/2026-07-21-step2-upstream.tsv`](results/2026-07-21-step2-upstream.tsv) — upstream v4.1.2 measurements at
  N=500,000, including the full 700,000-tested-variant quantitative anchor and
  chromosome-sized model/missingness runs.
- [`results/2026-07-20-step2-chromosome-allocation.tsv`](results/2026-07-20-step2-chromosome-allocation.tsv) — 100 million
  variants allocated across chromosomes 1-22, X, and Y by GRCh38 length.
- [`results/2026-07-20-step2-localization.tsv`](results/2026-07-20-step2-localization.tsv) — durable same-region
  localization measurement and machine storage limits.
- [`results/2026-07-20-step2-prices.tsv`](results/2026-07-20-step2-prices.tsv) — cloud price snapshot and source URLs.

The remaining Stage 2 TSVs are historical implementation, scaling, and kernel
diagnostics. They are retained for reproducibility and are not current branch
comparisons.
