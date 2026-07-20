# Benchmarking REGENIE

This directory holds the scripts and result snapshots for repeatable REGENIE
performance runs. `run_profiled.sh` wraps an ordinary REGENIE command, so no
separate benchmark binary is needed. It works for production-scale Step 1 runs
and shorter Step 2 throughput checks.

For each run, the wrapper saves:

- the exact shell-escaped command, revision, binary checksum, CPU topology,
  linked libraries, memory, disks, and GPU configuration;
- external wall time, CPU time, peak RSS, page faults, context switches, and
  filesystem operations from GNU `time`;
- one-second `vmstat` and `iostat` traces when those tools are installed;
- GPU utilization, memory-controller utilization, allocated memory, power,
  power limit, clocks, and temperature at a configurable interval;
- raw and timestamped console logs, plus every structured
  `STEP1_PROFILE` or `STEP2_PROFILE` field in long-form TSV; and
- interval-based host CPU utilization, I/O wait, and run-queue summaries from
  `vmstat`, plus whole-run and phase-level GPU summaries.

Example:

```bash
scripts/benchmark/run_profiled.sh \
  --label a100-n500k-m700k-p1 \
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
    --qt --bsize 1000 --threads 6 --lowmem \
    --lowmem-prefix /path/to/results/l0 \
    --compute-backend cuda --gpu-device 0 --step1-profile \
    --out /path/to/results/fit
```

Use a stable `--system-label` that describes the hardware configuration, not
the cloud instance name. This keeps result tables meaningful after the
original machines have been deleted or renamed.

Build the binary the way users are expected to run it. For optimized x86
measurements, set `MKLROOT` when configuring REGENIE and confirm that
`binary_libraries.txt` links to oneMKL. A BLAS mismatch can dominate the CPU
result.

`prepare_multitrait_fixture.py` makes a deterministic multi-phenotype cohort
from aligned phenotype and covariate files. By default it keeps 10,000 samples
per population group, generates 32 correlated quantitative traits, and writes
both a complete-case file and a version whose per-trait missingness increases
from 0% to 10%. It also writes a PLINK keep file; use that file to make a
physical PGEN subset so each benchmark reads the same number of samples. The
helper requires NumPy.

`prepare_nongaussian_fixture.py` derives binary and time-to-event outcomes
from the complete quantitative-trait table. Its defaults create eight binary
traits spanning 1% to 50% prevalence and eight survival traits spanning 10% to
80% observed events. Counts are exact and generation is deterministic, which
makes it straightforward to reproduce the same phenotype files on different
machines and verify them by checksum.

Each invocation creates `LABEL-YYYYMMDDTHHMMSSZ/`. `summary.tsv` is the compact
run summary, `profile_kv.tsv` preserves the structured REGENIE profile, and
`gpu.csv` is the raw telemetry. The wrapper exits with REGENIE's status, so it
can be used directly from CI or a batch scheduler.

## Recorded checkpoints

- [`results/2026-07-19-production.md`](results/2026-07-19-production.md) records
  the current A100, T4, and eight-core N2 benchmark on a shared real-LD-derived
  fixture, including v4.1.2 Step 1 and Step 2 CPU anchors, GPU utilization and
  power, quantitative, binary, and survival workloads, the retained Level 1
  GPU residency changes, phase bottlenecks, and Step 2 thread scaling.
- `results/2026-07-19-production.tsv` contains the retained raw run-level
  measurements used by that report.
