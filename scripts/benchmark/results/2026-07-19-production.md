# Production performance checkpoint: 2026-07-19

This report is a performance snapshot of `114ef81` at production-like Step 1
scale, with upstream v4.1.2 (`5f924b9`) as a CPU baseline. Unless noted
otherwise, timings are external wall time. Step 1 runs use quantitative traits,
five-fold cross-validation, `--bsize 1000`, `--lowmem`, and the default Level 0
and Level 1 ridge grids.

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

## Headline Step 1 results

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

## GPU saturation

GPU telemetry was sampled every 0.5 seconds on A100 and every second on T4.

| Run and phase | Mean GPU util. | Median | p10 / p90 | Samples >=90% | Mean power | Mean of limit | Peak device memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A100 700k, whole run | 50.6% | 58% | 0% / 64% | 0.3% | 185 W | 46.3% | 6,432 MiB |
| A100 700k, Level 0 | 58.8% | 58% | 56% / 64% | 0.0% | 212 W | 53.1% | 4,890 MiB |
| A100 700k, Level 1/output | 15.9% | 0% | 0% / 39.8% | 0.0% | 71 W | 17.8% | 6,432 MiB |
| T4 700k, whole run | 97.0% | 100% | 98% / 100% | 95.0% | 65 W | 93.5% | 6,107 MiB |
| T4 700k, Level 0 | 98.7% | 100% | 99% / 100% | 96.8% | 66 W | 94.3% | 4,567 MiB |
| T4 700k, Level 1/output | 35.1% | 0% | 0% / 100% | 28.7% | 44 W | 62.5% | 6,107 MiB |

The T4 is saturated during Level 0. Its full run spends 3,768.12 of 4,567.98
Level 0 seconds (82.5%) in Gram construction. It physically reads 58.1 GB, but
205.82 seconds of decoder service overlap GPU work and foreground wait is only
0.68 seconds. The host averages 13.6% busy across eight vCPUs. This is a
device-compute limit, not a host-thread or input-pipeline limit.

The A100 is not saturated. Level 0 uses only about 59% of the GPU and 12% of
its 40 GB memory. Its utilization is also extremely steady (56% p10, 64% p90),
which points to a repeated per-block work/transfer/orchestration ceiling rather
than occasional I/O stalls. Level 1 and prediction output leave the device
mostly idle. Level 0 power nevertheless reaches 363 W at p90 and 392 W at its
maximum, so individual kernels can drive the device; the low mean is consistent
with gaps and lower-intensity work between those bursts, not a low power cap.

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

The largest near-term opportunity is to keep more Level 0 state resident on
the device and submit larger units of work. Ridge spends only 16% of its wall
time in device compute; 61% is host orchestration and another 23% is transfer.
Genotype preprocessing uploads 87.5 GB of packed hardcalls and spends almost
as long transferring them as computing on them. The 40 GB A100 has ample
unused capacity for deeper block batching or persistent intermediates.

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

## Reproduction and interpretation notes

- Every retained run exited successfully. Current-branch runs use `114ef81`;
  the upstream CPU anchor is v4.1.2 at
  `5f924b9bf54c1c7597174345def6eb2f1dee712c`.
- The current binaries passed all three CTest targets on each system. Dynamic
  linkage was checked before measurement: all x86 builds use oneMKL 2026.1,
  and the GPU builds also link to the expected CUDA libraries. The full
  upstream run reproduced the current ridge scores and numerically equivalent
  LOCO predictions.
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
