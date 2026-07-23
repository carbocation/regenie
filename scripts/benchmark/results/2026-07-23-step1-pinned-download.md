# Default CUDA Level 0 pinned downloads

Date: 2026-07-23

Status: retained and enabled by default. No feature flag is required.

## Research question

Can the variable-missingness Level 0 path avoid the CUDA penalty for copying
large fold-prediction matrices into pageable Eigen storage without changing
any computed value?

The CUDA backend now temporarily registers each already-allocated prediction
and coefficient destination with `cudaHostRegister`, launches the existing
five fold-lane device-to-host copies, synchronizes those lanes, and unregisters
the destinations. Registration failure falls back to the existing pageable
copy behavior. The arithmetic, device buffers, copy order within each fold,
and host post-processing are unchanged.

This closes a gap between paths: the complete-case, device-normalized Level 0
path already registered its output destination, while the
variable-missingness path did not.

## Stage 1 performance

All runs used the same A100 40 GB, N=500,000, 700,000 model-fitting variants,
binary traits with 0-10% missingness, `--bsize 1000`, 12 threads, FP64 CUDA
solves, and SSD-backed Level 0 intermediates.

| Traits | Implementation | Level 0 (s) | Level 0 ridge (s) | Ridge transfer (s) | Level 1 (s) | Total (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | matched FP64 reference | 309.579 | — | — | 293.186 | 617.837 |
| 8 | path-Newton only | 310.281 | 232.417 | 90.431 | 157.339 | 482.626 |
| 8 | pinned download, trial A | 234.579 | 156.863 | 19.884 | 156.864 | 412.171 |
| 8 | final default, no-flags replay | **275.457** | **197.495** | **56.415** | 157.149 | **453.624** |
| 32 | current reference | 965.997 | — | — | 1,144.876 | 2,179.918 |
| 32 | path-Newton only | 981.171 | 901.874 | 363.493 | 614.276 | 1,664.349 |
| 32 | default path-Newton + pinned download | **843.341** | **763.618** | **232.617** | 614.274 | **1,526.362** |

The two byte-identical P=8 trials exposed variable Linux page-registration
cost. Trial A reduced Level 0 transfer by 78.0% and the path-Newton total by
14.6%. The final no-flags replay reduced transfer by 37.6%, Level 0 by 11.2%,
and the path-Newton total by 6.0%. Even the slower replay leaves the two
retained changes 26.6% faster than the matched FP64 reference. The variability
is a performance consideration, not a correctness difference, and motivates
a future persistent pinned-buffer pool.

At P=32, pinning reduced Level 0 transfer by 36.0%, Level 0 by 14.0%,
and the path-Newton total by 8.3%. The two retained changes together reduced
the current reference total by 30.0%, from 2,179.918 to 1,526.362 seconds.

The P=32 run successfully registered all 7,110 destinations: 459.52 GB
cumulatively over 711 blocks. At any one block, the registered prediction and
coefficient destinations occupy approximately 162 MB at P=8 and 646 MB at
P=32. Registration and unregistration overhead is included in the reported
transfer time.

## Correctness and final scientific output

The finalized no-flags default build passed the CPU and CUDA backend
conformance suites. Its full P=8 replay produced eight of eight LOCO files
byte-for-byte identical to the matched path-Newton-only control. The P=32
validation produced 32 of 32 exact files. Therefore the pinned download
changes no Stage 2 input and cannot change any final Stage 2 read-out.

Relative to the original non-path-Newton reference, the final Stage 2 result
remains the already-validated path-Newton result: low-order printed digits can
differ, but top-100 and top-1,000 sets, significance-threshold memberships,
and notable-effect signs were identical across 22.4 million P=32 association
rows. Full details are in `2026-07-23-step1-path-newton.md`.

## Rejected approximate experiment

A TensorFloat-32 Level 1 weighted-Gram experiment made an individual IRLS
iteration about 19% cheaper, but the exact penalized score stalled near
`4.4e-3` on the first P=8 fit instead of satisfying the existing convergence
tolerance. The experiment was stopped and all code reverted rather than
weakening convergence.

## Decision

Pinned Level 0 downloads are byte-exact, have a graceful fallback, and match
the existing complete-case design, so they are unconditional rather than
hidden behind a feature flag.

Path-Newton is also enabled by default after its full P=8/P=32 Stage 2
validation placed its numerical differences within the project's historically
accepted envelope. `REGENIE_STEP1_LEVEL1_PATH_NEWTON=0` remains available as
a diagnostic compatibility escape hatch. A user-facing command-line opt-in is
therefore not needed for either performance improvement.
