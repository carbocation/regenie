# Default CUDA Level 0 pinned downloads

Date: 2026-07-23

Status: retained and enabled unconditionally. No feature flag is required for
the byte-exact Level 0 transport change.

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

## Cross-model transport gate

The unconditional change was checked at P=8 and P=32 for quantitative,
binary, and survival traits. All runs used the same A100 40 GB, N=500,000,
700,000 model-fitting variants, `--bsize 1000`, 12 threads, FP64 CUDA solves,
and SSD-backed Level 0 intermediates. Path-Newton was disabled or unset.

| Model | Traits | Pre-change Level 0 (s) | Current Level 0 (s) | Pre-change total (s) | Current total (s) | Final Stage 1 output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Quantitative | 8 | 158.946 | 157.754 | 295.088 | 294.933 | 8/8 LOCO files byte-identical |
| Quantitative | 32 | 533.339 | 533.537 | 1,122.427 | 1,124.013 | 32/32 LOCO files byte-identical |
| Binary | 8 | 309.579 | 275.605 | 617.837 | 589.753 | 8/8 LOCO files byte-identical |
| Binary | 32 | 965.997 | 843.412 | 2,179.918 | 2,056.154 | 32/32 LOCO files byte-identical |
| Survival | 8 | 319.932 | 233.525 | 575.711 | 489.381 | 8/8 LOCO files byte-identical |
| Survival | 32 | 1,008.611 | 854.424 | 1,989.062 | 1,837.541 | 32/32 LOCO files byte-identical |

Quantitative does not enter the newly registered variable-missingness result
path; its P=8 total changed by -0.05% and its P=32 total by +0.14%, both run
noise. The new code therefore does not regress the unaffected model.

For binary, the current default reduced matched P=8 total time by 4.5% and
P=32 by 5.7%. For survival it reduced P=8 by 15.0% and P=32 by 7.6%.
The unusually fast P=8 survival registration trial transferred ridge results
in 19.35 seconds; the P=32 result, 235.12 seconds versus 398.37 seconds
before the change, demonstrates that the gain also scales under the larger
459.52 GB cumulative transfer workload.

Because every default LOCO file is byte-identical to its matched pre-change
control, the transport change leaves the input to Stage 2 unchanged and
therefore cannot alter the final Stage 2 read-out.

## Binary factorial performance

All runs used the same A100 40 GB, N=500,000, 700,000 model-fitting variants,
binary traits with 0-10% missingness, `--bsize 1000`, 12 threads, FP64 CUDA
solves, and SSD-backed Level 0 intermediates.

| Traits | Implementation | Level 0 (s) | Level 0 ridge (s) | Ridge transfer (s) | Level 1 (s) | Total (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | ordinary IRLS, pageable reference | 309.579 | 231.695 | 90.383 | 293.186 | 617.837 |
| 8 | ordinary IRLS + pinned download | **276.374** | **198.436** | **57.147** | 292.130 | **590.403** |
| 8 | path-Newton only | 310.281 | 232.417 | 90.431 | 157.339 | 482.626 |
| 8 | path-Newton + pinned download | **275.457** | **197.495** | **56.415** | 157.149 | **453.624** |
| 32 | ordinary IRLS, pageable reference | 965.997 | 886.691 | 360.450 | 1,144.876 | 2,179.918 |
| 32 | ordinary IRLS + pinned download | **843.412** | **763.320** | **231.726** | 1,143.685 | **2,056.154** |
| 32 | path-Newton only | 981.171 | 901.874 | 363.493 | 614.276 | 1,664.349 |
| 32 | path-Newton + pinned download | **843.341** | **763.618** | **232.617** | 614.274 | **1,526.362** |

With path-Newton explicitly disabled, pinning reduced P=8 transfer by 36.8%,
Level 0 by 10.7%, and total time by 4.4%. At P=32 it reduced transfer by
35.7%, Level 0 by 12.7%, and total time by 5.7%. These are direct full-run
measurements, not projections.

The gains stack. At P=8 the additive prediction from the two isolated wins is
455.192 seconds, versus an observed combined time of 453.624 seconds. At P=32
the additive prediction is 1,540.585 seconds, versus 1,526.362 observed.
The interactions are only 1.568 and 14.223 seconds, or 0.3% and 0.7% of the
respective baselines. When path-Newton is explicitly enabled, the two changes
together reduce P=8 by 26.6% and P=32 by 30.0%.

Linux page-registration latency is variable. An additional byte-identical P=8
trial completed in 412.171 seconds with 19.884 seconds of transfer, versus
453.624 seconds and 56.415 seconds in the final combined replay. Two
subsequent P=8 measurements, including the ordinary-IRLS isolation,
reproduced the 56-57-second range. Pinning remained faster in every matched
run, but the variance motivates a future persistent pinned-buffer pool.

The P=32 run successfully registered all 7,110 destinations: 459.52 GB
cumulatively over 711 blocks. At any one block, the registered prediction and
coefficient destinations occupy approximately 162 MB at P=8 and 646 MB at
P=32. Registration and unregistration overhead is included in the reported
transfer time.

## Correctness and final scientific output

The finalized no-flags default build passed the CPU and CUDA backend
conformance suites. With path-Newton disabled, the pinned-only runs produced
eight of eight P=8 and 32 of 32 P=32 LOCO files byte-for-byte identical to the
ordinary-IRLS references. The combined runs likewise matched all path-Newton
controls. Therefore the pinned download changes no Stage 2 input and cannot
change any final Stage 2 read-out, regardless of whether path-Newton is used.

Pinned-only final Stage 2 output is thus identical to the original reference.
When path-Newton is enabled, the final Stage 2 result remains its previously
validated result: low-order printed digits can differ, but top-100 and
top-1,000 sets, significance-threshold memberships, and notable-effect signs
were identical across 22.4 million P=32 association rows. Full details are in
`2026-07-23-step1-path-newton.md`.

## Rejected approximate experiment

A TensorFloat-32 Level 1 weighted-Gram experiment made an individual IRLS
iteration about 19% cheaper, but the exact penalized score stalled near
`4.4e-3` on the first P=8 fit instead of satisfying the existing convergence
tolerance. The experiment was stopped and all code reverted rather than
weakening convergence.

## Decision

Pinned Level 0 downloads are byte-exact, independently improve both ordinary
IRLS and path-Newton runs, have a graceful fallback, and match the existing
complete-case design. They should remain unconditional rather than hidden
behind a feature flag.

Path-Newton is a separate, non-byte-identical experiment. Its full P=8/P=32
Stage 2 validation placed the numerical differences within the project's
historically accepted envelope, but it remains disabled by default and can be
enabled explicitly with `REGENIE_STEP1_LEVEL1_PATH_NEWTON=1`.
