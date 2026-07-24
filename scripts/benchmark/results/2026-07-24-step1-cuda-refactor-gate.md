# CUDA Step 1 refactor cross-model performance gate

Date: 2026-07-24

Status: passed.

## Purpose

This gate checks that the `CudaStep1ComputeBackend` ownership and cache-state
refactor through revision `340677f3` did not introduce an unexpected
performance or correctness regression in the shared Level 0 path or in the
quantitative, logistic, and Cox Level 1 paths.

The benchmark uses the recent production-sized workloads rather than the older
N=50,000 fixture:

- A100 40 GB;
- 500,000 samples and 700,000 model-fitting variants;
- eight traits per model using phenotype fixtures with input missingness;
- `--bsize 1000` and 12 threads;
- FP64 CUDA Step 1 with SSD-backed Level 0 intermediates; and
- the same PGEN, covariates, model-specific phenotypes, options, seeds, and
  hardware as the retained 2026-07-23 controls.

The candidate was built from revision
`340677f3be22eddaa3032b901b4952116299371e`. Its binary SHA-256 was
`8f2e3552ddccaca4070315201a2dfec9237404a859d7a691f08072b64ea60c73`.
The retained nonlinear baseline binary SHA-256 was
`6684225b23cc9e5c18f31744057847dadf2cfb0fe06bbe571e05a18012a2f08b`;
its source snapshot was not a Git checkout, so the raw log and source snapshot
are authoritative.

## Result

### End-to-end timing

| Model | Baseline internal | Candidate internal | Change | Baseline process wall | Candidate process wall | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Quantitative | 294.933 s | 298.234 s | +1.119% | 297.19 s | 300.51 s | +1.119% |
| Binary logistic | 589.753 s | 546.969 s | -7.254% | 592.63 s | 549.55 s | -7.269% |
| Survival Cox | 489.381 s | 489.463 s | +0.017% | 491.65 s | 491.72 s | +0.014% |

### Stage timing

| Model and stage | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Quantitative Level 0 | 157.754 s | 159.092 s | +0.849% |
| Quantitative Level 1 | 124.142 s | 126.178 s | +1.640% |
| Quantitative output | 9.683 s | 9.863 s | +1.860% |
| Binary Level 0 | 275.605 s | 233.453 s | -15.294% |
| Binary Level 1 logistic | 293.187 s | 292.172 s | -0.346% |
| Binary output | 15.145 s | 15.634 s | +3.227% |
| Survival Level 0 | 233.525 s | 234.343 s | +0.350% |
| Survival Level 1 Cox | 233.425 s | 232.802 s | -0.267% |
| Survival output | 15.704 s | 15.629 s | -0.480% |

The compute counters for the code affected by the refactor were flat across
all three models:

| Model | Genotype preprocessing | CV matrices | Level 0 ridge kernels | Level 1 backend work |
| --- | ---: | ---: | ---: | ---: |
| Quantitative | 28.288 → 28.361 s (+0.258%) | 40.983 → 40.990 s (+0.016%) | 15.466 → 15.386 s (-0.518%) | 29.205 → 29.183 s (-0.075%) |
| Binary logistic | 28.397 → 28.337 s (-0.213%) | 41.045 → 41.037 s (-0.020%) | 13.289 → 13.341 s (+0.390%) | 260.765 → 260.922 s (+0.060%) |
| Survival Cox | 28.277 → 28.344 s (+0.236%) | 40.881 → 40.885 s (+0.009%) | 14.743 → 14.729 s (-0.097%) | 176.592 → 176.556 s (-0.020%) |

For logistic and Cox, “Level 1 backend work” is the sum of the reported
upload, Gram, cross-product, ridge, and download counters.

The quantitative increase was concentrated outside backend compute: Level 0
ridge transfer rose from 56.251 to 57.332 seconds, and Level 1 low-memory read
wait rose from 93.945 to 96.044 seconds. The binary candidate was faster
because its Level 0 transfer measurement was 19.350 seconds rather than the
baseline's 55.462 seconds. That single-run improvement is evidence of no
regression, but is not attributed to the ownership refactor. Survival was
effectively unchanged end to end.

Every candidate used all 711 packed/resident CUDA blocks with zero fallback
blocks. Logistic reported eight resident-design phenotypes, 40 resident-design
uploads, and 2,272 reuses. Cox reported eight resident-design phenotypes,
eight uploads, and 1,411 reuses.

## Correctness

All three candidate runs completed with exit status zero:

| Model | Comparison | Result |
| --- | --- | ---: |
| Quantitative | candidate `_1` through `_8` vs retained `current-qt8` controls | 8/8 byte-identical |
| Binary logistic | candidate `_1` through `_8` vs retained `headline-binary8` controls | 8/8 byte-identical |
| Survival Cox | candidate `{1,3,5,7,9,11,13,15}` vs the matched P=8 `survival8/fit` controls | 8/8 byte-identical |

The survival comparison deliberately uses the P=8 controls. Survival P=8 is
not expected to be byte-identical to the first eight fits from a P=32 run
because phenotype count affects the fit.

Full candidate output SHA-256 values are recorded in the adjacent
[`2026-07-24-step1-cuda-refactor-gate.sha256`](2026-07-24-step1-cuda-refactor-gate.sha256)
manifest.

Complete candidate telemetry and raw logs are retained on the A100 at:

- `/home/james/build/regenie-refactor-perf-340677f/step1-n500k-m700k-p8-qt-refactor-20260724T052051Z`;
- `/home/james/build/regenie-refactor-perf-340677f-binary/step1-n500k-m700k-p8-binary-refactor-20260724T053510Z`; and
- `/home/james/build/regenie-refactor-perf-340677f-survival/step1-n500k-m700k-p8-survival-refactor-20260724T054518Z`.

## Conclusion

The refactor passes the cross-model performance and correctness gate. The
largest end-to-end increase was 1.119% for quantitative traits; survival was
flat and binary was faster. The affected CUDA compute counters ranged from
-0.518% to +0.390%, no fallback path was entered, resident design was exercised
by both nonlinear models, and all 24 final outputs were byte-identical to their
matched controls.
