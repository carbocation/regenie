# CUDA Step 1 refactor performance gate

Date: 2026-07-24

Status: passed.

## Purpose

This gate checks that the `CudaStep1ComputeBackend` ownership and cache-state
refactor through revision `340677f3` did not introduce an unexpected
performance or correctness regression.

The benchmark uses the recent production-sized quantitative workload rather
than the older N=50,000 fixture:

- A100 40 GB;
- 500,000 samples and 700,000 model-fitting variants;
- eight quantitative traits with 0-10% input missingness;
- `--bsize 1000`, 12 threads, seed `20260722`;
- FP64 CUDA Step 1 with SSD-backed Level 0 intermediates; and
- the same PGEN, covariates, phenotypes, options, and hardware as the retained
  2026-07-23 `current-qt8` baseline.

The candidate was built from revision
`340677f3be22eddaa3032b901b4952116299371e`. Its binary SHA-256 was
`8f2e3552ddccaca4070315201a2dfec9237404a859d7a691f08072b64ea60c73`.

## Result

| Metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Internal total | 294.933 s | 298.234 s | +1.12% |
| Process wall time | 297.19 s | 300.51 s | +1.12% |
| Level 0 wall | 157.754 s | 159.092 s | +0.85% |
| Level 1 fit | 124.142 s | 126.178 s | +1.64% |
| Output | 9.683 s | 9.863 s | +1.86% |

The affected backend compute measurements were effectively flat:

| Compute scope | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Genotype preprocessing | 28.288 s | 28.361 s | +0.26% |
| Cross-validation matrices | 40.983 s | 40.990 s | +0.02% |
| Level 0 ridge kernels | 15.466 s | 15.386 s | -0.52% |
| Level 1 backend work | 29.205 s | 29.183 s | -0.08% |

The small end-to-end increase was concentrated outside the refactored compute
work: Level 0 ridge transfer rose from 56.251 to 57.332 seconds, and Level 1
low-memory read wait rose from 93.945 to 96.044 seconds. The candidate used
all 711 packed/resident CUDA blocks with zero fallback blocks.

## Correctness

The run completed with exit status zero. All eight candidate LOCO files were
byte-for-byte identical to the retained validated controls. Their SHA-256
values were unchanged:

1. `4712f8d5316498965bb7f091e0c744ac082fa8a819840929616d6cc388f2e1cb`
2. `e09cd9c0ab1377cdd30cf2d61c6ef93f6e7d94637eb68ad1af4eedb98e42e3a7`
3. `bf7ec387e08d320ce96bfb137f9e052b6877d35fa97b8af571d62b2a67df6fed`
4. `7b18fd52baed9153aa212c74f69aece7a71c34942dc62ed6e3295ab69672449a`
5. `b064f22219edb5804d630ed9dc72928e44344d92768a6db99fed431dbbd20d3d`
6. `b1faa0e67fc4c986112e939e80c3b25f97994077147fe904932c9c0528bee050`
7. `0d7f3a554350ffbd9a4a5cc9e02d5674f354c90cf558b0ff99e236513b95c92c`
8. `ef664222288931fb0c0aabd456d9f0241ef93fef8720667eba12d192a197325e`

The complete candidate telemetry and raw logs are retained on the A100 at
`/home/james/build/regenie-refactor-perf-340677f/step1-n500k-m700k-p8-qt-refactor-20260724T052051Z`.

## Conclusion

The refactor passes the performance gate. End-to-end time remained within
1.12% of the matched baseline, the CUDA compute scopes were unchanged within
-0.52% to +0.26%, no fallback path was entered, and every final output was
byte-identical.
