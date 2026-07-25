# CUDA Step 1 maintainability-refactor gate

Date: 2026-07-24

Status: passed.

## Scope

This gate covers the Stage 1 maintainability cleanup through revision
`29ba9641cefc2c0b2d4562e6c6771562b82b7745`. Stage 2 was deliberately left
unchanged. The final source revision is
`46fc29d000b7d3c0d04e13d1c89a972e224e90a7`; the tip adds its regression
test.

The final tree:

- makes CUDA Level 0 static-input cache identity and lifetime explicit;
- isolates fork-added Level 0 asynchronous pipeline and profiling state;
- encapsulates fork-added logistic and Cox Level 1 profiling;
- makes the fork-added Level 1 design-cache lifetime exception-safe;
- isolates the fork-added resident logistic iteration; and
- makes fork-added Level 0 asynchronous cleanup safe during exception
  unwinding.

The pre-carbocation constraint was applied during the cleanup. An extraction
of upstream-origin Level 0 setup/finalization code was reverted, and the
upstream logistic and Cox fold/numerical paths were left textually unchanged.

## Validation

The CPU gate used the two N2 machines for distinct revisions rather than
duplicate runs. Each checked revision was built with oneMKL and OpenMP in
Release mode and passed all four CTest targets. The final integrated revision
also passed the complete A100 `scripts/test_step1_cuda.sh` gate:

- CPU, CUDA, and automatic backend selection passed;
- all 20 CPU-versus-CUDA numeric-file comparisons had maximum absolute error
  `0.0`;
- quantitative, count, binary, and survival end-to-end cases passed; and
- compute-sanitizer reported `ERROR SUMMARY: 0 errors`.

The A100 gate now also contains a deterministic packed-PGEN Level 0
unwinding regression. It forces a low-memory write failure after block 2 has
been installed by the two-backend CUDA pipeline and the next asynchronous
preprocessing operation has been launched. The process exited normally with
the expected error, and the repeated compute-sanitizer run reported zero
errors.

## Production-scale performance

The final benchmark used the same recent production-sized fixture and retained
controls as the ownership-refactor gate:

- A100 40 GB;
- 500,000 samples, 700,000 model-fitting variants, and eight traits;
- `--bsize 1000`, 12 threads, and SSD-backed Level 0 intermediates; and
- the same PGEN, covariates, phenotypes, options, and seeds.

The final quantitative candidate was a clean CMake `Release` build with
oneMKL and CUDA architecture 80 at source revision `46fc29d`. Its binary
SHA-256 was
`a2c6c135195616c3cfa2aca0ab8b54b118835feaaba2046da40c8bd7cd20da0b`.
The matched control is revision `340677f3`, whose cross-model gate is recorded
in
[`2026-07-24-step1-cuda-refactor-gate.md`](2026-07-24-step1-cuda-refactor-gate.md).
Binary and survival figures below remain the immediately preceding
`99f1152` measurements: the final source change adds an exception-unwinding
teardown fallback without changing the successful compute path, while the
updated complete A100 gate revalidated their successful paths.

### End-to-end timing

| Model | Control internal | Final internal | Change | Control process wall | Final process wall | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Quantitative | 298.234 s | 296.667 s | -0.526% | 300.51 s | 298.83 s | -0.559% |
| Binary logistic | 546.969 s | 547.634 s | +0.121% | 549.55 s | 550.19 s | +0.116% |
| Survival Cox | 489.463 s | 490.308 s | +0.173% | 491.72 s | 492.55 s | +0.169% |

### Stage timing

| Model and stage | Control | Final | Change |
| --- | ---: | ---: | ---: |
| Quantitative Level 0 | 159.092 s | 158.013 s | -0.678% |
| Quantitative Level 1 | 126.178 s | 125.553 s | -0.495% |
| Quantitative output | 9.863 s | 9.751 s | -1.132% |
| Binary Level 0 | 233.453 s | 233.373 s | -0.034% |
| Binary Level 1 logistic | 292.172 s | 293.242 s | +0.366% |
| Binary output | 15.634 s | 15.155 s | -3.064% |
| Survival Level 0 | 234.343 s | 234.615 s | +0.116% |
| Survival Level 1 Cox | 232.802 s | 233.342 s | +0.232% |
| Survival output | 15.629 s | 15.621 s | -0.048% |

The backend counters stayed flat:

| Model | Control backend work | Final backend work | Change |
| --- | ---: | ---: | ---: |
| Quantitative Level 1 | 29.183 s | 29.080 s | -0.352% |
| Binary logistic Level 1 | 260.922 s | 260.921 s | -0.000% |
| Survival Cox Level 1 | 176.556 s | 176.762 s | +0.116% |

The binary run retained exactly 508 IRLS iterations, 508 line-search
iterations, 1,216 predictions, 508 weighted products, 508 scores, and 508
solves. Logistic retained 40 resident uploads and 2,272 reuses; Cox retained
eight resident uploads and 1,411 reuses. No production run entered a fallback
path.

An initial measurement used the CUDA validation build, whose
`CMAKE_BUILD_TYPE` was empty. It was excluded: unoptimized host orchestration
inflated binary Level 1 while GPU backend time remained flat. All figures above
come from the subsequently verified `Release` binary.

## Correctness

All 24 production outputs were byte-for-byte identical to the matched
`340677f3` candidates:

| Model | Result |
| --- | ---: |
| Quantitative | 8/8 byte-identical |
| Binary logistic | 8/8 byte-identical |
| Survival Cox | 8/8 byte-identical |

Their SHA-256 values are unchanged from
[`2026-07-24-step1-cuda-refactor-gate.sha256`](2026-07-24-step1-cuda-refactor-gate.sha256).

Complete final telemetry and raw logs are retained on the A100 at:

- `/home/james/build/regenie-final-unwind-release-46fc29d-qt`;
- `/home/james/build/regenie-final-refactor-release-99f1152-qt/step1-n500k-m700k-p8-qt-final-release-20260724T080644Z`;
- `/home/james/build/regenie-final-refactor-release-99f1152-binary/step1-n500k-m700k-p8-binary-final-release-20260724T081201Z`; and
- `/home/james/build/regenie-final-refactor-release-99f1152-survival/step1-n500k-m700k-p8-survival-final-release-20260724T082133Z`.

## Conclusion

The Stage 1 maintainability refactor passes the final correctness and
performance gate. The final quantitative rerun was 0.526% faster than its
control and all eight outputs were byte-identical. Across the three production
models, the largest end-to-end increase was 0.173%, all cache/call counters
were preserved, and all 24 production outputs remained byte-identical.
