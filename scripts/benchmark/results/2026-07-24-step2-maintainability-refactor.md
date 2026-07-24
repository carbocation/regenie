# Stage 2 maintainability-refactor gate

This gate compares the immediate pre-refactor revision `5063906` with the
final Stage 2 maintainability revision `1f2352d`. The refactor isolates backend
capabilities and build selection, contains the packed-scoring and LOCO-prefetch
lifecycles, and replaces parallel score matrices and validity flags with a
single validated score-batch contract.

The numerical algorithms were not rewritten. The quantitative, binary, and
survival score formulas remain in their established implementations; the
refactor changes how fork-added backend state is selected, published, and
cleaned up.

## Structural result

The final implementation:

- declares LOCO-prefetch and packed-pipeline capabilities on the backend
  interface instead of inspecting a concrete backend name;
- permits a Stage 1 CUDA build without Stage 2 CUDA through
  `REGENIE_WITH_STEP2_CUDA=OFF`;
- owns each outstanding packed-scoring future and its associated block in a
  `Step2PackedPipelineSession`;
- owns the asynchronous LOCO read, requested chromosome, and cleanup in a
  `Step2LocoPrefetchSession`;
- publishes numerators, denominators, and optional observed-trait counts
  atomically through `Step2ScoreBatch`, with dimension and bounds checks;
- makes the exact Cox path reject backend scores locally; and
- includes the Stage 2 conformance target in the CPU, CUDA, and macOS release
  test builds.

The remaining broader PGEN-reader signature and the borrowed quantitative
missing-index pointer were intentionally left alone. Changing them would add
merge surface in upstream-origin code without materially improving CUDA
isolation or fixing a demonstrated defect.

## N2 release and regression gate

Revision `1f2352d` was built on `regenie-n2-cores8` from an archive whose
SHA-256 was
`e12e634cc0663c8d5f583328132300ce65a23c2a3cdd1b713666c0f38b9f4ca1`.
The build used GCC 14.2, oneMKL 2026.1, BGEN v1.1.7, Release optimization, and
eight physical N2 cores.

- Clean release build and packaging passed.
- CTest passed 5/5: Step 1 CPU/auto, Step 2 CPU/auto, and Cox Firth.
- Explicit Stage 2 CPU and auto conformance passed; auto selected CPU.
- Repository regression tests 2-18 passed against both the build-tree and
  packaged binaries.
- The packaged CPU binary had no dynamic oneMKL, C++, or CUDA runtime
  dependencies.

The release build took 144.38 seconds. The managed BGEN download initially
received HTTP 429; the successful run used the host's existing BGEN v1.1.7
source through the release script's explicit `--allow-external-bgen` option.
This was an infrastructure retry, not a source change.

## N=500,000, P=8 CPU correctness and performance

The matched performance gate used the established real-LD-derived fixture:

- 500,000 samples;
- 700,000 variants in the physical PGEN;
- 57,821 chromosome 1 variants tested in Stage 2;
- eight quantitative, binary, or survival outcomes;
- block size 1,000 and eight physical N2 cores;
- CPU backend, oneMKL 2026.1, and `--ignore-pred`; and
- two cache-cold repetitions per revision and model, in alternating
  baseline/final order.

Both revisions were built identically with GCC 14.2, BGEN v1.1.7,
`-O3 -DNDEBUG -march=x86-64-v3`, and static oneMKL linkage.

| Model | `5063906` mean wall | `1f2352d` mean wall | Change | Output gate |
| --- | ---: | ---: | ---: | --- |
| Quantitative | 40.830 s | 40.095 s | -1.800% | 16/16 file pairs byte-identical |
| Binary | 38.160 s | 38.140 s | -0.052% | 16/16 file pairs byte-identical |
| Survival | 43.265 s | 43.225 s | -0.092% | 16/16 file pairs byte-identical |

The internal `STEP2_PROFILE` totals agreed with the external measurements:
-1.754%, -0.058%, and -0.110% for quantitative, binary, and survival,
respectively. Every run analyzed one chromosome, 58 blocks, 57,821 variants,
500,000 samples, and eight phenotypes, with zero corrected or failed tests.

Across all repetitions and models, 48 matched baseline/final association-file
pairs were byte-for-byte identical. Each file had the same schema and 57,822
lines: one header plus all 57,821 variants. Repeated runs within each revision
were also bytewise deterministic.

The first eight outcomes in this established fixture span 0-2.258% missingness.
The 32-outcome fixture reaches approximately 10%, but the requested P=8
performance gate did not use those additional outcomes. The deterministic
backend conformance suite separately covers phenotype-specific missingness and
observed-trait counts.

## A100 CUDA conformance

The same final source archive was configured on `regenie-a100` with CUDA
12.9, architecture 80, GCC 11.4, and oneMKL 2026.1.

- The Release CUDA build passed in 158.67 seconds.
- CTest passed 5/5.
- Explicit Stage 2 CPU, CUDA, and auto conformance passed.
- Auto selected CUDA.
- The deterministic CUDA comparisons cover quantitative complete and missing
  outcomes, binary missingness and observed counts, and Cox scoring against
  CPU references.

The extended Stage-1-only build-boundary check, compute-sanitizer run, and
end-to-end A100 integration were deferred when an unrelated CPU workload
started on the host. No process was interrupted and no competing validation
was launched. These checks remain operational follow-ups rather than failures
of the completed conformance gate.

## Conclusion

The Stage 2 refactor preserves exact CPU scientific output across all three
models and shows no performance regression at N=500,000 and P=8. The CPU
release/regression suite and the primary CUDA backend conformance suite pass.
The code is better isolated at the backend and build boundaries, while the
larger upstream-origin numerical and PGEN structures remain substantially
unchanged for future upstream merges.

Raw remote artifacts are retained at:

- N2 release:
  `/home/james/build/regenie-stage2-n2-release-1f2352d-20260724T1955Z`
- N2 matched integration/performance:
  `/home/james/build/stage2-refactor-e2e-20260724T1955Z`
- A100 CUDA conformance:
  `/home/james/build/regenie-stage2-validation-1f2352d-20260724t1308`
