# Stage 1 Level 1 bounded Newton-CG experiment

Date: 2026-07-25

Status: implementation and validation complete; retained temporarily as a
disabled-by-default experiment pending a keep/remove decision. Enable it with:

```bash
REGENIE_STEP1_LEVEL1_NEWTON_CG=1 regenie ...
```

Enabling Newton-CG also enables the existing path-Newton warm starts. The route
is currently limited to resident-CUDA Level 1 logistic models after the first
ridge parameter. It does not affect quantitative, count, survival, CPU, or
non-resident paths.

## Research question

Can a matrix-free Newton correction finish logistic Level 1 ridge fits faster
than path-Newton plus ordinary IRLS?

The experiment is deliberately bounded:

- path-Newton remains the first warm-start attempt;
- Newton-CG is tried only where path-Newton did not converge;
- the previous full weighted Hessian is factored once and reused for every
  preconditioner application;
- Hessian-vector products use the resident design and independent CUDA
  workspace, without materializing a new Gram or invalidating the retained
  factor;
- at most two outer corrections, eight PCG iterations per correction, and four
  step halvings are allowed;
- every candidate is accepted only when the exact penalized-score norm is
  finite and strictly smaller; and
- ordinary IRLS resumes from the best accepted point whenever Newton-CG does
  not finish the model.

There is no Shampoo-style EMA or gradient-statistic accumulation across
phenotypes or folds. Structured block or Kronecker preconditioners were not
implemented because the previous full Hessian was inexpensive and effective.

## Validation

The implementation was developed in four commits:

```text
896f9a7 feat: add resident Level 1 Hessian products
8443f45 feat: reuse Level 1 path preconditioners
7cd94ba feat: add bounded Newton-CG solver core
a1841d1 feat: add opt-in Level 1 Newton-CG corrections
```

The following gates passed:

- local CPU backend and bounded-PCG unit tests;
- an N2 Release build, all five CTest targets, repository regressions 2-18,
  and packaged regressions;
- a distinct N2 ASan/UBSan run of the Step 1 compute and PCG tests, with no
  sanitizer or leak findings;
- an A100 CUDA Release build and all five CTest targets;
- CUDA backend conformance, including a `1.48649e-16` relative HVP error, a
  `3.41949e-14` reusable-factor solve error, and unchanged cached-Gram and
  retained-factor solves after an HVP;
- compute-sanitizer with zero errors;
- exact CPU/CUDA validation across quantitative, low-memory, test-L0, count,
  binary, and survival routes; and
- repository and packaged regression suites on the CUDA artifact.

Validated artifacts:

```text
/home/james/build/regenie-newton-cg-a1841d1-release
/home/james/build/regenie-newton-cg-a1841d1-sanitizer-test
/home/james/build/regenie-newton-cg-a1841d1-cuda-release
/home/james/build/regenie-newton-cg-a1841d1-cuda-dist/regenie-4.1.2-ga1841d17a6aa-linux-x86_64-native-cuda-sm80.tar.gz
```

## N=500,000, P=8 A100 performance

All runs used commit `a1841d1`, the same A100 40 GB, 700,000 model-fitting
variants, eight binary traits with 0-10% missingness, `--bsize 1000`, 12
threads, and the same retained 106 GB Level 0 cache. These are Level 1 replays,
so identical Level 0 computation was not repeated.

| Route | Order | Level 1 wall (s) | Solver work (s) | Exposed L0-read wait (s) | Program total (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| path-Newton | first | 181.782 | 104.138 | 49.045 | 197.158 |
| Newton-CG | second | 151.075 | 80.216 | 42.404 | 166.647 |
| path-Newton | third | 154.413 | 103.657 | 22.296 | 169.730 |

`Solver work` is IRLS plus path-Newton plus Newton-CG wall time. Against the
reversed-order path control, Newton-CG reduced this work by 22.6% (1.292x), but
Level 1 wall time improved by only 2.2% (1.022x). The faster solver exposed
more SSD input wait that path-Newton had hidden behind compute. Against the
slower first control, Level 1 improved by 16.9% (1.203x), illustrating why the
reversed-order control is necessary.

Newton-CG attempted 92 models, accepted 127 safeguarded steps, and converged 77
models. It used 132 HVPs and 132 preconditioner applications: 1.04 HVPs per
accepted step. Ordinary IRLS iterations fell from 170 to 103. The 15
non-converged Newton-CG attempts fell through to the ordinary-IRLS safety net.

Raw benchmark directories:

```text
/home/james/build/regenie-newton-cg-a1841d1-benchmark/path-p8-l1-20260725T155551Z
/home/james/build/regenie-newton-cg-a1841d1-benchmark/newton-cg-p8-l1-20260725T155926Z
/home/james/build/regenie-newton-cg-a1841d1-benchmark/path-repeat-p8-l1-20260725T160546Z
```

## Numerical comparison

All printed cross-validation metrics, selected ridge parameters, and
missingness patterns matched. Three of eight LOCO files were byte-identical.
For the five differing traits:

| Trait | Compared finite values | Changed serialized values | Maximum absolute difference | RMSE |
| --- | ---: | ---: | ---: | ---: |
| BT1 | 11,500,000 | 81 | `1e-5` | `4.33e-9` |
| BT2 | 11,462,901 | 1,715,246 | `1e-5` | `4.28e-7` |
| BT3 | 11,425,802 | 156,782 | `1e-5` | `4.14e-8` |
| BT4 | 11,388,703 | 10,616 | `1e-6` | `7.97e-9` |
| BT6 | 11,314,505 | 1,448 | `1e-6` | `2.66e-9` |

BT5, BT7, and BT8 were byte-identical. This remains a non-byte-identical route,
as expected for the existing experimental fast/inexact family, but the
observed differences are small and the exact penalized-score acceptance guard
was active for every Newton-CG step.

## Decision point

The experiment did not meet the automatic-retention target of at least 1.5x
Level 1 speedup. It did demonstrate that factor-once/apply-many PCG is
algorithmically effective: approximately one HVP per accepted step and 22.6%
less nonlinear-solver work. On the current SSD-backed N=500,000/P=8 workload,
however, that work reduction produces only a 2.2% warm end-to-end Level 1 gain
because input becomes the bottleneck.

The implementation remains isolated, opt-in, and covered by CPU, sanitizer,
CUDA, and fallback tests while the maintenance-value tradeoff is discussed.
It should not be made the default on this evidence. A keep decision would
preserve it as an experimental option with potential upside if Level 0 input
latency is reduced; a remove decision would drop the Newton-CG integration
while the independently useful resident HVP and reusable-factor backend
primitives could be considered separately.
