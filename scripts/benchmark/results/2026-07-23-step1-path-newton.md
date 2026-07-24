# Stage 1 Level 1 path-Newton continuation

Date: 2026-07-23

Status: retained as an experimental opt-in after full P=8/P=32 Stage 2
validation. It can be enabled with:

```bash
REGENIE_STEP1_LEVEL1_PATH_NEWTON=1 regenie ...
```

Do not treat this as a byte-identical optimization. It preserves the
statistical model specification, ridge grids, convergence tolerance, and
downstream scientific conclusions in the tests below, but it changes rounding
in final Stage 2 read-outs.

## Research question

Can the final weighted Hessian from one Level 1 logistic ridge fit accelerate
the next point on the same ridge path without changing the statistical model?

At the converged coefficient vector for the previous ridge parameter, the
penalized score for the next parameter is available exactly:

```text
score(new lambda) =
  score(old lambda) - (new lambda - old lambda) * D * beta
```

The experiment reuses the resident CUDA weighted Gram matrix as a Hessian for
up to four monotone chord-Newton corrections. Each correction evaluates the
exact logistic score. A non-finite or non-improving correction is rejected,
and ordinary IRLS resumes from the best accepted coefficient vector. The
feature does not change the five Level 0 or five Level 1 ridge parameters.

## Stage 1 performance

Both workloads used an A100 40 GB, FP64 CUDA solves, N=500,000, 700,000
model-fitting variants, binary traits with 0-10% missingness, `--bsize 1000`,
12 threads, and SSD-backed Level 0 intermediates.

| Traits | Implementation | Level 0 (s) | Level 1 (s) | Total (s) | Level 1 weighted Grams |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | ordinary IRLS, pre-transport | 309.579 | 293.186 | 617.837 | 508 |
| 8 | path-Newton, pre-transport | 310.281 | 157.339 | 482.626 | 170 |
| 32 | ordinary IRLS, pre-transport | 965.997 | 1,144.876 | 2,179.918 | 2,038 |
| 32 | path-Newton, pre-transport | 981.171 | 614.276 | 1,664.349 | 695 |

The experiment reduced Level 1 by 46.3% at both P=8 and P=32. End-to-end
time fell by 21.9% in the matched P=8 pair and 23.7% at P=32.

Those rows isolate path-Newton before the later unconditional Level 0
transport change. The current matched opt-in sensitivity includes both:
589.753 to 453.624 seconds at P=8 and 2,056.154 to 1,526.362 seconds at P=32.
The cross-product is recorded in `2026-07-23-step1-pinned-download.md`.

At P=32, the method made 2,560 path corrections, allowed 323 ridge fits to
finish without another ordinary IRLS iteration, and reduced weighted-Gram
construction calls by 65.9%.

The finalized source passed the CPU and CUDA backend conformance suites. The
CUDA cached-Hessian solve matched the CPU oracle to `3.46e-14` relative
error. A final P=8 Level 1 replay reproduced all eight measured-candidate LOCO
files byte-for-byte; this is an implementation-regression check, not the
downstream scientific-output gate.

## Final Stage 2 validation

The P=8 A/B comparison used the exact same phenotype/covariate fixture and
seed, then tested all 700,000 variants in Stage 2 for all eight traits.

No Stage 2 result file was byte-identical. Depending on phenotype, 16 to
114,284 of 700,000 rows changed in at least one printed numeric field. The
maximum absolute printed differences were:

| Field | Maximum absolute difference |
| --- | ---: |
| `BETA` | 0.000001 |
| `SE` | 0.0000001 |
| `CHISQ` | 0.0001 |
| `LOG10P` | 0.00001 |

Across all traits:

- every numeric-field correlation exceeded 0.9999999999998;
- top-100 and top-1,000 variants were identical;
- membership at `p <= 1e-5` and genome-wide significance was identical; and
- there were no sign changes among variants with `abs(z) >= 3` in either run.

The matched P=32 A/B repeated the test over 22.4 million association rows
(32 traits by 700,000 variants). Between 8 and 114,696 rows per trait changed
in a printed numeric field. The maximum differences were identical to the
P=8 maxima above, all numeric-field correlations exceeded
0.99999999999984, and all 32 traits retained identical top-100, top-1,000,
`p <= 1e-5`, and genome-wide-significant sets. There were again no sign
changes among variants with `abs(z) >= 3`.

## Rejected alternatives

A post-selection canonical-refit experiment used path-Newton only for the
cross-validation grid, then discarded its selected coefficients and refit the
winning penalty with ordinary FP64 IRLS from zero. The premise of one final
fit per trait does not apply to this Stage 1 path: LOCO predictions are made
from five held-out-fold models, so preserving the algorithm requires five
refits per trait. Matched logs first confirmed that path-Newton and ordinary
IRLS selected the same penalty for 8/8 P=8 and 32/32 P=32 traits; the P=8
sequence was `5, 5, 2, 2, 5, 5, 3, 3`. The P=8 refits then added 53.5 seconds
and still left 0/8 LOCO files byte-identical. Total time was 507.154 seconds,
compared with 453.624 for path-Newton and 589.753 for ordinary IRLS.

The refit reduced changed printed LOCO values from 1,620,880 to 1,072,964 out
of 92 million, but not consistently by trait. In the full P=8 Stage 2 suite,
it reduced the aggregate number of association rows with a changed printed
numeric field from 467,231 to 386,994 out of 5.6 million, while increasing
the worst per-trait count from 114,284 to 149,929. Maximum differences were
unchanged, and both versions retained identical top-100, top-1,000, `p <=
1e-5`, and genome-wide-significant sets with no strong-signal sign flips.
This inconsistent rounding improvement did not strengthen either the exact
or scientific contract enough to justify giving back 39% of path-Newton's
time saving. P=32 was therefore not run.

A three-point Level 0 ridge grid reduced the P=8 Stage 1 total to 323.710
seconds, but changed top-ranked variants and significance-threshold
membership in Stage 2. It was rejected.

Eight path corrections reduced P=8 Level 1 by only another 1.8 seconds
relative to four corrections and increased final-read-out drift. Four
corrections are the selected Pareto point.

A single correction reduced P=8 Level 1 to 220.242 seconds and left four of
eight Stage 2 files byte-identical, but it still failed the strict output
contract while giving up 62.9 seconds of Level 1 performance versus four
corrections.

## Decision

The scaling result is large and repeatable enough to retain the implementation.
The final Stage 2 outputs are scientifically equivalent under the checks above
and their differences are within the project's historically accepted
numerical envelope. Because it changes printed final Stage 2 results, however,
it remains disabled by default and available only through the explicit
environment opt-in above.
