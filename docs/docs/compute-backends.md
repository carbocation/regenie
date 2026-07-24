# Compute backend architecture

This page describes the contributor contract around the Step 1 and Step 2
compute backends. It is intentionally about ownership, matrix layouts, state
lifetimes, and fallback behavior. User-facing build and tuning instructions
remain in [Install](install.md#experimental-cuda-backends).

## Code map and ownership

| Area | Interface and CPU implementation | CUDA implementation | Main callers |
| --- | --- | --- | --- |
| Step 1 | `src/Step1_Compute.hpp`, `src/Step1_Compute.cpp` | `src/Step1_Compute_CUDA.cu` | `src/Data.cpp`, `src/Step1_Models.cpp`, `src/cox_ridge.cpp` |
| Step 2 | `src/Step2_Compute.hpp`, `src/Step2_Compute.cpp` | `src/Step2_Compute_CUDA.cu` | `src/Data.cpp` |

`src/Cuda_Resources.hpp` contains the shared RAII guards for transient CUDA
events and host registrations. Use those guards for scoped resources instead
of adding raw create/destroy pairs to either backend.

The model code owns statistical decisions and the host fallback. A compute
backend owns linear algebra, transfers, reusable workspaces, and any explicitly
cached device state. Genotype readers remain outside the backend, except that
eligible PGEN hardcall paths pass their packed bytes directly to CUDA.

Step 1 is a broad numerical interface. Its mandatory operations cover products,
eigendecomposition, ridge prediction, and reusable factorizations. Its optional
operations expose resident-genotype, resident-design, batched-fold, and fused
solve fast paths.

Step 2 has a narrower prepare/score interface. A successful `prepare_*` call
caches one chromosome's null-model terms. The selected backend can then score
multiple genotype blocks against that state.

## Backend selection and fallback

`--compute-backend` is shared by both steps:

- `cpu` constructs the CPU backend.
- `auto` attempts to construct CUDA when the binary contains CUDA support and
  the selected device is visible; otherwise it constructs CPU.
- `cuda` requires a CUDA-enabled binary and a visible `--gpu-device`, and
  throws if either condition is not met.

A successfully constructed CUDA backend does not imply that every operation or
workflow will run on the GPU.

In Step 1, optional methods return `false` when a specialization is unsupported,
disabled, too large for its configured resident budget, or otherwise
inapplicable. Callers must retain a complete generic path. Methods that return
`void` are mandatory or require state established by an earlier call; malformed
inputs and invalid call order throw.

In Step 2, each `prepare_*` starts by clearing the previous mode. A `false`
result leaves `ready()` false, and `Data` uses the ordinary per-variant CPU
path. This per-workflow fallback also applies when `cuda` was requested
explicitly: explicit CUDA requires the device, not support for every analysis
mode. A prepared block scorer may also return `false`; the caller must not
consume its output matrices in that case.

Do not use `name()` as an operation-level capability test. Step 2's CUDA
backend may report `auto` before preparation and `cpu` after an attempted but
ineligible preparation. Use `ready()`, `uses_packed_hardcalls()`,
`provides_observed_trait_counts()`, or the relevant operation's Boolean result.

## Matrix and output layouts

Eigen's default column-major storage is assumed at the CUDA boundary.
Non-contiguous `Eigen::Ref` inputs are packed or copied by the implementation
where supported; new operations must not silently assume a tighter stride than
their validation permits.

### Step 1

| Value | Logical shape |
| --- | --- |
| Level 0 genotype block | variants x samples |
| Level 1 design | samples x features |
| phenotype/outcome matrix | samples x outcomes |
| Gram matrix | variants x variants, or features x features |
| crossproduct/right-hand sides | variants x outcomes, or features x outcomes |
| ridge predictions | samples x combinations |
| ridge coefficients/solutions | variants/features x combinations |
| grouped predictions | samples x groups |

`samples_in_columns` disambiguates prediction matrices: `true` means
features/variants x samples; `false` means samples x features.

The usual ridge output is parameter-major. For ridge parameter `p` and outcome
`o`, the output column is:

```
p * outcome_count + o
```

This applies to `ridge_predict`, `ridge_predict_factorized`,
`diagonal_penalty_predict`, the cached fold predictions, and their coefficient
or solution matrices. If cached fold systems were built with an
`active_phenotypes` mask, only active outcomes are downloaded, in original
outcome order, within each parameter group.

`ridge_predict_cached_preprocessed_systems_normalized` is the deliberate
exception. Its Level 1 design output is outcome-major:

```
o * parameter_count + p
```

For multiple outcomes, CUDA normalizes and reorders from the internal
parameter-major solve layout. Active-outcome compaction still preserves
original outcome order. Treat either ordering as part of the API contract and
cover any change with a conformance test.

### Step 2

| Value | Logical shape |
| --- | --- |
| dense genotype block | samples x variants |
| residuals, weights, observed masks | samples x phenotypes |
| per-phenotype design | samples x covariates |
| numerators and denominators | phenotypes x variants |
| optional observed allele sums/counts | phenotypes x variants |

Packed scoring receives one byte vector per variant. Each vector contains
`ceil(samples / 4)` PGEN two-bit hardcalls. The backend combines those vectors
into a variant-major transfer buffer, while all public score outputs remain
phenotype x variant.

## Step 1 resident-state lifetimes

The backend has several independent reusable states. Buffer capacity may be
retained after a state is invalidated; validity, not allocation, determines
whether a call is legal.

### Resident preprocessed genotypes

`preprocess_genotypes` and `preprocess_packed_hardcalls` may establish a
device-resident variants x samples block. A normal eigensystem sequence is:

```
preprocess_*
  -> compute_preprocessed_products
  -> factorize_ridge_system
  -> ridge_predict_preprocessed (one or more sample slices)
  -> release_preprocessed_genotypes
```

The Cholesky fast paths instead use:

```
preprocess_*
  -> [cache_preprocessed_fold_systems]
  -> ridge_predict_preprocessed_system[s]
     or ridge_predict_cached_preprocessed_systems
     or ridge_predict_cached_preprocessed_systems_normalized
  -> release_preprocessed_genotypes
```

Cached fold systems store training Gram matrices and right-hand sides in
per-fold CUDA lanes. `CudaResidentFoldSystems` keeps their dimensions, output
selection, validity, and genotype-versus-design orientation together. Fold
offsets/counts must match on reuse. A zero ridge parameter is not accepted by
the Cholesky specialization and causes a Boolean fallback to the eigensystem
path.

The resident genotype state is invalidated by another preprocessing call,
`release_preprocessed_genotypes`, or activation/caching of a resident design.
Invalidating it also invalidates genotype-oriented cached fold systems.

Packed host registrations are a separate resource. A successful
`register_packed_hardcall_buffer` remains in effect until
`release_packed_hardcall_buffers` or backend destruction. The caller must keep
the registered allocation and address stable for that interval.

### Resident Level 1 designs

A transient design is established by `cache_design_matrix` or
`cache_design_partitions`. Partitions must have equal feature counts and are
concatenated by rows in the order supplied:

```
cache_design_matrix/partitions
  -> predict_cached_design
     compute_cached_* products
     solve_cached_weighted_design
     grouped_predict_cached_design_partitions
     [cache_resident_design_fold_systems
        -> ridge_predict_cached_preprocessed_systems]
  -> release_cached_design
```

The persistent Level 1 cache supports construction a few columns at a time:

```
initialize_level1_design_cache(rows, columns)
  -> append_level1_design_cache(start=0, ...)
  -> append_level1_design_cache(start=previous_end, ...)
  -> activate_level1_design_cache(rows, columns)
  -> resident-design operations
  -> release_level1_design_cache
```

Appends must be contiguous, ordered, dimensionally exact, and collectively fill
the declared column count before activation. Activation aliases the persistent
allocation as the current resident design. `release_cached_design` removes that
active alias but does not free the persistent allocation;
`release_level1_design_cache` frees it and invalidates the alias when present.

Caching or activating a design invalidates the resident genotype state.
Replacing/releasing a design invalidates design-oriented fold systems and the
cached weighted Gram described below.

### Cached weighted Gram

`compute_cached_weighted_design_products` and
`solve_cached_weighted_design` both compute and retain the most recent
`X'WX` for the current resident design and weight vector.
`solve_cached_weighted_gram` may then solve new right-hand sides against that
exact Gram without rebuilding it:

```
cache design
  -> compute_cached_weighted_design_products
     or solve_cached_weighted_design
  -> solve_cached_weighted_gram (zero or more calls)
```

The cached Gram is replaced by the next cached weighted product/solve and
invalidated when the resident design changes or is released. The API does not
carry a weight identity, so correctness depends on the caller keeping the
right-hand side consistent with the retained Gram.

### Reusable factorizations

Two call-order contracts also exist without a resident design:

- `factorize_ridge_system` (or
  `compute_products_and_factorize_ridge`) precedes
  `ridge_predict_factorized`.
- `factorize_diagonal_penalty` precedes `solve_factorized` and
  `grouped_leave_one_out_predict_factorized`.

A later factorization of the same kind replaces the earlier state. Reuse calls
validate the system dimension and throw if factorization has not occurred.

## Step 2 prepared-state lifetime

The valid sequence is:

```
prepare_quantitative / prepare_binary / prepare_cox
  -> check ready()
  -> score_packed_block or score_dense_block (repeat for chromosome blocks)
  -> clear, or the next prepare_* call
```

Only one score mode is active at a time. Preparation owns copies of the
residuals, weights, designs, projections, masks, and small matrices required by
that mode. `clear()` makes the backend unready but may retain allocation
capacity for reuse.

The CPU backend implements dense block scoring and deliberately declines some
small quantitative panels for which the ordinary per-variant path is cheaper.
The CUDA backend implements packed-hardcall scoring, declines unsupported
shapes and workflows, and under `auto` declines quantitative scoring on
pre-Ampere devices. For phenotype-specific missingness it can also return
observed allele sums and nonmissing counts, allowing `Data` to avoid dense
genotype expansion.

The current application only prepares a Step 2 backend for additive,
single-variant PGEN hardcall analyses without dosage, interactions, masks,
sets, multi-phenotype tests, or other incompatible processing. Exact Cox stays
on CPU. Firth and SPA corrections occur outside the backend.

CUDA device selection is thread-local. Any backend method that can execute on a
pipeline worker must select its device on that thread. Backend instances are
not otherwise a general concurrent API; Step 1's two-block pipeline uses a
separate backend instance for overlapping work.

## Feature flags

These environment variables are diagnostic and tuning controls, not stable
user-facing algorithm choices. Boolean flags accept only `0` or `1`; invalid
values throw.

CUDA code is compiled only with `-DREGENIE_WITH_CUDA=ON`;
`REGENIE_CUDA_ARCHITECTURES` selects the generated device targets. At runtime,
`--gpu-device` selects a visible device, while `--step1-profile` and
`--step2-profile` expose backend placement and phase timings.

### CUDA and Step 1

| Variable | Default | Effect |
| --- | --- | --- |
| `REGENIE_CUDA_CHUNK_MB` | approximately 1,000 MB | Positive per-buffer streaming limit |
| `REGENIE_CUDA_RESIDENT_MB` | min(6,000 MB, 60% free memory) | Maximum resident genotype block; `0` disables |
| `REGENIE_CUDA_LEVEL1_RESIDENT_MB` | min(16,000 MB, 40% free memory) | Maximum size of one resident design copy; `0` disables |
| `REGENIE_CUDA_PINNED_STAGING_MB` | 64 MB | Size of each of two upload staging buffers; `0` disables |
| `REGENIE_CUDA_LEVEL0_CHOLESKY` | `1` | Enable nonzero-ridge Level 0 Cholesky path |
| `REGENIE_CUDA_LEVEL0_FOLD_BATCH` | `1` | Submit independent fold solves to reusable streams |
| `REGENIE_CUDA_LEVEL0_RESIDENT_FOLDS` | `1` | Cache fold products in CUDA lanes |
| `REGENIE_CUDA_REGISTER_PACKED` | `1` | Register reusable packed PGEN host buffers when possible |
| `REGENIE_CUDA_DIRECT_GROUPED_UPLOAD` | `1` | Copy grouped row slices without a full host materialization |
| `REGENIE_STEP1_PGEN_PACKED` | `1` | Enable direct packed-hardcall preprocessing when eligible |
| `REGENIE_STEP1_PGEN_PREFETCH_MB` | 4,096 MB | Maximum extra buffer for next-block PGEN prefetch; `0` disables |
| `REGENIE_STEP1_PGEN_TILE_VARIANTS` | 8 | PGEN materialization tile width, from 1 through 64 |
| `REGENIE_STEP1_LEVEL0_PIPELINE` | automatic | Force the two-backend packed Level 0 pipeline off/on; automatic use also requires an expanded block no larger than 1,000 MB |
| `REGENIE_STEP1_LEVEL1_PATH_NEWTON` | `0` | Opt into logistic Level 1 path continuation |

Host-side pipeline controls include `REGENIE_STEP1_BULK_L0_READ` (default
`1`), `REGENIE_STEP1_LEVEL1_L0_PREFETCH` (default `1`),
`REGENIE_STEP1_LEVEL1_L0_READ_THREADS` (default min(4, configured threads)),
`REGENIE_STEP1_LEVEL0_ASYNC_WRITE_MB` (default 1,024 MB),
`REGENIE_STEP1_OUTPUT_THREADS` (default configured threads), and
`REGENIE_STEP1_PREDICTION_CACHE` (default is eligibility-based; `0` disables).

### Step 2

`REGENIE_STEP2_QT_BLOCK_MIN_PHENOTYPES` overrides the CPU quantitative
block-scoring crossover with a positive phenotype count. Without the override,
dispatch depends on phenotype count, sample count, and whether phenotype masks
are complete.

## Testing entry points

For a normal test build:

```
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --target step1_compute_test step2_compute_test
ctest --test-dir build -R 'step[12]_compute' --output-on-failure
```

`test/test_step1_compute.cpp` is the CPU/CUDA numerical oracle, layout,
empty-shape, invalid-state, timing, and optional-fast-path suite. It can be run
directly with `--backend cpu`, `--backend auto`, or, in a CUDA build,
`--backend cuda --device N`; it also has benchmark modes.

`test/test_step2_compute.cpp` currently exercises the CPU quantitative, binary,
and approximate-Cox block scorer and dispatch thresholds. CUDA translation
units are compilation-checked in the CUDA CI build, while GPU behavior must be
covered by focused or end-to-end GPU validation.

On an NVIDIA development host, `scripts/test_step1_cuda.sh` builds both
backends, runs Step 1 conformance and compute-sanitizer when available,
benchmarks major paths, runs end-to-end Step 1 trait modes, and compares CPU
and CUDA LOCO output. It requires `BGEN_PATH`; its `CUDA_*` and `STEP1_*`
variables are harness aliases for the runtime controls described above.

The full application regression entry point is `test/test_bash.sh`. Use
`--step1-profile` and `--step2-profile` when validating placement, transfer,
cache-reuse, and fallback behavior; numerical agreement remains the acceptance
criterion.

## Extension guidelines

When adding or changing a backend operation:

1. Put statistical branching in the caller and numerical work in the backend.
   Preserve an executable CPU path for every optional specialization.
2. Decide whether the operation is mandatory or optional. Mandatory operations
   are pure virtual or have a complete base implementation. Optional
   operations return `false` without leaving outputs partially authoritative.
3. Treat `false` as capability/ineligibility, `std::invalid_argument` as a
   caller contract violation, and `std::runtime_error` as an execution or
   library failure.
4. Document the logical shapes, strides accepted, output-column order,
   prerequisite state, and every event that invalidates that state.
5. Keep release methods idempotent. If device allocations alias multiple
   logical states, make the ownership and invalidation relationship explicit.
6. Validate dimensions, finite values, nonnegative weights/penalties, integer
   limits, and empty shapes before launching work.
7. Accumulate timings into the supplied structure; never reset a caller-owned
   timing object.
8. Add CPU-oracle conformance coverage first. Then test CUDA agreement,
   unsupported-path fallback, invalid call order, empty inputs, active-outcome
   compaction, and any new state transition.
9. Benchmark both the fast path and its fallback with representative sizes.
   A faster kernel is not sufficient if transfers, host materialization, or
   downstream corrections dominate wall time.
