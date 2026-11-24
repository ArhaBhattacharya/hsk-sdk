# hsk-sdk Roadmap

Where the project is heading, in priority order, with what is deliberately not being built and why.

## Principles

- **Sketches stay compatible.** A change that would make existing saved sketches unreadable or unmergeable needs a format version bump, and is avoided otherwise.
- **Exact merge and exact delete are kept.** Any feature that would weaken them is rejected.
- **Measure before claiming.** No speed claim is made until it has been measured on the same data against the current code.
- **Known limit, stated up front:** the single-pass estimate is inflated on noise-dominated or flat-spectrum matrices. For those, a second pass (`rsvd_certified`) is the intended route, and it still needs to be benchmarked there.

## Shipped in v0.1.1

### 1. CSR ingestion: `update_csr(A, scale=1.0)`
- Accept a SciPy CSR matrix directly (int32 or int64 indices, no copy), with `scale=-1.0` to delete a whole block exactly.
- All input is validated before anything is applied: index ranges, pointer consistency, finite values. A bad input leaves the sketch unchanged.
- Rows are processed sequentially, which is the natural SciPy and Arrow layout and avoids building COO row arrays.
- **Status:** shipped in `v0.1.1` (zero-copy `int32`/`int64` CSR ingestion with atomic pre-validation, row-hoisting, `W` prefetching, and NumPy fallback parity).
- **Expectation:** a partial gain. Measure on large rank (for example 64) against `update_coo` before claiming a speedup.

### 2. Prefetching in the Python ingestion path
- **Status:** shipped in `v0.1.1` (`HSK_PREFETCH_W` wired for MSVC x86/x64 via `_mm_prefetch`, and applied in `Sketch::update_coo` and `Sketch::update_csr` in `native/bindings.cpp`).

### 3. DuckDB: respect `rank` in `hsk_sketch_svd`
- **Status:** shipped in `v0.1.1` (`StreamingLowRank::finalize` now clamps `k` to `min(rank, l)` matching Python's `finalize(k)`, and the "known gap" note has been removed from `README.md`).

## Later

### 4. DuckDB 2.0.0 support
- DuckDB 2.0.0 LTS was announced on 2026-10-21. The required API changes have not yet been verified against the official migration notes.
- Plan: keep v1.1.2 as the shipping target and add 2.0 as a separate build once the notes have been read.

### 5. Default oversampling
- Oversampling is already a user parameter. Only the default is open: on a planted-structure test, oversample 24 instead of 10 reduced the worst singular-value error from 1.8% to 0.23% (rank 8).
- A new default changes sketch parameters, so it must be introduced once, with a version note.

### 6. Multi-threaded ingestion
- The Python GIL is already released for larger batches. True parallel ingestion needs a partitioning scheme so workers never write the same state. Considered only if single-thread throughput becomes the bottleneck.

### 7. Dense-array input without copies
- Accept C-ordered NumPy arrays directly in `update_dense` and `rsvd_dense`. Low priority: sparse input is the main path.

### 8. Reduced-precision storage (float32)
- Would halve memory but weakens exact deletion and needs its own format version. Opt-in only, if at all.

## Considered and rejected

- **Changing the internal storage layout or SIMD alignment.** The updates already write contiguous memory; the remaining cost is random access, which a layout change does not fix. It would also break compatibility with saved sketches.
- **Replacing the hashing scheme.** It already avoids the costs this change targets, and changing it would invalidate every saved sketch.
- **A different fast-transform implementation.** The current one is already an in-place, O(s log s) transform.
- **Frequent Directions noise shrinkage.** It breaks linearity, so exact merge and exact delete would stop working. For noisy data the answer is a second pass, not shrinkage inside the stream.
