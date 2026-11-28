# hsk-sdk Roadmap

Where the project is heading, in priority order, with what is deliberately not being built and why.

## Principles

- **Sketches stay compatible.** A change that would make existing saved sketches unreadable or unmergeable needs a format version bump, and is avoided otherwise.
- **Exact merge and exact delete are kept.** Any feature that would weaken them is rejected.
- **Measure before claiming.** No speed claim is made until it has been measured on the same data against the current code.
- **Known limit, stated up front:** the single-pass estimate is inflated on noise-dominated or flat-spectrum matrices. For those, multi-pass randomized SVD (`rsvd_certified` / `adaptive_rsvd`) is the intended route, and it still needs to be benchmarked there.

## Shipped in v0.1.1

### 1. CSR ingestion: `update_csr(A, scale=1.0)`
- Accept a SciPy CSR matrix directly (`int32` or `int64` indices, zero-copy when dtypes match and arrays are C-contiguous), with `scale=-1.0` to delete a whole block exactly.
- All input is validated before anything is applied: index ranges, pointer consistency, finite values. A bad input leaves the sketch unchanged.
- Rows are processed sequentially, which is the natural SciPy and Arrow layout and avoids building COO row arrays.
- **Status:** shipped in `v0.1.1` (`_csr_arrays` + `Sketch::update_csr<int32_t|int64_t>` with atomic pre-validation, row-hoisting, `W` prefetching, and NumPy fallback parity).
- **Expectation:** a partial gain. Measure on large rank (for example 64) against `update_coo` before claiming a speedup.

### 2. Prefetching in the Python ingestion path
- **Status:** shipped in `v0.1.1` (`HSK_PREFETCH_W` wired for MSVC x86/x64 via `_mm_prefetch` and GCC/Clang via `__builtin_prefetch`, active in `Sketch::update_coo` and `Sketch::update_csr`).

### 3. DuckDB: respect `rank` in `hsk_sketch_svd`
- **Status:** shipped in `v0.1.1` (`StreamingLowRank::finalize` clamps `k` to `min(rank, l)` matching Python's `finalize(k)`, CI-tested up to `k=999`, and the known-gap note was removed from `README.md`).

### 4. Release packaging & cross-platform verification
- **Status:** shipped in `v0.1.1` (15 Python wheels for `cp39–cp313` across Windows, Linux `manylinux_2_28`, and macOS `arm64`, plus 3 DuckDB `v1.1.2` extensions including `linux_amd64_gcc4` with `EXTENSION_STATIC_BUILD=1`; all 18 artifacts CI-tested and live-verified on real Windows `cp312`, Linux `cp313`, and macOS Apple Silicon `cp312` hardware).

## Next: v0.1.2 (PyPI & License Patch)

### 5. OSI license & first PyPI release
- Add an explicit OSI-approved `LICENSE` file and PEP 639 license metadata in `pyproject.toml` (updating `sdist.include` and replacing the `"All rights reserved"` footer in `README.md`).
- Publish `hsk-sdk` wheels to PyPI via Trusted Publishing in `.github/workflows/release.yml` so `pip install hsk-sdk` works without manual GitHub Release URLs.

### 6. `adaptive_rsvd(..., rigorous=True)` pass-count fix
- Fix Python `adaptive_rsvd` in `service/engine.py` so `rigorous=True` skips the redundant 40-pass `residual_spectral_estimate` call on each rank trial (matching C++ `hsk::adaptive_rsvd`).

## Next: v0.2.0 (Core Systems, DuckDB LTS & Multi-Pass Solvers)

### 7. Zero-copy dense, block & typed-index ingestion
- **Zero-copy C-contiguous `update_dense` & `rsvd_dense`:** remove the pybind11 `Eigen::Ref<const MatrixXd>` C-to-Fortran copy (~23.7 MiB on a `300 x 10367` `float64` matrix) and eliminate the full `Dt = D.transpose()` heap allocation inside `Sketch::update_dense`.
- **Dense block ingestion:** add `update_dense_block(D, row_offset=0, col_offset=0, scale=1.0)` so sub-blocks can be streamed without matching full `(m, n)` shape.
- **Native `float32` input ingestion:** ingest `float32` dense/CSR arrays directly into `float64` sketch accumulators without creating double `float64` copies in Python + pybind11 (`47.46 MiB` peak on C-order `float32` `300 x 10367`), while keeping internal sketch state `float64`.
- **Zero-copy `int32` `update_coo`:** add an `int32` index overload to `Sketch::update_coo` (matching `update_csr`) and remove the forced `int64` copy in `_as_int_array`.

### 8. Matrix-free `LinearOperator`, out-of-core solvers & `NamedTuple` returns
- **Matrix-free & iterable inputs:** support `scipy.sparse.linalg.LinearOperator` and `(matvec, rmatvec)` callables alongside chunked/re-readable stream iterators and Randomized Block Krylov (`method="krylov"`) in `rsvd_certified` and `adaptive_rsvd`, with Python/C++ `adaptive_rsvd` pass-count parity.
- **Arity-preserving `NamedTuple` returns:** return `CertifiedSVDResult` (4 fields), `AdaptiveSVDResult` (6 fields), and optional `SketchSVDResult` (3 fields) so positional unpacking stays 100% backwards-compatible while preventing field mislabeling.

### 9. DuckDB `2.0.x` LTS via C Extension API & SQL hyperparameters
- Upgrade from EOL DuckDB `v1.1.2` to **DuckDB `2.0.x` LTS** using the ABI-stable **C Extension API**, eliminating full DuckDB C++ source compilation in CI and ~21 MB statically-linked extension binaries (`EXTENSION_STATIC_BUILD=1`), and enabling signed/community extension distribution.
- Add optional SQL parameters/overloads on `hsk_sketch_agg` for `(oversample, core_factor, c_factor)` so SQL users are no longer locked to `(10, 4.0, 8.0)`.

### 10. Cross-OS `HSK1` CI round-trip & expanded Python/ARM64 matrix
- Add a cross-OS `HSK1` blob round-trip job in `.github/workflows/release.yml` verifying Linux $\leftrightarrow$ Windows $\leftrightarrow$ macOS and Python $\leftrightarrow$ DuckDB serialization, merging, and SVD parity.
- Add Python `3.14` and `3.15` wheels (dropping EOL Python `3.9`) and free GitHub ARM64 runners (`ubuntu-24.04-arm`, `windows-11-arm`).

## Later

### 11. Default oversampling
- Oversampling is already a user parameter. Only the default is open: on a planted-structure test, oversample 24 instead of 10 reduced the worst singular-value error from 1.8% to 0.23% (rank 8).
- A new default changes sketch parameters, so it must be introduced once, with a version note.

### 12. Multi-threaded ingestion
- The Python GIL is already released for larger batches. True parallel ingestion needs a partitioning scheme so workers never write the same state. Considered only if single-thread throughput becomes the bottleneck.

### 13. Reduced-precision storage (`float32` state)
- Storing `Yt` and `W` in `float32` would halve memory, but weakens exact deletion and needs its own format version (`HSK2`). Opt-in only, if at all—separate from zero-copy `float32` input ingestion in item 7.

## Considered and rejected

- **Changing the internal storage layout or SIMD alignment.** The updates already write contiguous memory; the remaining cost is random access, which a layout change does not fix. It would also break compatibility with saved sketches.
- **Replacing the hashing scheme.** It already avoids the costs this change targets, and changing it would invalidate every saved sketch.
- **A different fast-transform implementation.** The current one is already an in-place, $O(s \log s)$ transform.
- **Frequent Directions noise shrinkage.** It breaks linearity, so exact merge and exact delete would stop working. For noisy data the answer is multi-pass SVD, not shrinkage inside the stream.
