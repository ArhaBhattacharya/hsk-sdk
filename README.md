# hsk-sdk

Streaming, mergeable, deletable low-rank sketches (CountSketch + SRHT, after Halko–Martinsson–Tropp).  
A native Eigen core ships two ways, with no server involved:

- **Python wheel** `hsk-sdk` (`import hsk`)
- **DuckDB extension** `hsk` (SQL aggregate, merge, and SVD functions)

Both read and write the same binary sketch format (`HSK1`), so a sketch built in SQL can be finished in Python and vice versa.

## What a sketch is

A sketch of an $m \times n$ matrix $A$ holds $Y^\top = (A\Omega)^\top$ ($l \times m$) and $W = \Psi A$ ($s \times n$). Every update is linear, which gives three properties:

- **One pass.** Feed `(row, col, value)` entries in any order. The matrix is never held in memory.
- **Turnstile.** A negative value subtracts, so deleting entries is exact up to floating-point rounding.
- **Mergeable.** Sketches with equal `(m, n, rank, oversample, core_factor, c_factor, seed)` add together. Use this to shard work across threads, files, or tables.

`finalize(k)` turns a sketch into approximate top-$k$ factors $U$ ($m \times k$), $S$ ($k$), and $V$ ($n \times k$).

## Repository layout

```text
native/hybrid_sketch.hpp            header-only C++17 engine (needs Eigen >= 3.3)
native/bindings.cpp                 pybind11 module hsk_native (zero-copy mmap state, HSK1 buffers)
native/hsk_duckdb_extension.cpp     DuckDB extension
native/hsk_extension.hpp            class declaration for DuckDB's static loader
service/engine.py                   Python API (installed as hsk.engine); NumPy fallback backend
docs/installation.md                platform details, DuckDB setup, source build & troubleshooting
tests/test_core_priorities.py       regression suite (CSR ingestion, atomic validation, rank clamping, blobs)
ROADMAP.md                          shipped v0.1.1 items, v0.1.2/v0.2.0 roadmap & non-goals
CMakeLists.txt                      SDK, Python module and DuckDB targets
pyproject.toml                      scikit-build-core wheel build
.github/workflows/release.yml       CI: wheels (Linux/Windows/macOS) and DuckDB extension
hsk_showcase.ipynb                  demo notebook (streaming, merge, delete, memmap, rolling anomalies)
Matrix_Sketching_foundations.ipynb  NumPy/SciPy prototype of HMT & CountSketch (algorithmic foundation)
```

## Install

See the full **[Installation Guide](docs/installation.md)** for platform details, checksums, DuckDB setup, and source builds.

Prebuilt wheels (CPython 3.9–3.13; all 15 CI-tested, and live-verified on Windows `cp312`, Linux `cp313`, and macOS Apple Silicon `cp312`) and the DuckDB v1.1.2 extensions (Windows, Linux, macOS; all 3 CI-tested and live-verified on real hardware) are attached to the [v0.1.1 GitHub Release](https://github.com/ArhaBhattacharya/hsk-sdk/releases/tag/v0.1.1) (not yet published to PyPI):

```bash
# Example: Windows 64-bit, Python 3.12
python -m pip install https://github.com/ArhaBhattacharya/hsk-sdk/releases/download/v0.1.1/hsk_sdk-0.1.1-cp312-cp312-win_amd64.whl

# Example: Linux x86_64, Python 3.12 (wheel tag glibc >= 2.27; modern SciPy may require >= 2.28)
python -m pip install https://github.com/ArhaBhattacharya/hsk-sdk/releases/download/v0.1.1/hsk_sdk-0.1.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

Prebuilt wheel users do not need Eigen or a C++ compiler installed. The C++ headers require **Eigen $\ge 3.3$** (`3.4.0` is used in CI and CMake `FetchContent`). Only `pyproject.toml` Python source/wheel builds require Eigen to already be available because `pyproject.toml` sets `HSK_FETCH_EIGEN=OFF`; standalone CMake SDK and DuckDB builds default to `HSK_FETCH_EIGEN=ON`.

`hsk.BACKEND` is `"native"` or `"numpy"` (force one with the `HSK_BACKEND` environment variable). The NumPy fallback shares the sketching API but uses a different RNG, so it **cannot read, write, or inspect `HSK1` blobs** (`to_bytes`, `from_bytes`, `write_into`, `load_bytes`, and `merge_bytes` raise `NotImplementedError`; `inspect_bytes` is `native`-only) and cannot be merged with native sketches.

## Python quickstart

```python
import numpy as np
import hsk

rng = np.random.default_rng(0)
rows = rng.integers(0, 10_000, size=1_000, dtype=np.int64)
cols = rng.integers(0, 10_000, size=1_000, dtype=np.int64)
vals = rng.standard_normal(1_000)

sk = hsk.StreamingSketch(10_000, 10_000, rank=16, seed=42)
sk.update_coo(rows, cols, vals)                      # zero-copy for int64 rows/cols, float64 vals
sk.update_coo(rows[:100], cols[:100], -vals[:100])   # delete 100 entries
U, S, V = sk.finalize(16)                            # U: m x 16, S: (16,), V: n x 16

blob = sk.to_bytes()                                 # HSK1 blob (usable as a DuckDB BLOB)
sk2 = hsk.StreamingSketch.from_bytes(blob)
sk.merge(sk2)                                        # params must match
```

Disk-backed state, updated in place with no serialization:

```python
import os
import numpy as np
import hsk

os.makedirs("state", exist_ok=True)
rng = np.random.default_rng(0)
rows = rng.integers(0, 10_000, size=1_000, dtype=np.int64)
cols = rng.integers(0, 10_000, size=1_000, dtype=np.int64)
vals = rng.standard_normal(1_000)

with hsk.StreamingSketch.create_memmap("state/a", 10_000, 10_000, 16, seed=42, overwrite=True) as sk:
    sk.update_coo(rows, cols, vals)                  # writes straight into the memmap files
sk = hsk.StreamingSketch.open_memmap("state/a")      # reopen later ('r+' or 'c')
```

Sliding window with an anomaly score:

```python
import numpy as np
import hsk

rng = np.random.default_rng(0)
m, n = 1_000, 1_000
rs = hsk.RollingFactorStream(m, n, rank=8, window=50)   # window counts pushes (ticks)
for _ in range(10):
    r = rng.integers(0, m, size=200, dtype=np.int64)
    c = rng.integers(0, n, size=200, dtype=np.int64)
    v = rng.standard_normal(200)
    score = rs.push(r, c, v)        # expired ticks are retracted with -val
U, S, V = rs.factors(8)
last_score = rs.anomaly_score()     # fraction of the last batch outside the window's top-8 subspace
```

## Python API

```python
hsk.StreamingSketch(m, n, rank, oversample=10, core_factor=4.0, c_factor=8.0, seed=0xC0FFEE, *, yt=None, w=None)
  .update_coo(I, J, V)  .update_csr(A, scale=1.0)  .update_dense(D)  .merge(other)  .clear()  .finalize(k)
  .yt (l x m)  .w (s x n)  .set_state(yt, w)  .p  .memory_bytes()
  .to_bytes()  .write_into(buf)  .load_bytes(data)  .merge_bytes(data)  .blob_size
  StreamingSketch.from_bytes(data)  StreamingSketch.inspect_bytes(data)      # HSK1 blobs: native backend only
  StreamingSketch.create_memmap(path, m, n, rank, ..., *, overwrite=False)
  StreamingSketch.open_memmap(path, mode="r+")   .flush()  .close()

hsk.RollingFactorStream(m, n, rank, window, *, oversample=10, core_factor=4.0, c_factor=8.0, seed=0xC0FFEE,
                        score_rank=None, refresh_every=8, rebuild_every=0, sketch=None)
  .push(rows, cols, vals)  .factors(k=None)  .singular_values(k=None)  .anomaly_score(batch=None)  .rebuild()
  .ticks_in_window  .total_ticks

hsk.sketch_to_bytes(sk)   hsk.sketch_from_bytes(data)   hsk.estimate_sketch_bytes(m, n, rank, ..., *, blob=False)

# Multi-pass randomized SVD (in-memory dense/sparse matrices) and spectral estimators
hsk.rsvd_certified(A, rank, oversample=10, power_iters=1, c_factor=8.0, seed=42)
  # -> (U, S, V, certified_err)          # 3 + 2*power_iters passes (5 by default)
hsk.adaptive_rsvd(A, rel_tol, max_rank, oversample=10, power_iters=1, seed=42, rigorous=False)
  # -> (U, S, V, bound, est, norm)       # doubles rank from 8 up to max_rank until error <= rel_tol * norm
hsk.spectral_norm_estimate(A, iters=12, seed=3)             # -> scalar ||A||_2 estimate
hsk.residual_norm_estimate(A, Q, r=10, seed=7)              # -> HMT probabilistic upper bound on ||(I - Q Q^T) A||_2
hsk.residual_spectral_estimate(A, Q, iters=20, seed=11)     # -> tight power-iteration estimate of ||(I - Q Q^T) A||_2
hsk.HybridSketch(n, l, c_factor=8.0, seed=1)                # standalone n x l test matrix (.apply(A), .rows(js), .dense())
```

Notes:

- **`finalize(k)` returns $V$ ($n \times k$), not $V^\top$.** NumPy's `svd` and SciPy's `svds` return $V^\top$ ($k \times n$), so transpose accordingly when reconstructing `U @ np.diag(S) @ V.T`.
- Python's `finalize` clamps `k` to `min(rank, l)`.
- **Input ingestion contracts:**
  - `update_coo(I, J, V)`: pass 1-D integer arrays `(rows, cols)` and float `vals` positionally (or with keywords `I`, `J`, `V`; floats in `I`/`J` raise `TypeError`; negative entries subtract via `-V`). On the `native` backend, C-contiguous `int64` indices + `float64` values are zero-copy; `int32` indices are cast/copied to `int64`.
  - `update_csr(A, scale=1.0)`: requires a SciPy sparse matrix or array (non-CSR formats are converted via `.tocsr()`; raw `(indptr, indices, data)` tuples are not accepted on `StreamingSketch`). Zero-copy on `native` when `indptr` and `indices` have matching C-contiguous `int32` or `int64` dtype and `data` is C-contiguous `float64` (`float32` `data` is upcast to `float64`).
  - `update_dense(D)`: requires exact `(m, n)` shape (no sub-block or offset ingestion yet). On the `native` backend, Fortran-ordered `float64` is zero-copy, while other layouts and numeric dtypes (`C`-order, `float32`, integers) are copied/converted to `float64` automatically.
- **State, blobs & memmap:**
  - `.p` holds the sketch parameters and derived dimensions `(l, s)`. `yt` / `w` (and `set_state(yt, w)`) use `(l, m)` and `(s, n)` Fortran-ordered `float64` arrays.
  - `write_into(buf)` serializes into a writable buffer (`bytearray` or `uint8` `ndarray`) of length $\ge$ `blob_size` and returns bytes written; `load_bytes` replaces and `merge_bytes` adds state from an `HSK1` buffer; `inspect_bytes(data)` returns header metadata as a `dict` without verifying the payload checksum.
  - `open_memmap` accepts `mode="r+"` (persist updates) or `mode="c"` (copy-on-write); passing `"r"` raises `ValueError`. `close()` flushes changes to disk; on Windows, release references (`del sk`) before overwriting or deleting the memmap files in the same process.
- **`RollingFactorStream` options:** `window` counts `push()` calls (ticks) and retracts expired ticks with `-vals`. `score_rank` sets the rank used for scoring (defaults to `rank`); `refresh_every` controls how many ticks pass before refreshing the cached scoring basis (`1` = exact per-tick refresh); `sketch` accepts a pre-existing or memmap `StreamingSketch`. `anomaly_score(batch=None)` returns the last push's residual energy fraction in `[0, 1]` (or scores `batch=(rows, cols, vals)` immediately against the current window), returning `NaN` when the window or batch is empty.
- **`rsvd_certified` and `adaptive_rsvd` contracts:**
  - Both require an in-memory dense `ndarray` or SciPy sparse matrix (`LinearOperator` and `(matvec, rmatvec)` callables are not yet supported). `rsvd_certified` makes `3 + 2 * power_iters` passes over `A`/`A.T` (**5 passes** by default with `power_iters=1`). `adaptive_rsvd` estimates `||A||_2` (`norm`) and doubles rank from `8` up to `max_rank` until `est <= rel_tol * norm` (or `bound <= rel_tol * norm` when `rigorous=True`), taking tens of matvec passes per rank trial (prefer `rsvd_certified` when `rank` is known).
  - `certified_err` (`bound` in `adaptive_rsvd`) is an **absolute** spectral-norm ($\ell_2$) probabilistic upper bound on $\lVert A - U\Sigma V^\top \rVert_2$ (HMT eq. 4.3, holding with probability $\ge 1 - 10^{-10}$; conservative on noisy matrices), while `est` is a tight power-iteration **absolute** estimate of $\lVert (I - UU^\top)A \rVert_2$ (neither is a relative error).
- **Error guarantees:** Bad input raises before any entry is applied (`ValueError`, `IndexError`); malformed or wrong-sized `HSK1` headers raise `ValueError`, and a payload checksum mismatch raises `RuntimeError`.

## DuckDB extension

Compiled against **DuckDB v1.1.2** (C++ extension API). Prebuilt binaries for Windows, Linux (`linux_amd64_gcc4`, matching official Linux `duckdb==1.1.2` Python wheels and CLI), and macOS are attached to the [v0.1.1 GitHub Release](https://github.com/ArhaBhattacharya/hsk-sdk/releases/tag/v0.1.1). Rename the downloaded release binary to `hsk.duckdb_extension` and load it with unsigned extensions enabled (`duckdb -unsigned` or `allow_unsigned_extensions=true`):

```sql
LOAD '/path/to/hsk.duckdb_extension';

-- Aggregate: m, n, rank, seed must be constants (cast seed to UBIGINT)
SELECT hsk_sketch_agg(row_idx, col_idx, val, 10000, 10000, 16, 42::UBIGINT) AS sk FROM entries;

-- Merge two sketches (parameters must match)
SELECT hsk_sketch_merge(a.sk, b.sk) FROM shard_a a, shard_b b;

-- Factors: use LATERAL
SELECT s.component_idx, s.singular_value, s.u_vec, s.v_vec
FROM sketches t, LATERAL hsk_sketch_svd(t.sk, 5) s;
```

- Rows with a `NULL` argument are skipped. `NaN`/`Inf` values and out-of-range indices (`row_idx` outside `[0, m)` or `col_idx` outside `[0, n)`) raise errors.
- `row_idx` and `col_idx` are `0`-based `BIGINT` indices; `hsk_sketch_svd` outputs `0`-based `component_idx`, descending `singular_value`, and `DOUBLE[]` lists `u_vec` (length `m`) and `v_vec` (length `n`).
- A negative `val` subtracts; duplicate `(row_idx, col_idx)` pairs sum; an aggregate over zero rows returns `NULL`.
- Defaults match Python and are currently hardcoded in SQL: `oversample = 10`, `core_factor = 4.0`, CountSketch width factor `c_factor = 8.0`.
- **Memory:** Each aggregate state (per thread, per group) holds a full sketch. The per-sketch cap is `1 GiB` (`HSK_MAX_SKETCH_BYTES`, hard limit `3500 MiB`). Prefer fewer groups with larger sketches over many groups.
- `hsk_sketch_svd(sk, k)` clamps `k` to `min(rank, l)` (matching Python's `finalize(k)`). See the [Installation Guide](docs/installation.md#duckdb-extension) for a complete Python (`duckdb==1.1.2`) and CLI example.

## Memory

- **In-RAM sketch state (`memory_bytes()` / `estimate_sketch_bytes(..., blob=False)`):** $8 \times (l \cdot m + s \cdot n)$ bytes, where `l = min(rank + oversample, m, n)` and `s = min(m, max(l, ceil(core_factor * l)))`. Memory usage is independent of the number of streamed entries. For example, $m = n = 10{,}000$ with `rank = 8` gives $l = 18$, $s = 72$, and **`6.87 MiB`** (`7,200,000` bytes) of state (plus up to **`8 MiB`** (`8,388,608` bytes) per cached test matrix for lookup tables).
- **Serialized `HSK1` blobs (`blob_size` / `estimate_sketch_bytes(..., blob=True)`):** $8 \times (l \cdot m + s \cdot n) + 88$ bytes (includes the 88-byte `HSK1` header and payload checksum).
- **Memmap-backed sketches (`create_memmap`):** store raw Fortran-ordered `float64` `.yt.f64` ($8 \cdot l \cdot m$ bytes) and `.w.f64` ($8 \cdot s \cdot n$ bytes) arrays plus a small `.hsk.json` metadata file, with no 88-byte `HSK1` header.

## Accuracy & benchmarks

`StreamingSketch` is a **single-pass** sketch with no power iterations, so accuracy depends on how much spectral energy sits outside the top-$k$ subspace:

- Works well when the matrix has clear low-rank structure and a small tail.
- Does **not** recover the top singular values of noise-dominated or flat-spectrum matrices.
- When multiple passes over an in-memory matrix are affordable, use `rsvd_certified` (5 passes at default `power_iters=1`) or `adaptive_rsvd`.

### Measured results

Single runs on a laptop with synthetic data (random indices for throughput rows; planted community structure for accuracy rows). Indicative, not a controlled benchmark study (exact sketch state sizes reported in binary `MiB` = $2^{20}$ bytes):

| Benchmark | Result |
|---|---|
| Ingestion, $10{,}000 \times 10{,}000$, `rank = 16`, 1M entries | ~0.1–0.14 s (7–9 M entries/s), `9.92 MiB` state |
| Streaming, $100{,}000 \times 100{,}000$, `rank = 16`, 350M entries | ~1.5–2.5 min (2.5–4.5 M entries/s), `99.18 MiB` sketch, ~150–215 MiB flat RSS |
| Throughput on demo stream | ~9–11 M entries/s |
| Planted structure, $5{,}000 \times 5{,}000$ (6 communities + noise): top singular value | rel. error `6e-4` |
| Same, worst of top 6 singular values | rel. error `3.9e-3` |
| Same, rank-6 reconstruction error | `3.28e-2` (`hsk`) vs `3.18e-2` (exact `svds`) |
| Same, worst subspace angle ($\sin \theta$) | `5.9e-2` |
| 4-shard merge, blob merge, delete-vs-rebuild | agrees with single pass to ~`1e-15` |
| Process RSS growth while streaming 20M entries | +24 MB (measured OS RSS delta) |

## Known limits & unverified areas

- **Unmeasured regimes:** Clustered or slowly decaying spectra (e.g., $100, 95, 92, \dots$), and `rsvd_certified` on noise-dominated matrices.
- **Coherent / hub-dominated matrices at larger rank:** Single-pass CountSketch + SRHT (`HSK1`) can suffer bucket collisions on extremely coherent or hub-dominated matrices at larger ranks (`rank >= 32`). Raising `c_factor` widens intermediate buckets without growing `Yt`/`W` state RAM, but breaks Python $\leftrightarrow$ DuckDB merge compatibility unless both sides match (DuckDB hardcodes `c_factor = 8.0`).
- **Floating-point non-determinism across merges:** Merged blobs are not byte-identical because summation order and floating-point contraction change last-bit rounding. Compare factors with tolerances rather than raw blob equality.
- **Sliding-window drift:** `RollingFactorStream` retracts expired ticks with `-val`, accumulating small floating-point residue over long runs. `rebuild_every` is off (`0`) by default—set it (e.g., every few hundred ticks) or call `rebuild()` periodically to recompute from the retained window in $O(\text{nnz in window})$ time.
- **Memmap recovery:** `RollingFactorStream` keeps its tick history in memory only; reopening a memmap sketch restores the sketch state but not the sliding window. Memmap writes go through the OS page cache and `flush()`, so take periodic `to_bytes()` snapshots if crash recovery is required.
- **Thread safety:** Sketches are not thread-safe for concurrent mutation.
- **RNG & platform portability:** Native sketches derive hashes, signs, and sampled Hadamard rows from `SplitMix64`, making `HSK1` blobs **designed to be reproducible** across compilers and little-endian OSes; however, **cross-OS blob exchange is not yet CI-verified**. The NumPy fallback uses `np.random.default_rng` and is incompatible. The certified error bound in `rsvd_certified` uses `std::mt19937_64` and `std::normal_distribution`, which may vary slightly across standard libraries.
- **Platform verification:** All 15 Python wheels (`cp39`–`cp313`) and all 3 DuckDB `v1.1.2` extensions (Windows x64, Linux `x86_64`, macOS `arm64`) are CI-tested via `.github/workflows/release.yml` and live-verified on hardware; see the [Release & Platform Support Matrix](docs/installation.md#release--platform-support-matrix) for details.

## License

Copyright (c) 2026 Arha Bhattacharya. All rights reserved.  
This is an early-stage research project. I'm happy to share it for academic use—please get in touch at arhabhattacharya@gmail.com.
