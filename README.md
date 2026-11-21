# hsk-sdk

Streaming, mergeable, deletable low-rank sketches (CountSketch + SRHT, after Halko–Martinsson–Tropp).  
A native Eigen core ships two ways, with no server involved:

- **Python wheel** `hsk-sdk` (`import hsk`)
- **DuckDB extension** `hsk` (SQL aggregate, merge, and SVD functions)

Both read and write the same binary sketch format (`HSK1`), so a sketch built in SQL can be finished in Python and vice versa.

## What a sketch is

A sketch of an $m \times n$ matrix $A$ holds $Y^\top = (A\Omega)^\top$ (l×m) and $W = \Psi A$ (s×n). Every update is linear, which gives three properties:

- **One pass.** Feed `(row, col, value)` entries in any order. The matrix is never held in memory.
- **Turnstile.** A negative value subtracts, so deleting entries is exact up to floating-point rounding.
- **Mergeable.** Sketches with equal `(m, n, rank, oversample, core_factor, c_factor, seed)` add together. Use this to shard work across threads, files, or tables.

`finalize(k)` turns a sketch into approximate top-k factors `U (m×k), S (k), V (n×k)`.

## Repository layout

```text
native/hybrid_sketch.hpp            header-only C++17 engine (needs Eigen >= 3.3)
native/bindings.cpp                 pybind11 module hsk_native (zero-copy mmap state, HSK1 buffers)
native/hsk_duckdb_extension.cpp     DuckDB extension
native/hsk_extension.hpp            class declaration for DuckDB's static loader
service/engine.py                   Python API (installed as hsk.engine); NumPy fallback backend
CMakeLists.txt                      SDK, Python module and DuckDB targets
pyproject.toml                      scikit-build-core wheel build
release.yml                         CI: wheels (Linux/Windows/macOS) and DuckDB extension
hsk_showcase.ipynb                  demo notebook (streaming, merge, delete, memmap, rolling anomalies)
Matrix_Sketching_foundations.ipynb  NumPy/SciPy prototype of HMT & CountSketch (algorithmic foundation)
```

## Install

See the full **[Installation Guide](docs/installation.md)** for platform details, checksums, DuckDB setup, and source builds.

Prebuilt wheels (CPython 3.9–3.13) and the DuckDB v1.1.2 extension are attached to the [v0.1.0 GitHub Release](https://github.com/ArhaBhattacharya/hsk-sdk/releases/tag/v0.1.0) (not yet published to PyPI):

```bash
# Example: Windows 64-bit, Python 3.12
python -m pip install https://github.com/ArhaBhattacharya/hsk-sdk/releases/download/v0.1.0/hsk_sdk-0.1.0-cp312-cp312-win_amd64.whl
```

Source builds require Eigen 3.4 to already be available because `HSK_FETCH_EIGEN=OFF`.

`hsk.BACKEND` is `"native"` or `"numpy"` (force one with the `HSK_BACKEND` environment variable). The NumPy fallback shares the same API but uses a different RNG, so it **cannot read or write `HSK1` blobs** and cannot be merged with native sketches.

## Python quickstart

```python
import numpy as np
import hsk

sk = hsk.StreamingSketch(10_000, 10_000, rank=16, seed=42)
sk.update_coo(rows, cols, vals)                      # int64 rows/cols, float64 vals
sk.update_coo(rows[:100], cols[:100], -vals[:100])   # delete 100 entries
U, S, V = sk.finalize(16)                            # U: m x 16, S: (16,), V: n x 16

blob = sk.to_bytes()                                 # HSK1 blob (usable as a DuckDB BLOB)
sk2 = hsk.StreamingSketch.from_bytes(blob)
sk.merge(sk2)                                        # params must match
```

Disk-backed state, updated in place with no serialization:

```python
with hsk.StreamingSketch.create_memmap("state/a", 10_000, 10_000, 16, seed=42) as sk:
    sk.update_coo(rows, cols, vals)                  # writes straight into the memmap files
sk = hsk.StreamingSketch.open_memmap("state/a")      # reopen later ('r+' or 'c')
```

Sliding window with an anomaly score:

```python
rs = hsk.RollingFactorStream(m, n, rank=8, window=50)   # window counts pushes (ticks)
score = rs.push(r, c, v)        # expired ticks are retracted with -val
U, S, V = rs.factors(8)
rs.anomaly_score()              # fraction of the last batch outside the window's top-8 subspace
```

## Python API

```python
hsk.StreamingSketch(m, n, rank, oversample=10, core_factor=4.0, c_factor=8.0, seed=0xC0FFEE, *, yt=None, w=None)
  .update_coo(I, J, V)  .update_dense(D)  .merge(other)  .clear()  .finalize(k)
  .yt (l x m)  .w (s x n)  .set_state(yt, w)  .p  .memory_bytes()
  .to_bytes()  .write_into(buf)  .load_bytes(data)  .merge_bytes(data)  .blob_size
  StreamingSketch.from_bytes(data)  StreamingSketch.inspect_bytes(data)      # blobs: native backend only
  StreamingSketch.create_memmap(path, m, n, rank, ..., *, overwrite=False)
  StreamingSketch.open_memmap(path, mode="r+")   .flush()  .close()

hsk.RollingFactorStream(m, n, rank, window, *, oversample=10, core_factor=4.0, c_factor=8.0, seed=0xC0FFEE,
                        score_rank=None, refresh_every=8, rebuild_every=0, sketch=None)
  .push(rows, cols, vals)  .factors(k=None)  .singular_values(k=None)  .anomaly_score(batch=None)  .rebuild()

hsk.sketch_to_bytes(sk)   hsk.sketch_from_bytes(data)   hsk.estimate_sketch_bytes(m, n, rank, ..., *, blob=False)

# Two-pass randomized SVD and HMT norm estimators
hsk.rsvd_certified(A, rank, oversample=10, power_iters=1, c_factor=8.0, seed=42)
  # -> (U, S, V, certified_err)
hsk.adaptive_rsvd(A, rel_tol, max_rank, oversample=10, power_iters=1, seed=42, rigorous=False)
  # -> (U, S, V, abs_err, rel_err, norm_est)
hsk.spectral_norm_estimate(A, iters=12, seed=3)
hsk.residual_norm_estimate(A, Q, r=10, seed=7)
hsk.residual_spectral_estimate(A, Q, iters=20, seed=11)
hsk.HybridSketch(n, l, c_factor=8.0, seed=1)     # NumPy port of the test matrix
```

Notes:

- **`finalize(k)` returns $V$ (n×k), not $V^\top$.** NumPy's `svd` and SciPy's `svds` return $V^\top$ ($k \times n$), so transpose accordingly when reconstructing `U @ np.diag(S) @ V.T`.
- Python's `finalize` clamps `k` to `min(rank, l)`.
- `rows` and `cols` must be integer arrays; floats raise `TypeError`.
- Bad input raises before any entry is applied (`ValueError`, `IndexError`), and a corrupt or truncated blob raises `ValueError`.
- `open_memmap` accepts `mode="r+"` (persist updates) or `mode="c"` (copy-on-write); passing `"r"` raises `ValueError`.

## DuckDB extension

Compiled against **DuckDB v1.1.2** (C++ extension API). Rename the downloaded release binary to `hsk.duckdb_extension` and load it with unsigned extensions enabled (`duckdb -unsigned` or `allow_unsigned_extensions=true`):

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

- Rows with a `NULL` argument are skipped. `NaN`/`Inf` values and out-of-range indices raise errors.
- A negative `val` subtracts; duplicate `(row_idx, col_idx)` pairs sum; an aggregate over zero rows returns `NULL`.
- Defaults match Python: `oversample = 10`, `core_factor = 4.0`, CountSketch width factor `c_factor = 8.0`.
- **Memory:** Each aggregate state (per thread, per group) holds a full sketch. The per-sketch cap is `1 GiB` (`HSK_MAX_SKETCH_BYTES`, hard limit `3500 MiB`). Prefer fewer groups with larger sketches over many groups.
- **Known gap:** `hsk_sketch_svd` can return components beyond `rank` (up to `rank + oversample`), which are mostly noise. The Python API clamps to `rank`, whereas the extension does not yet—request `k <= rank` until this is fixed.

## Memory

Sketch state is $8 \times (l \cdot m + s \cdot n)$ bytes plus an 88-byte header, where $l = \min(\text{rank} + \text{oversample}, m, n)$ and $s = \min(m, \max(l, \lceil \text{core_factor} \cdot l \rceil))$. Memory usage is independent of the number of streamed entries. For example, $m = n = 10{,}000$ with `rank = 8` gives $l = 18$, $s = 72$, and `7.2 MB` of state (shared sign tables add up to `8 MB` per test matrix).

## Accuracy & benchmarks

This is a **single-pass** sketch with no power iterations, so accuracy depends on how much spectral energy sits outside the top-k subspace:

- Works well when the matrix has clear low-rank structure and a small tail.
- Does **not** recover the top singular values of noise-dominated or flat-spectrum matrices.
- When a second pass over the data is affordable, use `rsvd_certified`.

### Measured results

Single runs on a laptop with synthetic data (random indices for throughput rows; planted community structure for accuracy rows). Indicative, not a controlled benchmark study:

| Benchmark | Result |
|---|---|
| Ingestion, $10{,}000 \times 10{,}000$, `rank = 16`, 1M entries | ~0.1–0.14 s (7–9 M entries/s), 10.4 MB state |
| Streaming, $100{,}000 \times 100{,}000$, `rank = 16`, 350M entries | ~1.5–2.5 min (2.5–4.5 M entries/s), 99 MiB sketch, ~150–215 MiB flat RSS |
| Throughput on demo stream | ~9–11 M entries/s |
| Planted structure, $5{,}000 \times 5{,}000$ (6 communities + noise): top singular value | rel. error `6e-4` |
| Same, worst of top 6 singular values | rel. error `3.9e-3` |
| Same, rank-6 reconstruction error | `3.28e-2` (`hsk`) vs `3.18e-2` (exact `svds`) |
| Same, worst subspace angle ($\sin \theta$) | `5.9e-2` |
| 4-shard merge, blob merge, delete-vs-rebuild | agrees with single pass to ~`1e-15` |
| Process RSS growth while streaming 20M entries | +24 MB |

## Known limits & unverified areas

- **Unmeasured regimes:** Clustered or slowly decaying spectra (e.g., $100, 95, 92, \dots$), and `rsvd_certified` on noise-dominated matrices.
- **Floating-point non-determinism across merges:** Merged blobs are not byte-identical because merge order and FMA contraction change last-bit rounding. Compare factors with tolerances rather than raw blob equality.
- **Sliding-window drift:** `RollingFactorStream` retracts expired ticks with `-val`, accumulating small floating-point residue over long runs. `rebuild_every` is off (`0`) by default—set it (e.g., every few hundred ticks) or call `rebuild()` periodically to recompute from the retained window in $O(\text{nnz in window})$ time.
- **Memmap recovery:** `RollingFactorStream` keeps its tick history in memory only; reopening a memmap sketch restores the sketch state but not the sliding window. Memmap writes go through the OS page cache and `flush()`, so take periodic `to_bytes()` snapshots if crash recovery is required.
- **Thread safety:** Sketches are not thread-safe for concurrent mutation.
- **RNG & platform portability:** Native sketches derive hashes, signs, and sampled Hadamard rows from `SplitMix64`, making `HSK1` blobs reproducible across compilers and OSes. The NumPy fallback uses `np.random.default_rng` and is incompatible. The certified error bound in `rsvd_certified` uses `std::mt19937_64` and `std::normal_distribution`, which may vary slightly across standard libraries.
- **Platform verification:** Windows x64 (Python 3.12), Linux x86_64 (Python 3.13), and the Windows DuckDB v1.1.2 extension have been runtime-tested; remaining Python wheels (CPython 3.9–3.13 across Windows, Linux, and macOS) and the Linux DuckDB extension are built by CI (`release.yml`) and awaiting runtime verification.

## License

Copyright (c) 2026 Arha Bhattacharya. All rights reserved.  
This is an early-stage research project. I'm happy to share it for academic use—please get in touch at arhabhattacharya@gmail.com.