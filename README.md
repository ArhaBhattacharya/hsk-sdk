# hsk-sdk

Streaming, mergeable, deletable low-rank sketches (CountSketch + SRHT, after Halko–Martinsson–Tropp).
A native Eigen core ships two ways, with no server involved:

- **Python wheel** `hsk-sdk` (`import hsk`)
- **DuckDB extension** `hsk` (SQL aggregate, merge and SVD functions)

Both read and write the same binary sketch format (HSK1), so a sketch built in SQL can be finished in Python and vice versa.

## What a sketch is

A sketch of an m×n matrix A holds `Yt = (AΩ)ᵀ` (l×m) and `W = ΨA` (s×n). Every update is linear, which gives three properties:

- **One pass.** Feed `(row, col, value)` entries in any order. The matrix is never held in memory.
- **Turnstile.** A negative value subtracts, so deleting entries is exact up to floating-point rounding.
- **Mergeable.** Sketches with equal `(m, n, rank, oversample, core_factor, c_factor, seed)` add together. Use this to shard work across threads, files or tables.

`finalize(k)` turns a sketch into approximate top-k factors `U (m×k), S (k), V (n×k)`.

## Repository layout

```
native/hybrid_sketch.hpp        header-only C++17 engine (needs Eigen >= 3.3)
native/bindings.cpp             pybind11 module hsk_native (zero-copy mmap state, HSK1 buffers)
native/hsk_duckdb_extension.cpp DuckDB extension
native/hsk_extension.hpp        class declaration for DuckDB's static loader
service/engine.py               Python API (installed as hsk.engine); NumPy fallback backend
CMakeLists.txt                  SDK, Python module and DuckDB targets
pyproject.toml                  scikit-build-core wheel build
release.yml                     CI: wheels (Linux/Windows/macOS) and DuckDB extension
hsk_showcase.ipynb              demo notebook (streaming, merge, delete, memmap, rolling anomalies)
```

## Install (Python)

See the [Python installation guide](docs/installation.md). In brief, download
the wheel artifact for your operating system from a successful
[Release workflow](https://github.com/ArhaBhattacharya/hsk-sdk/actions/workflows/release.yml)
run, extract it, and install it with:

    python -m pip install /path/to/hsk_sdk-<version>-<python>-<platform>.whl

Wheels are built for CPython 3.9–3.13. Source builds require Eigen 3.4 to
already be available because `HSK_FETCH_EIGEN=OFF`.

`hsk.BACKEND` is `"native"` or `"numpy"`. Force one with the environment variable `HSK_BACKEND`. The NumPy fallback has the same API but a different RNG, so it **cannot read or write HSK1 blobs** and cannot be merged with native sketches.

## Python quickstart

    import numpy as np, hsk

    sk = hsk.StreamingSketch(10_000, 10_000, rank=16, seed=42)
    sk.update_coo(rows, cols, vals)                      # int64 rows/cols, float64 vals
    sk.update_coo(rows[:100], cols[:100], -vals[:100])   # delete 100 entries
    U, S, V = sk.finalize(16)                            # U: m x 16, S: (16,), V: n x 16

    blob = sk.to_bytes()                                 # HSK1 blob (usable as a DuckDB BLOB)
    sk2 = hsk.StreamingSketch.from_bytes(blob)
    sk.merge(sk2)                                        # params must match

Disk-backed state, updated in place with no serialization:

    with hsk.StreamingSketch.create_memmap("state/a", 10_000, 10_000, 16, seed=42) as sk:
        sk.update_coo(rows, cols, vals)                  # writes straight into the memmap files
    sk = hsk.StreamingSketch.open_memmap("state/a")      # reopen later

Sliding window with an anomaly score:

    rs = hsk.RollingFactorStream(m, n, rank=8, window=50)   # window counts pushes (ticks)
    score = rs.push(r, c, v)        # expired ticks are retracted with -val
    U, S, V = rs.factors(8)
    rs.anomaly_score()              # fraction of the last batch outside the window's top-8 subspace

## Python API

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

    # two-pass randomized SVD and helpers
    hsk.rsvd_certified(A, rank, oversample=10, power_iters=1, c_factor=8.0, seed=42)   # -> (U, S, V, certified_err)
    hsk.adaptive_rsvd(A, rel_tol, max_rank, ...)
    hsk.residual_norm_estimate  hsk.residual_spectral_estimate  hsk.spectral_norm_estimate
    hsk.HybridSketch(n, l, c_factor=8.0, seed=1)     # NumPy port of the test matrix

Notes:

- **`finalize(k)` returns V (n×k), not Vᵀ.** NumPy's `svd` and SciPy's `svds` return Vᵀ (k×n), so transpose accordingly when reconstructing `U @ diag(S) @ V.T`.
- The Python `finalize` clamps `k` to `min(rank, l)`.
- `rows` and `cols` must be integer arrays. Floats raise `TypeError`.
- Bad input raises before any entry is applied (`ValueError`, `IndexError`).
- A corrupt blob raises `RuntimeError`.

## DuckDB extension

Built against **DuckDB v1.1.2** (C++ extension API). Load an unsigned local build by starting the CLI with `duckdb -unsigned`:

    LOAD '/path/to/hsk.duckdb_extension';

    -- aggregate: m, n, rank, seed must be constants
    SELECT hsk_sketch_agg(row_idx, col_idx, val, 10000, 10000, 16, 42) AS sk FROM entries;

    -- merge two sketches (parameters must match)
    SELECT hsk_sketch_merge(a.sk, b.sk) FROM shard_a a, shard_b b;

    -- factors: use LATERAL
    SELECT s.component_idx, s.singular_value, s.u_vec, s.v_vec
    FROM sketches t, LATERAL hsk_sketch_svd(t.sk, 5) s;

- Rows with a NULL argument are skipped. NaN/Inf values and out-of-range indices are errors.
- A negative `val` subtracts. Duplicate `(row_idx, col_idx)` pairs sum.
- An aggregate over zero rows returns NULL.
- Defaults match Python: oversample 10, core factor 4.0, CountSketch width factor 8.0.
- **Memory:** each aggregate state (per thread, per group) holds a full sketch. The per-sketch cap is 1 GiB (environment variable `HSK_MAX_SKETCH_BYTES`, hard limit below 4 GiB). Prefer few groups with large sketches over many groups.
- **Known gap:** `hsk_sketch_svd` can return components beyond `rank` (up to `rank + oversample`), and those are mostly noise. The Python API clamps to `rank`. The extension does not yet. Request `k <= rank` until this is fixed.

## Memory

Sketch state is `8 * (l*m + s*n)` bytes plus an 88-byte header, where `l = min(rank + oversample, m, n)` and `s = min(m, max(l, ceil(core_factor * l)))`. It does not depend on the number of entries streamed. Example: m = n = 10,000, rank 8 gives l = 18, s = 72 and 7.2 MB. The shared sign tables add up to 8 MB per test matrix.

## Accuracy: read this before relying on the numbers

This is a **single-pass** sketch with no power iterations. Its error depends on how much energy sits outside the top-k subspace.

- It works well when the matrix has a clear low-rank structure and a small tail.
- It does **not** recover the top singular values of noise-dominated or flat-spectrum matrices.
- When you can afford a second pass over the data, use `rsvd_certified`. It has not been benchmarked on noise-dominated matrices yet.

## Measured results

Single runs on a laptop with synthetic data: random indices for the throughput rows, planted structure for the accuracy rows. Indicative, not a controlled study.

| What | Result |
|---|---|
| Ingestion, 10,000 × 10,000, rank 16, 1M entries | ~0.1–0.14 s (7–9 M entries/s), 10.4 MB state |
| Streaming, 100,000 × 100,000, rank 16, 350M entries | ~1.5–2.5 minutes (2.5–4.5 M entries/s), 99 MiB sketch, process memory flat at ~150–215 MiB |
| Throughput on the demo stream | ~9–11 M entries/s |
| Planted structure, 5000 × 5000 (six communities + noise): top singular value | rel. error 6e-4 |
| Same, worst of top 6 singular values | rel. error 3.9e-3 |
| Same, rank-6 reconstruction error | 3.28e-2 (hsk) vs 3.18e-2 (exact `svds`) |
| Same, worst subspace angle, sin θ | 5.9e-2 |
| 4-shard merge, blob merge, delete-vs-rebuild | agree with a single pass to ~1e-15 |
| Memory while streaming 20M entries | +24 MB process growth |

## Not yet measured

- **Clustered / slowly decaying spectra** (for example 100, 95, 92, ...), where iterative solvers need many restarts.
- `rsvd_certified` on noise-dominated matrices.

## Known limits

- Merged blobs are not byte-deterministic (merge order and FMA contraction change last-bit results). Compare with tolerances, not blob equality.
- **Sliding-window drift.** `RollingFactorStream` retracts expired ticks with `-val`, which leaves small rounding residue over long runs. `rebuild_every` is **off by default** (0). Set it (for example every few hundred expired ticks) or call `rebuild()` to recompute the sketch from the retained window. The cost is O(nnz in window).
- `RollingFactorStream` keeps its window of ticks in memory only. Reopening a memmap sketch restores the sketch but not the window.
- Memmap state persists through the OS page cache and `flush()`. A crash mid-update can leave a partially applied entry. Take periodic `to_bytes()` snapshots if you need recovery points.
- Sketches are not thread-safe for concurrent mutation.
- **RNG and portability.** Native sketches derive hashes, signs and sampled Hadamard rows from SplitMix64, so blobs are reproducible across compilers and OSes by design. The NumPy fallback uses NumPy's `default_rng` for the sign vector and row sampling, so its sketches are incompatible. The certified error estimate in `rsvd_certified` uses `std::mt19937_64` and `std::normal_distribution` and may differ slightly by platform. Sketches are unaffected.
- Built and exercised so far: Windows wheels (CPython 3.9–3.13) and a DuckDB v1.1.2 extension binary. Linux and macOS builds are defined in `release.yml` but are unverified here.

## License

Copyright (c) 2026 Arha Bhattacharya. All rights reserved.
This is an early-stage research project. I'm happy to share it for academic use. Please get in touch: arhabhattacharya@gmail.com