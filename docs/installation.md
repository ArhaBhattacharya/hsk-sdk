# Installation

Install the **`hsk`** Python SDK or DuckDB extension for certified randomized SVD and matrix sketching.

> [!NOTE]
> **Pre-release (`v0.1.1`):** `hsk-sdk` is not yet published to PyPI. Prebuilt binaries are hosted on the [v0.1.1 GitHub Release](https://github.com/ArhaBhattacharya/hsk-sdk/releases/tag/v0.1.1). The distribution name for `pip` is **`hsk-sdk`**, while the Python import name is **`hsk`**. Prebuilt wheel users do **not** need Eigen 3.4 or a C++ compiler installed.

---

## Quick Start: Python SDK

### 1. Install the prebuilt wheel

Run the command matching your OS (examples shown for **Python 3.12**—replace both instances of `cp312` with `cp39`, `cp310`, `cp311`, or `cp313` to match your interpreter):

```bash
# Windows (64-bit) — cp312 verified
python -m pip install https://github.com/ArhaBhattacharya/hsk-sdk/releases/download/v0.1.1/hsk_sdk-0.1.1-cp312-cp312-win_amd64.whl

# Linux (x86_64, glibc >= 2.17) — cp313 verified
python -m pip install https://github.com/ArhaBhattacharya/hsk-sdk/releases/download/v0.1.1/hsk_sdk-0.1.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# macOS (Apple Silicon, macOS 11+)
python -m pip install https://github.com/ArhaBhattacharya/hsk-sdk/releases/download/v0.1.1/hsk_sdk-0.1.1-cp312-cp312-macosx_11_0_arm64.whl
```

> [!TIP]
> **Using Jupyter, Google Colab, or multiple Python environments?** Always invoke `python -m pip` with your target interpreter—or use `%pip install ...` inside a notebook—so `pip` installs into the active environment.

<details>
<summary><strong>Wheel filename reference &amp; offline installation</strong></summary>

#### Supported wheel tags
Prebuilt wheels follow the format `hsk_sdk-0.1.1-<py>-<py>-<platform>.whl`:

| System | Platform tag | Supported Python tags |
|---|---|---|
| Windows (64-bit) | `win_amd64` | `cp39`, `cp310`, `cp311`, `cp312`, `cp313` |
| Linux (x86_64, glibc ≥ 2.17) | `manylinux_2_17_x86_64.manylinux2014_x86_64` | `cp39`, `cp310`, `cp311`, `cp312`, `cp313` |
| macOS (Apple Silicon, 11+) | `macosx_11_0_arm64` | `cp39`, `cp310`, `cp311`, `cp312`, `cp313` |

*No prebuilt wheels exist for Intel Macs, Linux `aarch64`, Alpine/`musl` Linux, 32-bit Windows, PyPy, or Python 3.14. On those platforms, [build from source](#build-python-sdk-from-source).*

#### Installing from a local download or offline folder
If you downloaded the `.whl` file manually:
```bash
python -m pip install hsk_sdk-0.1.1-cp312-cp312-win_amd64.whl
```

To let `pip` pick the right wheel from a local directory (ensure `numpy>=1.23` and `scipy>=1.9` are already installed or present in `./wheelhouse`):
```bash
python -m pip install --no-index --find-links ./wheelhouse hsk-sdk
```
*(Avoid passing shell globs like `hsk_sdk-*.whl` to `pip install`, as your shell will expand all files and attempt to install incompatible wheels.)*

#### Verify download checksums (optional)
Compare your downloaded file against the SHA-256 digest listed on the Release page:
```bash
sha256sum <file>                       # Linux
shasum -a 256 <file>                   # macOS
```
```powershell
Get-FileHash <file> -Algorithm SHA256  # Windows
```
</details>

### 2. Verify the installation

Confirm that the compiled C++ backend loaded (rather than the slower pure-Python fallback) and run a quick randomized SVD check:

```python
import numpy as np
import hsk

print("Backend:", hsk.BACKEND)  # Must print: native

rng = np.random.default_rng(0)
A = rng.standard_normal((2000, 20)) @ rng.standard_normal((20, 1500))  # exact rank 20
U, S, V, bound = hsk.rsvd_certified(A, 20)
print("Relative error:", np.linalg.norm(A - (U * S) @ V.T) / np.linalg.norm(A))
# Expected: Backend: native | Relative error: ~3e-15
```

If `hsk.BACKEND` prints `numpy`, the native module failed to import—see [Troubleshooting](#troubleshooting).

---

## DuckDB Extension

> [!WARNING]
> **Exact DuckDB version required (`v1.1.2`):** Loadable C++ extensions are compiled against a specific DuckDB release and will refuse to load in any other version. Ensure you are running `duckdb==1.1.2`.

### 1. Download and rename (Windows x64)

DuckDB derives the extension entry point from its **filename**, so the downloaded asset (`hsk-Windows.duckdb_extension`, ~21 MB) **must be renamed to `hsk.duckdb_extension`**.

```powershell
# Download and save with the required filename in one step
Invoke-WebRequest -Uri "https://github.com/ArhaBhattacharya/hsk-sdk/releases/download/v0.1.1/hsk-Windows.duckdb_extension" -OutFile hsk.duckdb_extension
```

*(Linux `x86_64` and macOS `arm64` builds are available in the [CI workflow artifacts](#ci-artifacts-fallback); unzip and rename `hsk-Linux.duckdb_extension` or `hsk-macOS.duckdb_extension` to `hsk.duckdb_extension`.)*

### 2. Load and verify

Because the `v0.1.1` binary is unsigned, enable unsigned extensions when connecting:

**From Python (`pip install duckdb==1.1.2`):**
```python
import duckdb

con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
con.execute("LOAD 'C:/path/to/hsk.duckdb_extension'")

# Verify sketching and SVD
con.execute("""
    CREATE TABLE e AS
    SELECT (i % 1000)::BIGINT r, ((i*7) % 500)::BIGINT c, (i % 13)::DOUBLE v
    FROM range(100000) t(i);
""")
print(con.execute("SELECT hsk_sketch_agg(r, c, v, 1000, 500, 5, 42::UBIGINT) IS NOT NULL FROM e;").fetchone())
# Expected: (True,)

print(con.execute("""
    WITH s AS (SELECT hsk_sketch_agg(r, c, v, 1000, 500, 5, 42::UBIGINT) sk FROM e)
    SELECT t.component_idx, t.singular_value FROM s, LATERAL hsk_sketch_svd(s.sk, 5) t;
""").fetchall())
# Expected: 5 rows with singular values in descending order
```

**From the DuckDB CLI (`duckdb -unsigned`):**
```sql
LOAD '/path/to/hsk.duckdb_extension';

CREATE TABLE e AS
SELECT (i % 1000)::BIGINT r, ((i*7) % 500)::BIGINT c, (i % 13)::DOUBLE v
FROM range(100000) t(i);

WITH s AS (SELECT hsk_sketch_agg(r, c, v, 1000, 500, 5, 42::UBIGINT) sk FROM e)
SELECT t.component_idx, t.singular_value FROM s, LATERAL hsk_sketch_svd(s.sk, 5) t;
```

---

## Building from Source

### Requirements

| Component | Python Wheel | DuckDB Extension |
|---|---|---|
| **Compiler & Build** | C++17 compiler, CMake ≥ 3.15 | C++17 compiler, CMake ≥ 3.15 |
| **Dependencies** | Eigen ≥ 3.3 (CI uses `3.4.0`), `numpy>=1.23`, `scipy>=1.9` | Eigen ≥ 3.3, DuckDB `v1.1.2` source tree |
| **Python** | 3.9 – 3.13 (3.14 untested) | Optional (only if driving DuckDB via Python) |

### Build Python SDK from source

Prebuilt wheel users do not need Eigen 3.4 or a C++ compiler installed. Only source builds require Eigen 3.4 (and a C++17 compiler) to already be available unless Eigen is installed system-wide via CMake's `find_package(Eigen3)`, because `pyproject.toml` sets `HSK_FETCH_EIGEN=OFF`.

```bash
# Linux / macOS
git clone https://github.com/ArhaBhattacharya/hsk-sdk.git && cd hsk-sdk
# Optional: run `git checkout v0.1.1` to match the release tag
git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git /tmp/eigen
CMAKE_ARGS="-DHSK_EIGEN_INCLUDE_DIR=/tmp/eigen" python -m pip install .
```

```powershell
# Windows (run inside Developer PowerShell for VS so MSVC is on PATH)
git clone https://github.com/ArhaBhattacharya/hsk-sdk.git; cd hsk-sdk
git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git C:\src\eigen
$env:CMAKE_ARGS = "-DHSK_EIGEN_INCLUDE_DIR=C:/src/eigen"
python -m pip install .
```

To build a redistributable wheel instead of installing directly (or for offline builds with pre-installed `scikit-build-core pybind11 numpy scipy`):
```bash
CMAKE_ARGS="-DHSK_EIGEN_INCLUDE_DIR=/path/to/eigen" python -m pip wheel . --no-build-isolation --no-deps -w dist/
```

### Build DuckDB extension from source

> [!NOTE]
> Compiling DuckDB from source is resource-intensive (CI allows up to 2 hours). CMake will use a system Eigen installation (`-DHSK_EIGEN_INCLUDE_DIR=<dir>`) or download it automatically.

```bash
git clone --depth 1 --branch v1.1.2 https://github.com/duckdb/duckdb.git ~/duckdb
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHSK_DUCKDB_SOURCE_DIR=$HOME/duckdb
cmake --build build --config Release --target duckdb_hsk --parallel
# Output: build/duckdb-build/extension/hsk/hsk.duckdb_extension (already named properly)
```

---

## Troubleshooting

### Installation & Loading Issues

| Symptom | Cause & Fix |
|---|---|
| `pip show hsk` returns *not found* | The package name is `hsk-sdk`, not `hsk`, or `pip` points to a different Python environment. Run `python -m pip show -f hsk-sdk`. |
| `No matching distribution found` / `not a supported wheel` | Either no wheel exists for your OS/architecture (e.g., Intel Mac, Linux `aarch64`, Python 3.14), or `pip` is outdated. Upgrade `pip` (`python -m pip install --upgrade pip`) so it recognizes `manylinux2014` and `macosx_11_0` tags. |
| `404` when installing from a GitHub Release URL | Check the URL path (`/releases/download/v0.1.1/...`), or if accessing a private fork, download the asset via your browser while signed in to GitHub and install the local file. |
| `hsk.BACKEND` prints `numpy` | The C++ extension failed to load and fell back to pure Python. Run `import hsk.hsk_native` to inspect the import error, or set `HSK_BACKEND=native` before importing to raise a hard error. |
| Extension fails to load with a version mismatch | Your DuckDB version is not `1.1.2`. Install `duckdb==1.1.2` or use the `v1.1.2` CLI. |
| Extension refuses to load as *unsigned* | Initialize DuckDB with `config={"allow_unsigned_extensions": "true"}` or launch the CLI with `duckdb -unsigned`. |
| Extension loads, but entry point is not found | Rename the binary to `hsk.duckdb_extension`. |

### Runtime & Usage Caveats

| Behavior / Error | Explanation |
|---|---|
| `Binder Error: No function matches ... hsk_sketch_agg(..., INTEGER)` | The `seed` argument requires `UBIGINT` and DuckDB does not implicitly cast `INTEGER`. Cast explicitly: `42::UBIGINT`. |
| `argument 'm' must be a constant expression` (also `n`, `rank`, `seed`) | Dimension, rank, and seed parameters must be literal constants rather than column references. |
| `sketch would need N bytes, above the limit` | Sketch size exceeds the default `1 GiB` per-sketch cap. Reduce `m`, `n`, or `rank`, or increase `HSK_MAX_SKETCH_BYTES` (hard cap: `3500 MiB`). |
| Binary blob interoperability | The native Python backend and DuckDB extension share the `HSK1` binary sketch format (cross-OS blob exchange is untested). The pure-Python `numpy` fallback uses a different RNG and cannot read, write, or merge `HSK1` blobs. Sketches only merge when `(m, n, rank, seed)` match. |

---

## Release & Platform Support Matrix

The `v0.1.1` release artifacts are built from the `v0.1.1` git tag (or CI release workflow).

| Component | Verification Status | Notes |
|---|---|---|
| **Windows x64, Python 3.12** wheel | **Tested** | Verified locally by maintainer |
| Windows x64, Python 3.9–3.11, 3.13 wheels | Built by CI | Packaged in Release; not yet runtime-tested |
| **Linux `x86_64` (`manylinux`), Python 3.13** wheel | **Tested** | Verified on Python 3.13.18 (Google Colab) |
| Linux `x86_64` (`manylinux`), Python 3.9–3.12 wheels | Built by CI | Packaged in Release; not yet runtime-tested |
| macOS `arm64`, Python 3.9–3.13 wheels | Built by CI | Packaged in Release; not yet runtime-tested |
| **DuckDB extension, Windows x64 (`v1.1.2`)** | **Tested** | Verified locally by maintainer |
| DuckDB extension, Linux `x86_64` (`v1.1.2`) | Built by CI | Available in GitHub Actions artifacts only; not yet runtime-tested |
| DuckDB extension, macOS `arm64` (`v1.1.2`) | Built by CI | Available in GitHub Actions artifacts only; not yet runtime-tested |
| Python 3.14 | Not built | Source build is untested |

### CI Artifacts Fallback

If you need the Linux/macOS DuckDB extension or an unreleased build, download the zipped artifacts from a [Release workflow run](https://github.com/ArhaBhattacharya/hsk-sdk/actions/workflows/release.yml) (requires GitHub sign-in; artifacts expire after the repository retention window):
* `wheels-windows-latest`, `wheels-ubuntu-latest`, `wheels-macos-latest`
* `duckdb-extension-Windows`, `duckdb-extension-Linux`, `duckdb-extension-macOS`