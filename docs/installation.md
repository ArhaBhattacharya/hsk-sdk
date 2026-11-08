# Installation

The Python package is distributed as platform-specific wheels. The package
name is `hsk-sdk`, and the import name is `hsk`.

## Requirements

- CPython 3.9 through 3.13
- macOS, Linux, or Windows
- NumPy and SciPy (installed automatically by pip when installing a wheel)

## Install a wheel from GitHub Actions

The release workflow currently publishes wheels as GitHub Actions artifacts,
not to PyPI. Open the [Release workflow][release-workflow], select a
successful run, and download the artifact for your operating system:

- `wheels-macos-latest`
- `wheels-ubuntu-latest`
- `wheels-windows-latest`

Extract the downloaded artifact, then install the wheel with:

```bash
python -m pip install /path/to/hsk_sdk-<version>-<python>-<platform>.whl
```

If you put several wheels in the same directory, pip can select the matching
wheel automatically:

```bash
python -m pip install /path/to/wheelhouse/hsk_sdk-*.whl
```

For a private repository, download the artifact while signed in to GitHub.
The artifact is intended for testing and internal distribution; it is not a
stable package URL.

## Verify the installation

```bash
python -c "import hsk; print(hsk.BACKEND)"
```

The native wheel should print `native`. The package also supports a NumPy
fallback; set `HSK_BACKEND=numpy` before importing `hsk` to select it.

## Build and install from source

Eigen 3.4 must be available locally because this project disables dependency
downloads during the build. With the repository checked out:

```bash
python -m pip install --upgrade pip
python -m pip install scikit-build-core pybind11 numpy scipy
CMAKE_ARGS="-DHSK_EIGEN_INCLUDE_DIR=/path/to/eigen" \
  python -m pip install . --no-build-isolation
```

On Windows PowerShell, use:

```powershell
$env:CMAKE_ARGS = "-DHSK_EIGEN_INCLUDE_DIR=C:\path\to\eigen"
python -m pip install . --no-build-isolation
```

To build a wheel instead of installing directly:

```bash
CMAKE_ARGS="-DHSK_EIGEN_INCLUDE_DIR=/path/to/eigen" \
  python -m pip wheel . --no-build-isolation --no-deps -w dist/
```

## Compatibility note

Native sketches use the portable HSK1 binary format and can be exchanged
between supported platforms. The NumPy fallback uses a different random
number generator and cannot read, write, or merge native HSK1 blobs.

[release-workflow]: https://github.com/ArhaBhattacharya/hsk-sdk/actions/workflows/release.yml
