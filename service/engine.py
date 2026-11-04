"""Low-rank engine: CountSketch + SRHT hybrid (HMT-style), native or pure-NumPy backend.

NOT TESTED. The sandbox could not build the native module or install dependencies.

* "native": the Eigen/C++ module `hsk_native`. Fast path. Sketch blobs are the HSK1 binary format,
            byte-compatible with the C++ SDK and the DuckDB extension (hsk_sketch_agg / _merge / _svd).
* "numpy" : pure NumPy/SciPy port of the same algorithm. Same API, but its RNG differs from C++, so
            it cannot read or write HSK1 blobs (to_bytes / from_bytes raise NotImplementedError).
Force a backend with env HSK_BACKEND=numpy|native.

State layout is identical to C++ in both backends:  Yt = (A Omega)^T is (l x m),  W = Psi A is (s x n).

Persistence:
* to_bytes() / from_bytes(): HSK1 blob, raw buffer. The old .npz path is gone.
* create_memmap() / open_memmap(): Yt and W live in Fortran-ordered np.memmap files that the native
  sketch mutates in place, with no serialization. Durability is the OS page cache plus flush(); a
  crash mid-update can leave a partially applied entry. Take periodic to_bytes() snapshots if you
  need atomic recovery points.
"""
from __future__ import annotations

import json
import math
import os
from collections import deque

import numpy as np
import scipy.sparse as sp

_native = None
_forced = os.environ.get("HSK_BACKEND", "").lower()
if _forced != "numpy":
    try:
        from . import hsk_native as _native  # installed next to this file in the wheel
    except ImportError:
        try:
            import hsk_native as _native
        except ImportError:
            _native = None
    if _native is None and _forced == "native":
        raise ImportError("HSK_BACKEND=native but hsk_native is not built")

BACKEND = "native" if _native is not None else "numpy"
HEADER_BYTES = 88

_U64 = np.uint64
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_MMAP_FORMAT = "hsk-mmap-1"

__all__ = [
    "BACKEND", "StreamingSketch", "RollingFactorStream", "sketch_to_bytes", "sketch_from_bytes",
    "estimate_sketch_bytes", "rsvd_certified", "adaptive_rsvd", "HybridSketch",
    "residual_norm_estimate", "residual_spectral_estimate", "spectral_norm_estimate",
]


# ----------------------------------------------------------------------------
# NumPy port of HybridSketch (fallback backend only)
# ----------------------------------------------------------------------------
def _splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = x + _U64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> _U64(30))) * _U64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> _U64(27))) * _U64(0x94D049BB133111EB)
        return x ^ (x >> _U64(31))


def _parity32(x: np.ndarray) -> np.ndarray:
    x = x ^ (x >> 16)
    x = x ^ (x >> 8)
    x = x ^ (x >> 4)
    x = x ^ (x >> 2)
    x = x ^ (x >> 1)
    return (x & 1).astype(np.float64)


def fwht_rows(Z: np.ndarray) -> np.ndarray:
    """Unnormalized Walsh-Hadamard transform of every row; row length must be 2^k."""
    r, N = Z.shape
    h = 1
    while h < N:
        Z = Z.reshape(r, N // (2 * h), 2, h)
        a, b = Z[:, :, 0, :], Z[:, :, 1, :]
        Z = np.stack((a + b, a - b), axis=2)
        h *= 2
    return Z.reshape(r, N)


class HybridSketch:
    """Implicit n x l test matrix: Omega(j,t) = sg(j) d(b(j)) (-1)^{popcount(b(j)&r_t)} / sqrt(l)."""

    def __init__(self, n: int, l: int, c_factor: float = 8.0, seed: int = 1):
        if n < 1 or l < 1:
            raise ValueError("bad dims")
        rng = np.random.default_rng(seed)
        self.n, self.l = int(n), int(l)
        c = 1 << max(0, int(math.ceil(c_factor * l)) - 1).bit_length()
        self.use_cs = c < n
        self.N = c if self.use_cs else 1 << max(0, n - 1).bit_length()
        if l > self.N:
            raise ValueError("l > N")
        self.hash_seed = _U64(rng.integers(0, 2**63, dtype=np.int64))
        self.d = rng.choice([-1.0, 1.0], self.N)
        self.samples = np.sort(rng.choice(self.N, self.l, replace=False)).astype(np.uint32)
        self.scale = 1.0 / math.sqrt(l)
        self._cs_mat = None

    def _bucket_sign(self, js: np.ndarray):
        js = np.asarray(js)
        if not self.use_cs:
            return js.astype(np.uint32), np.ones(js.shape, dtype=np.float64)
        with np.errstate(over="ignore"):
            h = _splitmix64(self.hash_seed ^ (js.astype(np.uint64) * _U64(0x9E3779B97F4A7C15)))
            b = (((h >> _U64(32)) * _U64(self.N)) >> _U64(32)).astype(np.uint32)
        sg = 1.0 - 2.0 * (h & _U64(1)).astype(np.float64)
        return b, sg

    def rows(self, js: np.ndarray) -> np.ndarray:
        """Omega[js, :]  (len(js) x l)."""
        b, sg = self._bucket_sign(js)
        par = _parity32(b[:, None] & self.samples[None, :])
        return (sg * self.d[b] * self.scale)[:, None] * (1.0 - 2.0 * par)

    def _cs(self) -> sp.csr_matrix:  # n x N sparse CountSketch (transposed)
        if self._cs_mat is None:
            b, sg = self._bucket_sign(np.arange(self.n))
            self._cs_mat = sp.csr_matrix((sg, (np.arange(self.n), b)), shape=(self.n, self.N))
        return self._cs_mat

    def apply(self, A, chunk: int = 4096) -> np.ndarray:
        """A @ Omega for dense ndarray or scipy sparse A (m x n)."""
        if A.shape[1] != self.n:
            raise ValueError("A.shape[1] != n")
        if sp.issparse(A):
            A = A.tocsr()
        m = A.shape[0]
        out = np.empty((m, self.l))
        C = self._cs() if self.use_cs else None
        for r0 in range(0, m, chunk):
            blk = A[r0:r0 + chunk]
            if self.use_cs:
                Z = blk @ C
                Z = Z.toarray() if sp.issparse(Z) else np.asarray(Z)
            else:
                Z = np.zeros((blk.shape[0], self.N))
                Z[:, :self.n] = blk.toarray() if sp.issparse(blk) else blk
            Z = Z * self.d
            out[r0:r0 + blk.shape[0]] = fwht_rows(Z)[:, self.samples] * self.scale
        return out

    def dense(self) -> np.ndarray:
        return self.rows(np.arange(self.n))


# ----------------------------------------------------------------------------
# Shared numerics (two-pass randomized SVD and error estimates)
# ----------------------------------------------------------------------------
def _svd_from_Bt(Q: np.ndarray, Bt: np.ndarray, k: int):
    """A ~= Q B with B^T = Bt (n x l). Returns U (m x k), S (k), V (n x k)."""
    Qb, Rb = np.linalg.qr(Bt)
    Ur, S, Vrt = np.linalg.svd(Rb)
    k = min(k, S.shape[0])
    return Q @ Vrt.T[:, :k], S[:k], Qb @ Ur[:, :k]


def residual_norm_estimate(A, Q: np.ndarray, r: int = 10, seed: int = 7) -> float:
    """With prob >= 1-10^-r:  ||A - QQ^T A||_2 <= this  (HMT eq. 4.3)."""
    rng = np.random.default_rng(seed)
    AG = A @ rng.standard_normal((A.shape[1], r))
    AG = AG - Q @ (Q.T @ AG)
    return 10.0 * _SQRT_2_OVER_PI * float(np.linalg.norm(AG, axis=0).max())


def spectral_norm_estimate(A, iters: int = 12, seed: int = 3) -> float:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(A.shape[1])
    x /= np.linalg.norm(x)
    sigma = 0.0
    for _ in range(iters):
        z = A.T @ (A @ x)
        nz = float(np.linalg.norm(z))
        if nz == 0.0:
            return 0.0
        sigma = math.sqrt(nz)
        x = z / nz
    return sigma


def residual_spectral_estimate(A, Q: np.ndarray, iters: int = 20, seed: int = 11) -> float:
    """Power-iteration estimate of ||(I - QQ^T) A||_2 (tight in practice, no guarantee)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(A.shape[1])
    x /= np.linalg.norm(x)
    sigma = 0.0
    for _ in range(iters):
        y = A @ x
        y = y - Q @ (Q.T @ y)
        z = A.T @ y
        nz = float(np.linalg.norm(z))
        if nz == 0.0:
            return 0.0
        sigma = math.sqrt(nz)
        x = z / nz
    return sigma


def _rsvd_numpy(A, rank, oversample, power_iters, c_factor, seed):
    m, n = A.shape
    l = min(rank + oversample, m, n)
    Q, _ = np.linalg.qr(HybridSketch(n, l, c_factor, seed).apply(A))
    for _ in range(power_iters):
        Z, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ Z)
    U, S, V = _svd_from_Bt(Q, np.asarray(A.T @ Q), rank)
    return U, S, V, residual_norm_estimate(A, U)


def rsvd_certified(A, rank: int, oversample: int = 10, power_iters: int = 1,
                   c_factor: float = 8.0, seed: int = 42):
    """Two-pass randomized SVD. A: dense ndarray or scipy sparse. -> (U, S, V, certified_spectral_err)."""
    if _native is None:
        return _rsvd_numpy(A, rank, oversample, power_iters, c_factor, seed)
    if sp.issparse(A):
        c = A.tocoo()
        return _native.rsvd_coo(A.shape[0], A.shape[1], c.row.astype(np.int64),
                                c.col.astype(np.int64), c.data.astype(np.float64),
                                rank, oversample, power_iters, c_factor, seed)
    return _native.rsvd_dense(np.asarray(A, dtype=np.float64), rank, oversample,
                              power_iters, c_factor, seed)


def adaptive_rsvd(A, rel_tol: float, max_rank: int, oversample: int = 10,
                  power_iters: int = 1, seed: int = 42, rigorous: bool = False):
    """Double the rank until the residual is <= rel_tol * ||A||_2.

    Returns (U, S, V, bound, estimate, norm_estimate).
    """
    norm = spectral_norm_estimate(A)
    target = rel_tol * norm
    kmax = min(max_rank, *A.shape)
    k = 8
    while True:
        k = min(k, kmax)
        U, S, V, bound = rsvd_certified(A, k, oversample, power_iters, seed=seed)
        est = residual_spectral_estimate(A, U)
        if (bound if rigorous else est) <= target or k >= kmax:
            return U, S, V, bound, est, norm
        k *= 2


# ----------------------------------------------------------------------------
# Streaming sketches (single pass, turnstile, mergeable)
# ----------------------------------------------------------------------------
class _Params:
    def __init__(self, m, n, rank, oversample, core_factor, c_factor, seed):
        self.m, self.n, self.rank = int(m), int(n), int(rank)
        self.oversample, self.core_factor = int(oversample), float(core_factor)
        self.c_factor, self.seed = float(c_factor), int(seed)
        if min(self.m, self.n, self.rank) < 1 or self.oversample < 0:
            raise ValueError("m, n, rank must be >= 1 and oversample >= 0")
        if not 0 <= self.seed < 2**64:
            raise ValueError("seed must be in [0, 2**64)")
        # same derivation as hsk::detail::make_dims
        self.l = min(self.rank + self.oversample, self.m, self.n)
        self.s = min(self.m, max(self.l, int(math.ceil(self.core_factor * self.l))))

    def as_dict(self):
        return dict(m=self.m, n=self.n, rank=self.rank, oversample=self.oversample,
                    core_factor=self.core_factor, c_factor=self.c_factor, seed=self.seed)

    @property
    def state_bytes(self) -> int:  # Yt + W payload
        return 8 * (self.l * self.m + self.s * self.n)


def _mmap_paths(path):
    base = os.fspath(path)
    return base + ".yt.f64", base + ".w.f64", base + ".hsk.json"


def _as_int_array(x, name):
    a = np.asarray(x)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if a.size and not np.issubdtype(a.dtype, np.integer):
        raise TypeError(f"{name} must be an integer array")
    return a.astype(np.int64, copy=False)


class _StreamingBase:
    """Common construction, memmap persistence and lifecycle."""

    p: _Params
    _attached: tuple = ()

    @classmethod
    def create_memmap(cls, path, m, n, rank, oversample=10, core_factor=4.0, c_factor=8.0,
                      seed=0xC0FFEE, *, overwrite=False):
        """Create a zeroed, disk-backed sketch. Files: <path>.yt.f64, <path>.w.f64, <path>.hsk.json."""
        p = _Params(m, n, rank, oversample, core_factor, c_factor, seed)
        yt_p, w_p, meta_p = _mmap_paths(path)
        if not overwrite:
            for f in (yt_p, w_p, meta_p):
                if os.path.exists(f):
                    raise FileExistsError(f)
        yt = np.memmap(yt_p, dtype=np.float64, mode="w+", shape=(p.l, p.m), order="F")
        w = np.memmap(w_p, dtype=np.float64, mode="w+", shape=(p.s, p.n), order="F")
        with open(meta_p, "w") as f:
            json.dump({"format": _MMAP_FORMAT, "layout": "column-major",
                       "yt_shape": [p.l, p.m], "w_shape": [p.s, p.n], "params": p.as_dict()}, f)
        return cls(**p.as_dict(), yt=yt, w=w)

    @classmethod
    def open_memmap(cls, path, mode="r+"):
        """Re-attach to files made by create_memmap. mode 'r+' persists updates; 'c' is copy-on-write."""
        if mode not in ("r+", "c"):
            raise ValueError("mode must be 'r+' (persist) or 'c' (copy-on-write)")
        yt_p, w_p, meta_p = _mmap_paths(path)
        with open(meta_p) as f:
            meta = json.load(f)
        if meta.get("format") != _MMAP_FORMAT:
            raise ValueError("not an hsk memmap sketch")
        p = _Params(**meta["params"])
        if meta["yt_shape"] != [p.l, p.m] or meta["w_shape"] != [p.s, p.n]:
            raise ValueError("memmap metadata is inconsistent with its parameters")
        if os.path.getsize(yt_p) != 8 * p.l * p.m or os.path.getsize(w_p) != 8 * p.s * p.n:
            raise ValueError("memmap file size does not match the recorded shape")
        yt = np.memmap(yt_p, dtype=np.float64, mode=mode, shape=(p.l, p.m), order="F")
        w = np.memmap(w_p, dtype=np.float64, mode=mode, shape=(p.s, p.n), order="F")
        return cls(**p.as_dict(), yt=yt, w=w)

    def flush(self):
        for a in self._attached:
            if isinstance(a, np.memmap):
                a.flush()

    def close(self):
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def memory_bytes(self) -> int:
        return self.p.state_bytes


class _NativeStreaming(_StreamingBase):
    def __init__(self, m, n, rank, oversample=10, core_factor=4.0, c_factor=8.0, seed=0xC0FFEE,
                 *, yt=None, w=None):
        self.p = _Params(m, n, rank, oversample, core_factor, c_factor, seed)
        p = self.p
        self.s_ = _native.StreamingLowRank(p.m, p.n, p.rank, p.oversample, p.core_factor,
                                           p.c_factor, p.seed, yt, w)
        self._attached = tuple(a for a in (yt, w) if a is not None)

    # ---- updates -----------------------------------------------------------
    def update_coo(self, I, J, V):
        self.s_.update_coo(_as_int_array(I, "rows"), _as_int_array(J, "cols"),
                           np.ascontiguousarray(V, dtype=np.float64))

    def update_dense(self, D):
        self.s_.update_dense(np.asarray(D, dtype=np.float64))

    def merge(self, other):
        self.s_.merge(other.s_)

    def clear(self):
        self.s_.clear()

    def finalize(self, k):
        return self.s_.finalize(int(k))

    # ---- state (standardized shapes: Yt is l x m, W is s x n) --------------
    @property
    def yt(self):
        return self.s_.yt

    @property
    def w(self):
        return self.s_.w

    def set_state(self, yt, w):
        self.s_.set_state(yt, w)

    # ---- HSK1 binary buffers ----------------------------------------------
    def to_bytes(self) -> bytes:
        return self.s_.serialize()

    def write_into(self, buf) -> int:
        """Serialize into a pre-allocated writable buffer (bytearray, mmap, uint8 array)."""
        return self.s_.serialize_into(buf)

    def load_bytes(self, data):
        self.s_.load_from_buffer(data)

    def merge_bytes(self, data):
        self.s_.merge_from_buffer(data)

    @property
    def blob_size(self) -> int:
        return self.s_.blob_size

    @classmethod
    def from_bytes(cls, data):
        s = _native.StreamingLowRank.deserialize_from_buffer(data)
        self = cls.__new__(cls)
        self.p = _Params(s.m, s.n, s.rank, s.oversample, s.core_factor, s.c_factor, s.seed)
        self.s_ = s
        self._attached = ()
        return self

    @staticmethod
    def inspect_bytes(data) -> dict:
        return _native.StreamingLowRank.inspect_buffer(data)


class _NumpyStreaming(_StreamingBase):
    def __init__(self, m, n, rank, oversample=10, core_factor=4.0, c_factor=8.0, seed=0xC0FFEE,
                 *, yt=None, w=None):
        self.p = _Params(m, n, rank, oversample, core_factor, c_factor, seed)
        p = self.p
        self.om = HybridSketch(p.n, p.l, p.c_factor, p.seed)
        self.psi = HybridSketch(p.m, p.s, p.c_factor, p.seed ^ 0xA5A5A5A5DEADBEEF)
        if (yt is None) != (w is None):
            raise ValueError("pass both yt and w, or neither")
        if yt is None:
            self.Yt = np.zeros((p.l, p.m), order="F")
            self.W = np.zeros((p.s, p.n), order="F")
        else:
            for name, a, shape in (("yt", yt, (p.l, p.m)), ("w", w, (p.s, p.n))):
                if a.dtype != np.float64 or a.shape != shape or not a.flags.writeable:
                    raise ValueError(f"{name} must be a writeable float64 array of shape {shape}")
            self.Yt, self.W = yt, w
            self._attached = (yt, w)

    def update_coo(self, I, J, V, chunk: int = 50_000):
        p = self.p
        I, J = _as_int_array(I, "rows"), _as_int_array(J, "cols")
        V = np.ascontiguousarray(V, dtype=np.float64)
        if not (I.shape == J.shape == V.shape):
            raise ValueError("rows/cols/vals must have equal length")
        if V.size:
            if I.min() < 0 or I.max() >= p.m or J.min() < 0 or J.max() >= p.n:
                raise IndexError("update_coo: index outside [0,m) x [0,n)")
            if not np.isfinite(V).all():
                raise ValueError("update_coo: value must be finite")
        for a in range(0, len(V), chunk):
            i, j, v = I[a:a + chunk], J[a:a + chunk], V[a:a + chunk]
            idx = np.arange(len(v))
            self.Yt += (sp.csr_matrix((v, (i, idx)), shape=(p.m, len(v))) @ self.om.rows(j)).T
            self.W += (sp.csr_matrix((v, (j, idx)), shape=(p.n, len(v))) @ self.psi.rows(i)).T

    def update_dense(self, D):
        D = np.asarray(D, dtype=np.float64)
        if D.shape != (self.p.m, self.p.n) or not np.isfinite(D).all():
            raise ValueError("update_dense: bad shape or non-finite values")
        self.Yt += self.om.apply(D).T
        self.W += self.psi.apply(D.T).T

    def merge(self, other):
        if other.p.as_dict() != self.p.as_dict():
            raise ValueError("incompatible sketches")
        self.Yt += other.Yt
        self.W += other.W

    def clear(self):
        self.Yt[:] = 0.0
        self.W[:] = 0.0

    def finalize(self, k):
        k = max(1, min(int(k), self.p.rank, self.p.l))
        Q, _ = np.linalg.qr(self.Yt.T)
        PQ = self.psi.apply(Q.T).T                       # s x l  (= Psi Q)
        X = np.linalg.lstsq(PQ, self.W, rcond=None)[0]   # l x n
        return _svd_from_Bt(Q, X.T, k)

    @property
    def yt(self):
        v = self.Yt.view()
        v.flags.writeable = False
        return v

    @property
    def w(self):
        v = self.W.view()
        v.flags.writeable = False
        return v

    def set_state(self, yt, w):
        yt, w = np.asarray(yt, dtype=np.float64), np.asarray(w, dtype=np.float64)
        if yt.shape != self.Yt.shape or w.shape != self.W.shape:
            raise ValueError("set_state: wrong shapes")
        if not (np.isfinite(yt).all() and np.isfinite(w).all()):
            raise ValueError("set_state: non-finite values")
        self.Yt[:] = yt
        self.W[:] = w

    def _no_blob(self, *_a, **_k):
        raise NotImplementedError("HSK1 blobs need the native backend (the NumPy RNG differs from C++)")

    to_bytes = write_into = load_bytes = merge_bytes = _no_blob

    @property
    def blob_size(self) -> int:
        return HEADER_BYTES + self.p.state_bytes

    @classmethod
    def from_bytes(cls, data):
        raise NotImplementedError("HSK1 blobs need the native backend (the NumPy RNG differs from C++)")


StreamingSketch = _NativeStreaming if _native is not None else _NumpyStreaming


def sketch_to_bytes(sk) -> bytes:
    """HSK1 blob (raw binary; replaces the old .npz container)."""
    return sk.to_bytes()


def sketch_from_bytes(data):
    return StreamingSketch.from_bytes(data)


def estimate_sketch_bytes(m, n, rank, oversample=10, core_factor=4.0, c_factor=8.0, *, blob=False) -> int:
    """Bytes of sketch state, 8*(l*m + s*n). With blob=True add the 88-byte HSK1 header."""
    p = _Params(m, n, rank, oversample, core_factor, c_factor, 0)
    return p.state_bytes + (HEADER_BYTES if blob else 0)


# ----------------------------------------------------------------------------
# RollingFactorStream : sliding-window low-rank factors + spectral residual anomaly score
# ----------------------------------------------------------------------------
def _residual_fraction(U: np.ndarray, rows, cols, vals, m: int) -> float:
    """sqrt(||(I - U U^T) B||_F^2 / ||B||_F^2) for the sparse batch B. 0 = inside the subspace, 1 = orthogonal."""
    uc, inv = np.unique(cols, return_inverse=True)
    B = sp.coo_matrix((vals, (rows, inv.ravel())), shape=(m, uc.size)).tocsr()  # duplicates sum
    total = float(B.multiply(B).sum())
    if total <= 0.0:
        return 0.0
    P = np.asarray(B.T @ U)                          # (#distinct cols) x k
    proj = float(np.einsum("ij,ij->", P, P))
    return math.sqrt(max(total - proj, 0.0) / total)


class RollingFactorStream:
    """Sliding window of matrix updates over a StreamingSketch.

    Each push() is one tick: a COO batch (rows, cols, vals). The window holds the last `window` ticks.
    Ticks that fall out are retracted with v = -val (exact up to floating-point rounding).

    anomaly_score(): for the most recent push, the fraction of the batch's Frobenius energy that lies
    outside the rank-`score_rank` column space of the window *before* that batch was added. It is in [0, 1];
    NaN while the window is empty or the batch is empty. The subspace is cached and refreshed every
    `refresh_every` ticks, so it can lag by that many ticks. Use refresh_every=1 for exact but
    slower scoring (one finalize per tick).

    Drift control: cancellation leaves small rounding residue. rebuild_every=N rebuilds the sketch from
    the retained window every N expired ticks (cost O(window nnz)); 0 disables it. rebuild() does it on demand.

    Memory: the window stores copies of its COO batches, so it is O(nnz in window) beyond the sketch.
    The window is not persisted when the sketch is memmap-backed. Reopening the sketch restores the
    sketch state but not the deque, so retraction of old ticks is lost; keep your own tick log if you need that.
    """

    def __init__(self, m, n, rank, window, *, oversample=10, core_factor=4.0, c_factor=8.0,
                 seed=0xC0FFEE, score_rank=None, refresh_every=8, rebuild_every=0, sketch=None):
        if int(window) < 1:
            raise ValueError("window must be >= 1 tick")
        if sketch is None:
            sketch = StreamingSketch(m, n, rank, oversample, core_factor, c_factor, seed)
        elif (sketch.p.m, sketch.p.n, sketch.p.rank) != (int(m), int(n), int(rank)):
            raise ValueError("sketch parameters do not match (m, n, rank)")
        self.sketch = sketch
        self.m, self.n, self.rank = sketch.p.m, sketch.p.n, sketch.p.rank
        self.window = int(window)
        self.score_rank = self.rank if score_rank is None else int(score_rank)
        self.refresh_every = max(1, int(refresh_every))
        self.rebuild_every = max(0, int(rebuild_every))
        self._win: deque = deque()
        self._ticks = 0
        self._version = 0
        self._expired = 0
        self._fcache = None
        self._basis = None          # (U, S0) used for scoring
        self._basis_tick = 0
        self._last_score = float("nan")

    # ---- helpers -----------------------------------------------------------
    def _coerce(self, rows, cols, vals):
        r = _as_int_array(rows, "rows").copy()
        c = _as_int_array(cols, "cols").copy()
        v = np.array(vals, dtype=np.float64, copy=True).ravel()
        if not (r.shape == c.shape == v.shape):
            raise ValueError("rows/cols/vals must have equal length")
        if v.size:
            if r.min() < 0 or r.max() >= self.m or c.min() < 0 or c.max() >= self.n:
                raise IndexError("index outside [0,m) x [0,n)")
            if not np.isfinite(v).all():
                raise ValueError("values must be finite")
        return r, c, v

    def _ensure_basis(self, force=False):
        if not self._win:
            return None
        if force or self._basis is None or self._ticks - self._basis_tick >= self.refresh_every:
            U, S, _ = self.factors(self.score_rank)
            self._basis = (U, float(S[0]) if S.size else 0.0)
            self._basis_tick = self._ticks
        return self._basis

    def _score(self, r, c, v, force=False) -> float:
        if r.size == 0:
            return float("nan")
        basis = self._ensure_basis(force)
        if basis is None or basis[1] <= 0.0:
            return float("nan")
        return _residual_fraction(basis[0], r, c, v, self.m)

    # ---- streaming ---------------------------------------------------------
    def push(self, rows, cols, vals) -> float:
        """Ingest one tick (may be empty) and expire old ticks. Returns this batch's anomaly score."""
        r, c, v = self._coerce(rows, cols, vals)
        score = self._score(r, c, v)           # against the window as it was before this tick
        if r.size:
            self.sketch.update_coo(r, c, v)    # validates before mutating, so a failure leaves state intact
        self._win.append((r, c, v))
        self._ticks += 1
        self._version += 1
        while len(self._win) > self.window:
            er, ec, ev = self._win.popleft()
            if er.size:
                self.sketch.update_coo(er, ec, -ev)   # turnstile retraction: v = -val
            self._expired += 1
        if self.rebuild_every and self._expired >= self.rebuild_every:
            self.rebuild()
        self._last_score = score
        return score

    def rebuild(self):
        """Recompute the sketch exactly from the retained window (removes accumulated rounding drift)."""
        self.sketch.clear()
        for r, c, v in self._win:
            if r.size:
                self.sketch.update_coo(r, c, v)
        self._expired = 0
        self._version += 1
        self._basis = None
        self._fcache = None

    # ---- outputs -----------------------------------------------------------
    def factors(self, k=None):
        """Top-k factors of the window: (U m x k, S k, V n x k). k is clamped to the sketch rank."""
        k = self.rank if k is None else int(k)
        if k < 1:
            raise ValueError("k must be >= 1")
        kk = min(k, self.rank, self.sketch.p.l)
        if not self._win:
            return np.zeros((self.m, kk)), np.zeros(kk), np.zeros((self.n, kk))
        key = (self._version, k)
        if self._fcache is not None and self._fcache[0] == key:
            return self._fcache[1]
        res = self.sketch.finalize(k)
        self._fcache = (key, res)
        return res

    def singular_values(self, k=None) -> np.ndarray:
        return self.factors(k)[1]

    def anomaly_score(self, batch=None) -> float:
        """Score of the latest push, or of `batch=(rows, cols, vals)` against the current window (fresh subspace)."""
        if batch is None:
            return self._last_score
        r, c, v = self._coerce(*batch)
        return self._score(r, c, v, force=True)

    @property
    def ticks_in_window(self) -> int:
        return len(self._win)

    @property
    def total_ticks(self) -> int:
        return self._ticks