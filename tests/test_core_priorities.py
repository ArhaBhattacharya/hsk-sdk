import numpy as np
import pytest
import scipy.sparse as sp
import hsk


@pytest.mark.parametrize("idx_dtype", [np.int32, np.int64])
def test_update_csr_matches_coo_and_deletes_exact(idx_dtype):
    rng = np.random.default_rng(42)
    A = sp.random(1200, 900, density=0.015, format="csr", random_state=rng, dtype=np.float64)
    A.indptr = A.indptr.astype(idx_dtype, copy=False)
    A.indices = A.indices.astype(idx_dtype, copy=False)

    sk_csr = hsk.StreamingSketch(1200, 900, rank=12, seed=7)
    sk_coo = hsk.StreamingSketch(1200, 900, rank=12, seed=7)

    sk_csr.update_csr(A)
    coo = A.tocoo()
    sk_coo.update_coo(coo.row, coo.col, coo.data)

    np.testing.assert_allclose(sk_csr.yt, sk_coo.yt, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sk_csr.w, sk_coo.w, rtol=1e-12, atol=1e-12)

    # Scale factor != 1.0
    sk_csr.update_csr(A, scale=0.5)
    sk_coo.update_coo(coo.row, coo.col, 0.5 * coo.data)
    np.testing.assert_allclose(sk_csr.yt, sk_coo.yt, rtol=1e-12, atol=1e-12)

    # Exact turnstile deletion via scale=-1.5
    sk_csr.update_csr(A, scale=-1.5)
    assert np.max(np.abs(sk_csr.yt)) < 1e-11
    assert np.max(np.abs(sk_csr.w)) < 1e-11


def test_update_csr_spare_capacity_and_csc_conversion():
    A = sp.random(300, 250, density=0.02, format="csr", random_state=1)
    # Simulate over-allocated buffers past indptr[-1] with garbage beyond nnz
    nnz = int(A.indptr[-1])
    padded_indices = np.concatenate([A.indices, np.array([999999, -5], dtype=A.indices.dtype)])
    padded_data = np.concatenate([A.data, np.array([np.nan, np.inf], dtype=np.float64)])
    A_padded = sp.csr_matrix((A.data.copy(), A.indices.copy(), A.indptr.copy()), shape=A.shape)
    A_padded.indices = padded_indices
    A_padded.data = padded_data

    sk1 = hsk.StreamingSketch(300, 250, rank=8, seed=99)
    sk2 = hsk.StreamingSketch(300, 250, rank=8, seed=99)
    sk1.update_csr(A_padded)
    sk2.update_csr(A.tocsc())  # auto-converts non-CSR sparse input

    np.testing.assert_allclose(sk1.yt, sk2.yt, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sk1.w, sk2.w, rtol=1e-12, atol=1e-12)


def test_update_csr_atomic_validation():
    A = sp.random(200, 150, density=0.05, format="csr", random_state=3)
    sk = hsk.StreamingSketch(200, 150, rank=6, seed=5)
    sk.update_csr(A)
    yt_before = sk.yt.copy()
    w_before = sk.w.copy()

    # 1. Non-sparse input -> TypeError
    with pytest.raises(TypeError):
        sk.update_csr(A.toarray())

    # 2. Wrong shape -> ValueError
    with pytest.raises(ValueError):
        sk.update_csr(sp.csr_matrix((199, 150)))

    # 3. Non-finite scale -> ValueError
    with pytest.raises(ValueError):
        sk.update_csr(A, scale=float("nan"))

    # 4. Out-of-bounds column index -> IndexError
    bad_col = A.copy()
    bad_col.indices = bad_col.indices.copy()
    bad_col.indices[len(bad_col.indices) // 2] = 150
    with pytest.raises(IndexError):
        sk.update_csr(bad_col)

    # 5. Negative column index -> IndexError
    neg_col = A.copy()
    neg_col.indices = neg_col.indices.copy()
    neg_col.indices[-1] = -1
    with pytest.raises(IndexError):
        sk.update_csr(neg_col)

    # 6. NaN in data (even with scale=0.0) -> ValueError
    bad_val = A.copy()
    bad_val.data = bad_val.data.copy()
    bad_val.data[-1] = np.nan
    with pytest.raises(ValueError):
        sk.update_csr(bad_val, scale=0.0)

    # 7. Corrupt indptr -> ValueError
    bad_ptr = A.copy()
    bad_ptr.indptr = bad_ptr.indptr.copy()
    bad_ptr.indptr[10] = bad_ptr.indptr[9] - 1
    with pytest.raises(ValueError):
        sk.update_csr(bad_ptr)

    # Sketch state must remain completely untouched across all failed calls
    np.testing.assert_array_equal(sk.yt, yt_before)
    np.testing.assert_array_equal(sk.w, w_before)


def test_finalize_clamps_k_to_rank():
    rng = np.random.default_rng(0)
    A = sp.random(100, 80, density=0.1, format="csr", random_state=rng)
    sk = hsk.StreamingSketch(100, 80, rank=5, oversample=10, seed=11)
    sk.update_csr(A)

    U, S, V = sk.finalize(12)  # > rank (5), < l (15)
    assert U.shape == (100, 5)
    assert S.shape == (5,)
    assert V.shape == (80, 5)

    with pytest.raises(ValueError):
        sk.finalize(0)
