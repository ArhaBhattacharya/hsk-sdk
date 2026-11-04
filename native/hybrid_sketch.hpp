// hybrid_sketch.hpp : single-header, Eigen-based randomized low-rank engine
// =========================================================================
// Omega = CountSketch(n -> c) composed with SRHT(c -> l),  c ~ c_factor * l
//
//   * CountSketch : O(nnz(A))            (HMT-style structured test matrices, sec. 4.6)
//   * SRHT        : O(m c log c)
//   * Streaming   : linear sketches => turnstile updates, exact shard merging,
//                   and a portable binary format (below) usable as a SQL BLOB.
//
// Build: C++17, Eigen >= 3.3.  Optional: -fopenmp-simd (vectorized hot loops),
//        -fopenmp (parallel dense apply; do NOT use inside host processes that own
//        their thread pool, e.g. DuckDB; define EIGEN_DONT_PARALLELIZE there).
//
// Binary sketch format v1 (little-endian, no padding, 88-byte header):
//    off  0  u32  magic 0x314B5348 ("HSK1")      off 40  i64  l  (derived, validated)
//    off  4  u32  format version = 1             off 48  i64  s  (derived, validated)
//    off  8  i64  m                              off 56  f64  core_factor
//    off 16  i64  n                              off 64  f64  c_factor
//    off 24  i64  rank                           off 72  u64  seed
//    off 32  i64  oversample                     off 80  u64  checksum of payload
//    payload @88: Yt (l x m, column-major f64) then W (s x n, column-major f64)
// Sketches are mergeable iff (m, n, rank, oversample, core_factor, c_factor, seed) match.
// The random test matrices come from SplitMix64 only (no <random> distributions), so
// sketches are bit-for-bit reproducible across compilers/OSes; merged sums differ only
// by floating-point rounding order.
//
// Thread-safety: const methods are thread-safe; update*/merge*/set_state are not.
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "hsk requires IEEE-754 binary64 doubles");
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "hsk binary format is little-endian; big-endian hosts are unsupported");
#endif

#if defined(_MSC_VER)
#define HSK_RESTRICT __restrict
#define HSK_SIMD
#define HSK_PREFETCH_W(p) ((void)0)
#else
#define HSK_RESTRICT __restrict__
#define HSK_PREFETCH_W(p) __builtin_prefetch((p), 1, 3)
#if defined(_OPENMP) || defined(HSK_OMP_SIMD)
#define HSK_PRAGMA_(x) _Pragma(#x)
#define HSK_SIMD HSK_PRAGMA_(omp simd)
#else
#define HSK_SIMD
#endif
#endif

namespace hsk {

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using SpMat = Eigen::SparseMatrix<double>;  // column-major (CSC)

struct SvdResult {
  MatrixXd U;  // m x k, orthonormal columns
  VectorXd S;  // k singular values, descending
  MatrixXd V;  // n x k, orthonormal columns        A ~= U * diag(S) * V^T
};

// ============================================================================
// Pure helpers (no Eigen): RNG, checksum, blob header, blob merge.
// ---- BEGIN PURE HELPERS ----
// ============================================================================
namespace detail {

constexpr std::int64_t kMaxDim = 2147483647LL;          // m, n, rank, oversample limit
constexpr std::uint32_t kMagic = 0x314B5348u;           // bytes 'H','S','K','1'
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::size_t kHeaderBytes = 88;
constexpr std::size_t kChecksumOffset = 80;

inline std::uint64_t splitmix64(std::uint64_t x) {      // stateless mixer: mix(x + golden)
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// Deterministic stream (standard SplitMix64). Platform-independent, unlike <random> distributions.
class SplitMix64 {
 public:
  explicit SplitMix64(std::uint64_t seed) : s_(seed) {}
  std::uint64_t next() {
    const std::uint64_t out = splitmix64(s_);
    s_ += 0x9E3779B97F4A7C15ULL;
    return out;
  }
  // Uniform in [0, bound), bound >= 1. Rejection sampling => exactly uniform and portable.
  std::uint64_t bounded(std::uint64_t bound) {
    const std::uint64_t threshold = (0 - bound) % bound;   // 2^64 mod bound
    for (;;) {
      const std::uint64_t r = next();
      if (r >= threshold) return r % bound;
    }
  }

 private:
  std::uint64_t s_;
};

inline std::size_t next_pow2(std::size_t x) {
  std::size_t p = 1;
  while (p < x) p <<= 1;
  return p;
}

inline int parity32(std::uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_parity(x);
#else
  x ^= x >> 16; x ^= x >> 8; x ^= x >> 4; x ^= x >> 2; x ^= x >> 1;
  return static_cast<int>(x & 1u);
#endif
}

template <class T>
inline T load_pod(const std::uint8_t* p, std::size_t off) {
  T v;
  std::memcpy(&v, p + off, sizeof(T));
  return v;
}
template <class T>
inline void store_pod(std::uint8_t* p, std::size_t off, T v) {
  std::memcpy(p + off, &v, sizeof(T));
}

inline std::uint64_t rotl64(std::uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }

// Fast order-dependent 64-bit checksum over 8-byte words (~5 GB/s). Detects corruption and
// truncation; it is NOT cryptographic.
class Checksum {
 public:
  void word(std::uint64_t w) {
    h_ = rotl64(h_ ^ (w * kK1), 27) * kK2;
    ++n_;
  }
  void doubles(const double* p, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
      std::uint64_t w;
      std::memcpy(&w, p + i, sizeof(w));
      word(w);
    }
  }
  void bytes(const std::uint8_t* p, std::size_t ndoubles) {
    for (std::size_t i = 0; i < ndoubles; ++i) {
      std::uint64_t w;
      std::memcpy(&w, p + 8 * i, sizeof(w));
      word(w);
    }
  }
  std::uint64_t digest() const {
    std::uint64_t x = h_ ^ (n_ * kK1);
    x ^= x >> 33; x *= kK1; x ^= x >> 29; x *= kK2; x ^= x >> 32;
    return x;
  }

 private:
  static constexpr std::uint64_t kK1 = 0xFF51AFD7ED558CCDULL;
  static constexpr std::uint64_t kK2 = 0xC4CEB9FE1A85EC53ULL;
  std::uint64_t h_ = 0x243F6A8885A308D3ULL;
  std::uint64_t n_ = 0;
};

struct BlobHeader {
  std::int64_t m = 0, n = 0, rank = 0, oversample = 0, l = 0, s = 0;
  double core_factor = 0.0, c_factor = 0.0;
  std::uint64_t seed = 0, checksum = 0;
};

struct Dims {
  std::int64_t l, s;
};

// Validates sketch parameters and derives (l, s). Throws std::invalid_argument.
inline Dims make_dims(std::int64_t m, std::int64_t n, std::int64_t rank, std::int64_t oversample,
                      double core_factor, double c_factor) {
  if (m < 1 || n < 1 || m > kMaxDim || n > kMaxDim)
    throw std::invalid_argument("m and n must be in [1, 2147483647]");
  if (rank < 1 || rank > kMaxDim) throw std::invalid_argument("rank must be in [1, 2147483647]");
  if (oversample < 0 || oversample > kMaxDim)
    throw std::invalid_argument("oversample must be in [0, 2147483647]");
  if (!(core_factor >= 1.0 && core_factor <= 16.0))
    throw std::invalid_argument("core_factor must be in [1, 16]");
  if (!(c_factor >= 2.0 && c_factor <= 64.0))
    throw std::invalid_argument("c_factor must be in [2, 64]");
  const std::int64_t l = std::min<std::int64_t>(rank + oversample, std::min(m, n));
  const double want = std::ceil(core_factor * static_cast<double>(l));
  const std::int64_t s = std::min<std::int64_t>(m, std::max<std::int64_t>(l, static_cast<std::int64_t>(want)));
  return Dims{l, s};
}

inline std::size_t checked_mul(std::size_t a, std::size_t b) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::overflow_error("hsk: size overflow");
  return a * b;
}
inline std::size_t checked_add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a) throw std::overflow_error("hsk: size overflow");
  return a + b;
}
inline std::size_t payload_doubles(const BlobHeader& h) {   // l*m + s*n
  return checked_add(checked_mul(static_cast<std::size_t>(h.l), static_cast<std::size_t>(h.m)),
                     checked_mul(static_cast<std::size_t>(h.s), static_cast<std::size_t>(h.n)));
}
inline std::size_t blob_bytes(const BlobHeader& h) {
  return checked_add(kHeaderBytes, checked_mul(payload_doubles(h), 8));
}

inline void write_header(std::uint8_t* dst, const BlobHeader& h) {
  store_pod<std::uint32_t>(dst, 0, kMagic);
  store_pod<std::uint32_t>(dst, 4, kFormatVersion);
  store_pod<std::int64_t>(dst, 8, h.m);
  store_pod<std::int64_t>(dst, 16, h.n);
  store_pod<std::int64_t>(dst, 24, h.rank);
  store_pod<std::int64_t>(dst, 32, h.oversample);
  store_pod<std::int64_t>(dst, 40, h.l);
  store_pod<std::int64_t>(dst, 48, h.s);
  store_pod<double>(dst, 56, h.core_factor);
  store_pod<double>(dst, 64, h.c_factor);
  store_pod<std::uint64_t>(dst, 72, h.seed);
  store_pod<std::uint64_t>(dst, kChecksumOffset, h.checksum);
}

// Parses + validates structure and exact length. Does NOT verify the payload checksum.
inline BlobHeader read_header(const std::uint8_t* buf, std::size_t size) {
  if (buf == nullptr || size < kHeaderBytes) throw std::invalid_argument("hsk blob: truncated header");
  if (load_pod<std::uint32_t>(buf, 0) != kMagic)
    throw std::invalid_argument("hsk blob: bad magic (not an hsk sketch)");
  if (load_pod<std::uint32_t>(buf, 4) != kFormatVersion)
    throw std::invalid_argument("hsk blob: unsupported format version");
  BlobHeader h;
  h.m = load_pod<std::int64_t>(buf, 8);
  h.n = load_pod<std::int64_t>(buf, 16);
  h.rank = load_pod<std::int64_t>(buf, 24);
  h.oversample = load_pod<std::int64_t>(buf, 32);
  h.l = load_pod<std::int64_t>(buf, 40);
  h.s = load_pod<std::int64_t>(buf, 48);
  h.core_factor = load_pod<double>(buf, 56);
  h.c_factor = load_pod<double>(buf, 64);
  h.seed = load_pod<std::uint64_t>(buf, 72);
  h.checksum = load_pod<std::uint64_t>(buf, kChecksumOffset);
  const Dims d = make_dims(h.m, h.n, h.rank, h.oversample, h.core_factor, h.c_factor);
  if (d.l != h.l || d.s != h.s) throw std::invalid_argument("hsk blob: inconsistent dimensions");
  if (blob_bytes(h) != size) throw std::invalid_argument("hsk blob: size does not match header");
  return h;
}

// Name of the first differing parameter, or nullptr if the sketches are mergeable.
inline const char* first_param_mismatch(const BlobHeader& a, const BlobHeader& b) {
  if (a.m != b.m) return "m";
  if (a.n != b.n) return "n";
  if (a.rank != b.rank) return "rank";
  if (a.oversample != b.oversample) return "oversample";
  if (a.core_factor != b.core_factor) return "core_factor";
  if (a.c_factor != b.c_factor) return "c_factor";
  if (a.seed != b.seed) return "seed";
  return nullptr;
}

inline std::uint64_t checksum_bytes(const std::uint8_t* payload, std::size_t ndoubles) {
  Checksum cs;
  cs.bytes(payload, ndoubles);
  return cs.digest();
}

// dst[i] += src[i] for `count` doubles stored (possibly unaligned) in a byte buffer.
inline void add_from_bytes(double* HSK_RESTRICT dst, const std::uint8_t* src, std::size_t count) {
  constexpr std::size_t kBlock = 256;
  double tmp[kBlock];
  for (std::size_t i = 0; i < count; i += kBlock) {
    const std::size_t c = std::min(kBlock, count - i);
    std::memcpy(tmp, src + i * 8, c * 8);
    for (std::size_t k = 0; k < c; ++k) dst[i + k] += tmp[k];
  }
}

// out = a + b, single fused pass: verifies both input checksums and computes the output's.
// `out` must have exactly nout == na bytes and must not alias a or b.
inline void merge_blobs(const std::uint8_t* a, std::size_t na, const std::uint8_t* b, std::size_t nb,
                        std::uint8_t* out, std::size_t nout) {
  const BlobHeader ha = read_header(a, na);
  const BlobHeader hb = read_header(b, nb);
  if (const char* bad = first_param_mismatch(ha, hb))
    throw std::invalid_argument(std::string("incompatible sketches: parameter '") + bad + "' differs");
  if (out == nullptr || nout != na) throw std::invalid_argument("hsk merge: bad output buffer size");

  const std::size_t total = payload_doubles(ha);
  const std::uint8_t* pa = a + kHeaderBytes;
  const std::uint8_t* pb = b + kHeaderBytes;
  std::uint8_t* po = out + kHeaderBytes;
  Checksum ca, cb, co;
  constexpr std::size_t kBlock = 256;
  double ta[kBlock], tb[kBlock];
  for (std::size_t i = 0; i < total; i += kBlock) {
    const std::size_t c = std::min(kBlock, total - i);
    std::memcpy(ta, pa + i * 8, c * 8);
    std::memcpy(tb, pb + i * 8, c * 8);
    ca.doubles(ta, c);
    cb.doubles(tb, c);
    for (std::size_t k = 0; k < c; ++k) ta[k] += tb[k];
    co.doubles(ta, c);
    std::memcpy(po + i * 8, ta, c * 8);
  }
  if (ca.digest() != ha.checksum) throw std::runtime_error("hsk blob: first sketch is corrupt (checksum)");
  if (cb.digest() != hb.checksum) throw std::runtime_error("hsk blob: second sketch is corrupt (checksum)");
  BlobHeader hout = ha;
  hout.checksum = co.digest();
  write_header(out, hout);
}

}  // namespace detail
// ---- END PURE HELPERS ----

namespace detail {

// Thin orthonormal basis of range(M).
inline MatrixXd orth(const MatrixXd& M) {
  Eigen::HouseholderQR<MatrixXd> qr(M);
  return qr.householderQ() * MatrixXd::Identity(M.rows(), M.cols());
}

// Given Q (m x l, orthonormal) and Bt = B^T = A^T Q (n x l):  A ~= Q B.
// Tall-skinny SVD of Bt via QR first (cheap l x l SVD), then lift.
inline SvdResult svd_from_Bt(const MatrixXd& Q, const MatrixXd& Bt, Index k) {
  const Index n = Bt.rows(), l = Bt.cols();
  Eigen::HouseholderQR<MatrixXd> qr(Bt);
  MatrixXd Rb = qr.matrixQR().topRows(l).triangularView<Eigen::Upper>();
  Eigen::BDCSVD<MatrixXd> svd(Rb, Eigen::ComputeThinU | Eigen::ComputeThinV);
  k = std::min<Index>(k, l);
  MatrixXd Uext = MatrixXd::Zero(n, l);
  Uext.topRows(l) = svd.matrixU();
  SvdResult r;
  r.S = svd.singularValues().head(k);
  MatrixXd P = qr.householderQ() * Uext;          // Bt = P S R^T,  P = Qb * Ur
  r.V = P.leftCols(k);
  r.U = Q * svd.matrixV().leftCols(k);            // A ~= (Q R) S P^T
  return r;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// HybridSketch : implicit n x l test matrix
//   Omega(j,t) = sigma(j) * d(b(j)) * (-1)^{popcount(b(j) & r_t)} / sqrt(l)
// b(j): hashed bucket (CountSketch), sigma(j): hashed sign, d: random signs,
// r_t: the l sampled Hadamard rows. If c >= n, CountSketch is skipped (b(j)=j, sigma=1).
//
// Hot path (streaming): add_scaled_row() is dst[0..l) += w * T[b][0..l) where T is a
// precomputed sign table (N x l, row-contiguous) => one unit-stride FMA stream. For very
// large l the table would not fit in cache and a branch-free integer parity kernel is used.
// ---------------------------------------------------------------------------
class HybridSketch {
 public:
  struct Loc {
    std::uint32_t bucket;
    double sigma;
  };

  HybridSketch(Index n, Index l, double c_factor = 8.0, std::uint64_t seed = 1) : n_(n), l_(l) {
    if (n < 1 || l < 1) throw std::invalid_argument("HybridSketch: bad dims");
    detail::SplitMix64 rng(seed);
    hash_seed_ = rng.next();
    const std::size_t c = detail::next_pow2(
        static_cast<std::size_t>(std::ceil(c_factor * static_cast<double>(l))));
    use_cs_ = static_cast<Index>(c) < n;
    N_ = use_cs_ ? c : detail::next_pow2(static_cast<std::size_t>(n));
    if (static_cast<std::size_t>(l) > N_) throw std::invalid_argument("HybridSketch: l > N");
    if (N_ > (static_cast<std::size_t>(1) << 32)) throw std::invalid_argument("HybridSketch: transform size too large");
    scale_ = 1.0 / std::sqrt(static_cast<double>(l));

    d_.resize(N_);
    for (std::size_t i = 0; i < N_; ++i) d_[i] = (rng.next() & 1u) ? -1.0 : 1.0;

    std::vector<std::uint32_t> perm(N_);               // partial Fisher-Yates (portable RNG)
    std::iota(perm.begin(), perm.end(), 0u);
    for (Index t = 0; t < l; ++t) {
      const std::size_t j = static_cast<std::size_t>(t) +
                            static_cast<std::size_t>(rng.bounded(N_ - static_cast<std::size_t>(t)));
      std::swap(perm[static_cast<std::size_t>(t)], perm[j]);
    }
    samples_.assign(perm.begin(), perm.begin() + l);
    std::sort(samples_.begin(), samples_.end());       // better gather locality
    samples64_.assign(samples_.begin(), samples_.end());

    if (N_ * static_cast<std::size_t>(l_) * sizeof(double) <= kSignTableMaxBytes) {
      tab_.resize(N_ * static_cast<std::size_t>(l_));
      for (std::size_t b = 0; b < N_; ++b) {
        double* row = tab_.data() + b * static_cast<std::size_t>(l_);
        const double base = d_[b] * scale_;
        for (Index t = 0; t < l_; ++t)
          row[t] = detail::parity32(static_cast<std::uint32_t>(b) & samples_[static_cast<std::size_t>(t)])
                       ? -base : base;
      }
    }
  }

  Index in_dim() const { return n_; }
  Index out_dim() const { return l_; }
  bool uses_countsketch() const { return use_cs_; }
  std::size_t transform_size() const { return N_; }
  bool uses_sign_table() const { return !tab_.empty(); }

  // Y = A * Omega   (A: m x n dense; pass an evaluated matrix or a cheap view)
  template <class Derived>
  MatrixXd apply(const Eigen::MatrixBase<Derived>& A) const {
    if (A.cols() != n_) throw std::invalid_argument("HybridSketch::apply: A.cols() != n");
    const Index m = A.rows();
    MatrixXd Y(m, l_);

    std::vector<std::uint32_t> bkt;
    std::vector<double> sg;
    if (use_cs_) {
      bkt.resize(static_cast<std::size_t>(n_));
      sg.resize(static_cast<std::size_t>(n_));
      for (Index j = 0; j < n_; ++j) {
        const std::uint64_t h = hash(j);
        bkt[static_cast<std::size_t>(j)] = bucket_from(h);
        sg[static_cast<std::size_t>(j)] = sign_from(h);
      }
    }

    const Index nchunks = (m + kChunk - 1) / kChunk;
#pragma omp parallel
    {
      MatrixXd Zc(kChunk, static_cast<Index>(N_));       // per-thread, L2-resident
      std::vector<double> buf(static_cast<std::size_t>(kSub) * N_);
#pragma omp for schedule(static)
      for (Index ch = 0; ch < nchunks; ++ch) {
        const Index r0 = ch * kChunk;
        const Index len = std::min<Index>(kChunk, m - r0);
        auto Zv = Zc.topRows(len);
        Zv.setZero();
        if (use_cs_) {                                    // CountSketch: unit-stride axpys
          for (Index j = 0; j < n_; ++j)
            Zv.col(bkt[static_cast<std::size_t>(j)]) +=
                sg[static_cast<std::size_t>(j)] * A.col(j).segment(r0, len);
        } else {
          Zv.leftCols(n_) = A.middleRows(r0, len);
        }
        srht_rows(Zc.data(), Zc.outerStride(), len, Y.data() + r0, Y.outerStride(), buf);
      }
    }
    return Y;
  }

  // Y = A * Omega for sparse A (CSC). Picks the cheaper of two exact algorithms:
  //   explicit : O(nnz * l)                      (no FWHT, tiny memory)
  //   fwht     : O(nnz + m * N * log2 N)         (CountSketch then SRHT)
  MatrixXd apply(const SpMat& A) const {
    if (A.cols() != n_) throw std::invalid_argument("HybridSketch::apply: A.cols() != n");
    const double nnz = static_cast<double>(A.nonZeros());
    const double cost_explicit = nnz * static_cast<double>(l_);
    const double cost_fwht =
        nnz + static_cast<double>(A.rows()) * static_cast<double>(N_) *
                  std::log2(static_cast<double>(N_));
    return (cost_explicit <= cost_fwht) ? apply_sparse_explicit(A) : apply_sparse_fwht(A);
  }

  // Bucket and sign of input coordinate j.
  Loc locate(Index j) const {
    if (!use_cs_) return Loc{static_cast<std::uint32_t>(j), 1.0};
    const std::uint64_t h = hash(j);
    return Loc{bucket_from(h), sign_from(h)};
  }

  // dst[0..l) += v * Omega(j, :)      O(l). The streaming hot path.
  void add_scaled_row(Index j, double v, double* HSK_RESTRICT dst) const {
    add_scaled_bucket(locate(j), v, dst);
  }

  void add_scaled_bucket(const Loc& loc, double v, double* HSK_RESTRICT dst) const {
    const double w = v * loc.sigma;
    const Index L = l_;
    if (!tab_.empty()) {
      // Row of precomputed (+-d*scale) values: a single unit-stride multiply-add stream.
      const double* HSK_RESTRICT row = tab_.data() + static_cast<std::size_t>(loc.bucket) * static_cast<std::size_t>(L);
      HSK_SIMD
      for (Index t = 0; t < L; ++t) dst[t] += w * row[t];
    } else {
      // Large-l fallback: branch-free parity in integer lanes, sign applied by flipping the
      // IEEE sign bit of w (x < 2^32 since bucket, sample < N <= 2^32).
      const double ws = w * d_[loc.bucket] * scale_;
      std::uint64_t wbits;
      std::memcpy(&wbits, &ws, sizeof(wbits));
      const std::uint64_t b64 = loc.bucket;
      const std::uint64_t* HSK_RESTRICT smp = samples64_.data();
      HSK_SIMD
      for (Index t = 0; t < L; ++t) {
        std::uint64_t x = b64 & smp[t];
        x ^= x >> 16; x ^= x >> 8; x ^= x >> 4; x ^= x >> 2; x ^= x >> 1;
        const std::uint64_t bits = wbits ^ ((x & 1ULL) << 63);
        double val;
        std::memcpy(&val, &bits, sizeof(val));
        dst[t] += val;
      }
    }
  }

  void row(Index j, double* out) const {
    std::fill(out, out + l_, 0.0);
    add_scaled_row(j, 1.0, out);
  }

  // Explicit n x l matrix (testing / small problems only).
  MatrixXd dense() const {
    MatrixXd O(n_, l_);
    VectorXd tmp(l_);
    for (Index j = 0; j < n_; ++j) {
      row(j, tmp.data());
      O.row(j) = tmp.transpose();
    }
    return O;
  }

 private:
  static constexpr Index kChunk = 256;  // rows per chunk (per-thread working set)
  static constexpr Index kSub = 8;      // rows transformed together
  static constexpr std::size_t kSignTableMaxBytes = 8u << 20;

  std::uint64_t hash(Index j) const {
    return detail::splitmix64(hash_seed_ ^ (static_cast<std::uint64_t>(j) * 0x9E3779B97F4A7C15ULL));
  }
  std::uint32_t bucket_from(std::uint64_t h) const {  // Lemire fastrange on the high bits
    return static_cast<std::uint32_t>(((h >> 32) * static_cast<std::uint64_t>(N_)) >> 32);
  }
  static double sign_from(std::uint64_t h) { return (h & 1u) ? -1.0 : 1.0; }

  // Z: (len x N) column-major, leading dim ldz.  Y: (len x l), leading dim ldy.
  // Y(r,t) = scale * (H D Z(r,:)^T)[samples_[t]]
  void srht_rows(const double* Z, Index ldz, Index len, double* Y, Index ldy,
                 std::vector<double>& buf) const {
    const std::size_t N = N_;
    for (Index s0 = 0; s0 < len; s0 += kSub) {
      const Index sl = std::min<Index>(kSub, len - s0);
      for (std::size_t c = 0; c < N; ++c) {
        const double dc = d_[c];
        const double* zc = Z + static_cast<Index>(c) * ldz + s0;   // contiguous in r
        for (Index r = 0; r < sl; ++r) buf[static_cast<std::size_t>(r) * N + c] = dc * zc[r];
      }
      for (Index r = 0; r < sl; ++r) fwht(&buf[static_cast<std::size_t>(r) * N], N);
      for (Index t = 0; t < l_; ++t) {
        const std::uint32_t s = samples_[static_cast<std::size_t>(t)];
        double* yt = Y + t * ldy + s0;
        for (Index r = 0; r < sl; ++r) yt[r] = scale_ * buf[static_cast<std::size_t>(r) * N + s];
      }
    }
  }

  // In-place unnormalized fast Walsh-Hadamard transform, N a power of two. Radix-4 passes
  // halve the sweeps over memory; unit-stride inner loops auto-vectorize.
  static void fwht(double* HSK_RESTRICT x, std::size_t N) {
    std::size_t h = 1;
    for (; h * 4 <= N; h *= 4) {
      for (std::size_t i = 0; i < N; i += 4 * h) {
        double* p0 = x + i;
        double* p1 = p0 + h;
        double* p2 = p1 + h;
        double* p3 = p2 + h;
        for (std::size_t j = 0; j < h; ++j) {
          const double a = p0[j], b = p1[j], c = p2[j], d = p3[j];
          const double ab = a + b, amb = a - b, cd = c + d, cmd = c - d;
          p0[j] = ab + cd;
          p1[j] = amb + cmd;
          p2[j] = ab - cd;
          p3[j] = amb - cmd;
        }
      }
    }
    if (h < N) {  // one leftover radix-2 pass when log2(N) is odd
      for (std::size_t i = 0; i < N; i += 2 * h) {
        double* p0 = x + i;
        double* p1 = p0 + h;
        for (std::size_t j = 0; j < h; ++j) {
          const double a = p0[j], b = p1[j];
          p0[j] = a + b;
          p1[j] = a - b;
        }
      }
    }
  }

  MatrixXd apply_sparse_explicit(const SpMat& A) const {
    MatrixXd Yt = MatrixXd::Zero(l_, A.rows());           // transposed => contiguous column adds
    VectorXd w(l_);
    for (Index j = 0; j < n_; ++j) {
      SpMat::InnerIterator it(A, j);
      if (!it) continue;
      row(j, w.data());
      for (; it; ++it) Yt.col(it.row()) += it.value() * w;
    }
    return Yt.transpose();
  }

  MatrixXd apply_sparse_fwht(const SpMat& A) const {
    const Index m = A.rows();
    MatrixXd Z = MatrixXd::Zero(m, static_cast<Index>(N_));
    for (Index j = 0; j < n_; ++j) {
      const Loc loc = locate(j);
      for (SpMat::InnerIterator it(A, j); it; ++it) Z(it.row(), loc.bucket) += loc.sigma * it.value();
    }
    MatrixXd Y(m, l_);
    const Index nchunks = (m + kChunk - 1) / kChunk;
#pragma omp parallel
    {
      std::vector<double> buf(static_cast<std::size_t>(kSub) * N_);
#pragma omp for schedule(static)
      for (Index ch = 0; ch < nchunks; ++ch) {
        const Index r0 = ch * kChunk;
        const Index len = std::min<Index>(kChunk, m - r0);
        srht_rows(Z.data() + r0, Z.outerStride(), len, Y.data() + r0, Y.outerStride(), buf);
      }
    }
    return Y;
  }

  Index n_, l_;
  bool use_cs_ = false;
  std::size_t N_ = 0;
  double scale_ = 1.0;
  std::uint64_t hash_seed_ = 0;
  std::vector<double> d_;
  std::vector<std::uint32_t> samples_;
  std::vector<std::uint64_t> samples64_;
  std::vector<double> tab_;   // N x l, row b = d[b]*scale*(-1)^{parity(b & r_t)}; empty if too large
};

// ---------------------------------------------------------------------------
// Two-pass randomized SVD (HMT sec 4.5 + 5.1) with the hybrid test matrix.
// MatT: Eigen::MatrixXd or hsk::SpMat.
// ---------------------------------------------------------------------------
struct RsvdOptions {
  Index rank = 20;
  Index oversample = 10;
  int power_iters = 1;       // 0 for fast-decaying spectra; 1-2 for slow decay / noisy data
  double c_factor = 8.0;     // CountSketch width = nextpow2(c_factor * l)
  std::uint64_t seed = 42;
};

template <class MatT>
SvdResult rsvd(const MatT& A, const RsvdOptions& o = RsvdOptions()) {
  const Index m = A.rows(), n = A.cols();
  const Index l = std::min<Index>(o.rank + o.oversample, std::min(m, n));
  HybridSketch om(n, l, o.c_factor, o.seed);
  MatrixXd Q = detail::orth(om.apply(A));
  for (int i = 0; i < o.power_iters; ++i) {              // re-orthonormalized for stability
    MatrixXd Z = detail::orth(MatrixXd(A.transpose() * Q));
    Q = detail::orth(MatrixXd(A * Z));
  }
  MatrixXd Bt = A.transpose() * Q;                       // n x l  (= (Q^T A)^T)
  return detail::svd_from_Bt(Q, Bt, o.rank);
}

// Certified error: with probability >= 1 - 10^-r,
//   ||A - Q Q^T A||_2 <= 10 sqrt(2/pi) * max_i ||(I - Q Q^T) A g_i||      (HMT eq. 4.3)
// (rigorous but conservative on noisy data; see residual_spectral_estimate for a tight estimate)
template <class MatT>
double residual_norm_estimate(const MatT& A, const MatrixXd& Q, int r = 10,
                              std::uint64_t seed = 7) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  MatrixXd G(A.cols(), r);
  for (Index j = 0; j < G.cols(); ++j)
    for (Index i = 0; i < G.rows(); ++i) G(i, j) = g(rng);
  MatrixXd AG = A * G;
  MatrixXd T = Q.transpose() * AG;
  AG.noalias() -= Q * T;
  constexpr double kSqrt2OverPi = 0.7978845608028654;
  return 10.0 * kSqrt2OverPi * AG.colwise().norm().maxCoeff();
}

// Power-iteration estimate of ||(I - QQ^T) A||_2. Tight in practice, converges from below,
// carries no guarantee.
template <class MatT>
double residual_spectral_estimate(const MatT& A, const MatrixXd& Q, int iters = 20,
                                  std::uint64_t seed = 11) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  VectorXd x(A.cols());
  for (Index i = 0; i < x.size(); ++i) x(i) = g(rng);
  x.normalize();
  double sigma = 0.0;
  for (int i = 0; i < iters; ++i) {
    VectorXd y = A * x;
    y -= Q * (Q.transpose() * y);
    VectorXd z = A.transpose() * y;
    const double nz = z.norm();
    if (nz == 0.0) return 0.0;
    sigma = std::sqrt(nz);
    x = z / nz;
  }
  return sigma;
}

template <class MatT>
double spectral_norm_estimate(const MatT& A, int iters = 12, std::uint64_t seed = 3) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  VectorXd x(A.cols());
  for (Index i = 0; i < x.size(); ++i) x(i) = g(rng);
  x.normalize();
  double sigma = 0.0;
  for (int i = 0; i < iters; ++i) {
    VectorXd y = A * x;
    VectorXd z = A.transpose() * y;
    const double nz = z.norm();
    if (nz == 0.0) return 0.0;
    sigma = std::sqrt(nz);
    x = z / nz;
  }
  return sigma;
}

// Doubles the rank until the residual estimate <= rel_tol * ||A||_2 (or rigorous bound if requested).
template <class MatT>
SvdResult adaptive_rsvd(const MatT& A, double rel_tol, Index max_rank,
                        RsvdOptions o = RsvdOptions(), double* residual_est = nullptr,
                        bool rigorous = false) {
  const double target = rel_tol * spectral_norm_estimate(A);
  const Index kmax = std::min<Index>(max_rank, std::min(A.rows(), A.cols()));
  Index k = std::max<Index>(8, o.rank);
  for (;;) {
    o.rank = std::min(k, kmax);
    SvdResult r = rsvd(A, o);
    const double e = rigorous ? residual_norm_estimate(A, r.U) : residual_spectral_estimate(A, r.U);
    if (e <= target || o.rank >= kmax) {
      if (residual_est) *residual_est = e;
      return r;
    }
    k *= 2;
  }
}

// ---------------------------------------------------------------------------
// StreamingLowRank : single-pass, turnstile, mergeable low-rank sketch (HMT sec. 5.5).
//   Keeps  Yt = (A Omega)^T (l x m)  and  W = Psi A (s x n); both Omega and Psi are hybrid sketches.
//   finalize:  Q = orth(Y),  X = argmin ||(Psi Q) X - W||,  A ~= Q X.
// Memory: (l*m + s*n) doubles. Cost per entry: O(l + s).
// ---------------------------------------------------------------------------
class StreamingLowRank {
 public:
  using BlobInfo = detail::BlobHeader;

  static constexpr Index kDefaultOversample = 10;
  static constexpr double kDefaultCoreFactor = 4.0;
  static constexpr double kDefaultCFactor = 8.0;

  StreamingLowRank(Index m, Index n, Index rank, Index oversample = kDefaultOversample,
                   double core_factor = kDefaultCoreFactor, double c_factor = kDefaultCFactor,
                   std::uint64_t seed = 0xC0FFEEULL)
      : StreamingLowRank(m, n, rank, oversample, core_factor, c_factor, seed,
                         detail::make_dims(m, n, rank, oversample, core_factor, c_factor)) {}

  // Serialized size for given parameters (SIZE_MAX if it does not fit in size_t). Throws on bad params.
  static std::size_t state_bytes(Index m, Index n, Index rank, Index oversample = kDefaultOversample,
                                 double core_factor = kDefaultCoreFactor,
                                 double c_factor = kDefaultCFactor) {
    const detail::Dims d = detail::make_dims(m, n, rank, oversample, core_factor, c_factor);
    const double bytes = 8.0 * (static_cast<double>(d.l) * static_cast<double>(m) +
                                static_cast<double>(d.s) * static_cast<double>(n)) +
                         static_cast<double>(detail::kHeaderBytes);
    if (bytes >= 1.8e19) return std::numeric_limits<std::size_t>::max();
    return static_cast<std::size_t>(bytes);
  }

  // ---- parameters ---------------------------------------------------------
  Index m() const { return m_; }
  Index n() const { return n_; }
  Index rank() const { return rank_; }
  Index oversample() const { return oversample_; }
  double core_factor() const { return core_factor_; }
  double c_factor() const { return c_factor_; }
  std::uint64_t seed() const { return seed_; }
  Index sketch_rank() const { return l_; }   // l
  Index core_size() const { return s_; }     // s

  bool same_params(const StreamingLowRank& o) const {
    return detail::first_param_mismatch(header(0), o.header(0)) == nullptr;
  }

  // ---- state accessors ----------------------------------------------------
  const MatrixXd& yt() const { return Yt_; }   // l x m
  const MatrixXd& w() const { return W_; }     // s x n

  // Replaces the state. Strict: exact dimensions and finite values are required.
  void set_state(const MatrixXd& yt, const MatrixXd& w) {
    if (yt.rows() != l_ || yt.cols() != m_)
      throw std::invalid_argument("set_state: yt must be " + std::to_string(l_) + " x " +
                                  std::to_string(m_) + " (l x m)");
    if (w.rows() != s_ || w.cols() != n_)
      throw std::invalid_argument("set_state: w must be " + std::to_string(s_) + " x " +
                                  std::to_string(n_) + " (s x n)");
    if (!yt.allFinite() || !w.allFinite()) throw std::invalid_argument("set_state: non-finite values");
    Yt_ = yt;
    W_ = w;
  }

  void clear() {
    Yt_.setZero();
    W_.setZero();
  }

  // ---- updates ------------------------------------------------------------
  // A(i,j) += v   (v may be negative: deletions / unlearning are exact)
  void update(Index i, Index j, double v) {
    if (i < 0 || i >= m_ || j < 0 || j >= n_) throw std::out_of_range("update: index outside [0,m) x [0,n)");
    if (!std::isfinite(v)) throw std::invalid_argument("update: value must be finite");
    update_unchecked(i, j, v);
  }

  // Batch of COO entries. Atomic with respect to bad input: all entries are validated first.
  // The apply loop is a pure unit-stride FMA stream per entry, with software prefetch of the
  // destination columns a few entries ahead (they are random-access into Yt_ and W_).
  void update_coo(const std::int64_t* rows, const std::int64_t* cols, const double* vals, std::size_t count) {
    if (count != 0 && (rows == nullptr || cols == nullptr || vals == nullptr))
      throw std::invalid_argument("update_coo: null input");
    for (std::size_t k = 0; k < count; ++k) {
      if (rows[k] < 0 || rows[k] >= m_ || cols[k] < 0 || cols[k] >= n_)
        throw std::out_of_range("update_coo: index outside [0,m) x [0,n)");
      if (!std::isfinite(vals[k])) throw std::invalid_argument("update_coo: value must be finite");
    }
    constexpr std::size_t kAhead = 8;
    for (std::size_t k = 0; k < count; ++k) {
      if (k + kAhead < count) {
        prefetch_columns(rows[k + kAhead], cols[k + kAhead]);
      }
      update_unchecked(static_cast<Index>(rows[k]), static_cast<Index>(cols[k]), vals[k]);
    }
  }

  // Sparse delta: best when nnz(D) is small relative to m*n.
  void update(const SpMat& D) {
    if (D.rows() != m_ || D.cols() != n_) throw std::invalid_argument("update: shape mismatch");
    for (Index j = 0; j < D.outerSize(); ++j)
      for (SpMat::InnerIterator it(D, j); it; ++it) update(it.row(), j, it.value());
  }

  // Dense delta block (full m x n): uses the fast FWHT path for both sketches.
  void update_dense(const MatrixXd& D) {
    if (D.rows() != m_ || D.cols() != n_) throw std::invalid_argument("update_dense: shape mismatch");
    if (!D.allFinite()) throw std::invalid_argument("update_dense: non-finite values");
    Yt_ += om_.apply(D).transpose();
    W_ += psi_.apply(D.transpose()).transpose();
  }

  // ---- merging ------------------------------------------------------------
  void merge(const StreamingLowRank& o) {
    if (const char* bad = detail::first_param_mismatch(header(0), o.header(0)))
      throw std::invalid_argument(std::string("incompatible sketches: parameter '") + bad + "' differs");
    Yt_ += o.Yt_;
    W_ += o.W_;
  }

  // Adds a serialized sketch straight from bytes (no intermediate object or copy).
  // Atomic: the checksum is verified before this sketch is modified.
  void merge_from_buffer(const std::uint8_t* buf, std::size_t size) {
    const BlobInfo h = verify_buffer(buf, size);
    if (const char* bad = detail::first_param_mismatch(header(0), h))
      throw std::invalid_argument(std::string("incompatible sketches: parameter '") + bad + "' differs");
    const std::size_t ny = static_cast<std::size_t>(Yt_.size());
    detail::add_from_bytes(Yt_.data(), buf + detail::kHeaderBytes, ny);
    detail::add_from_bytes(W_.data(), buf + detail::kHeaderBytes + ny * 8, static_cast<std::size_t>(W_.size()));
  }

  // out = a + b on raw blobs (what a SQL scalar function needs). nout must equal na; no aliasing.
  static void merge_buffers(const std::uint8_t* a, std::size_t na, const std::uint8_t* b, std::size_t nb,
                            std::uint8_t* out, std::size_t nout) {
    detail::merge_blobs(a, na, b, nb, out, nout);
  }

  // ---- binary serialization ----------------------------------------------
  std::size_t serialized_size() const {
    return detail::blob_bytes(header(0));
  }

  // Writes the blob directly into caller memory (e.g. a DuckDB string buffer). Returns bytes written.
  std::size_t serialize_to_buffer(std::uint8_t* dst, std::size_t capacity) const {
    const std::size_t need = serialized_size();
    if (dst == nullptr || capacity < need) throw std::length_error("serialize_to_buffer: buffer too small");
    const std::size_t ny = static_cast<std::size_t>(Yt_.size());
    const std::size_t nw = static_cast<std::size_t>(W_.size());
    std::uint8_t* p = dst + detail::kHeaderBytes;
    std::memcpy(p, Yt_.data(), ny * 8);
    std::memcpy(p + ny * 8, W_.data(), nw * 8);
    detail::Checksum cs;
    cs.doubles(Yt_.data(), ny);
    cs.doubles(W_.data(), nw);
    detail::write_header(dst, header(cs.digest()));
    return need;
  }

  std::vector<std::uint8_t> serialize() const {
    std::vector<std::uint8_t> out(serialized_size());
    serialize_to_buffer(out.data(), out.size());
    return out;
  }

  // Validates structure, exact length and the payload checksum. Throws std::invalid_argument /
  // std::runtime_error on bad input.
  static BlobInfo verify_buffer(const std::uint8_t* buf, std::size_t size) {
    const BlobInfo h = detail::read_header(buf, size);
    if (detail::checksum_bytes(buf + detail::kHeaderBytes, detail::payload_doubles(h)) != h.checksum)
      throw std::runtime_error("hsk blob: corrupt (checksum mismatch)");
    return h;
  }

  // Structure-only inspection (cheap; no checksum pass).
  static BlobInfo inspect_buffer(const std::uint8_t* buf, std::size_t size) {
    return detail::read_header(buf, size);
  }

  // Rebuilds a sketch from a blob: one memcpy of the payload straight into Eigen storage.
  static StreamingLowRank deserialize_from_buffer(const std::uint8_t* buf, std::size_t size) {
    const BlobInfo h = detail::read_header(buf, size);   // size == expected, so allocation <= blob size
    StreamingLowRank sk(h.m, h.n, h.rank, h.oversample, h.core_factor, h.c_factor, h.seed);
    const std::size_t ny = static_cast<std::size_t>(sk.Yt_.size());
    const std::size_t nw = static_cast<std::size_t>(sk.W_.size());
    std::memcpy(sk.Yt_.data(), buf + detail::kHeaderBytes, ny * 8);
    std::memcpy(sk.W_.data(), buf + detail::kHeaderBytes + ny * 8, nw * 8);
    detail::Checksum cs;
    cs.doubles(sk.Yt_.data(), ny);
    cs.doubles(sk.W_.data(), nw);
    if (cs.digest() != h.checksum) throw std::runtime_error("hsk blob: corrupt (checksum mismatch)");
    return sk;
  }

  static StreamingLowRank deserialize(const std::vector<std::uint8_t>& v) {
    return deserialize_from_buffer(v.data(), v.size());
  }

  // ---- factorization ------------------------------------------------------
  SvdResult finalize(Index k) const {
    if (k < 1) throw std::invalid_argument("finalize: k must be >= 1");
    MatrixXd Q = detail::orth(MatrixXd(Yt_.transpose()));             // m x l
    MatrixXd PQ = psi_.apply(Q.transpose()).transpose();              // s x l  (= Psi Q)
    Eigen::ColPivHouseholderQR<MatrixXd> qr(PQ);
    MatrixXd X = qr.solve(W_);                                        // l x n
    return detail::svd_from_Bt(Q, MatrixXd(X.transpose()), k);
  }

 private:
  StreamingLowRank(Index m, Index n, Index rank, Index oversample, double core_factor, double c_factor,
                   std::uint64_t seed, const detail::Dims& d)
      : m_(m), n_(n), rank_(rank), oversample_(oversample), core_factor_(core_factor),
        c_factor_(c_factor), seed_(seed), l_(static_cast<Index>(d.l)), s_(static_cast<Index>(d.s)),
        om_(n, l_, c_factor, seed),
        psi_(m, s_, c_factor, seed ^ 0xA5A5A5A5DEADBEEFULL),
        Yt_(MatrixXd::Zero(l_, m)),
        W_(MatrixXd::Zero(s_, n)) {}

  detail::BlobHeader header(std::uint64_t checksum) const {
    detail::BlobHeader h;
    h.m = m_; h.n = n_; h.rank = rank_; h.oversample = oversample_;
    h.l = l_; h.s = s_;
    h.core_factor = core_factor_; h.c_factor = c_factor_;
    h.seed = seed_; h.checksum = checksum;
    return h;
  }

  void update_unchecked(Index i, Index j, double v) {
    om_.add_scaled_bucket(om_.locate(j), v, Yt_.data() + i * l_);
    psi_.add_scaled_bucket(psi_.locate(i), v, W_.data() + j * s_);
  }

  void prefetch_columns(std::int64_t i, std::int64_t j) const {
    const double* y = Yt_.data() + static_cast<Index>(i) * l_;
    const double* w = W_.data() + static_cast<Index>(j) * s_;
    const Index ny = std::min<Index>(l_, 32), nw = std::min<Index>(s_, 32);   // up to 4 cache lines each
    for (Index t = 0; t < ny; t += 8) HSK_PREFETCH_W(y + t);
    for (Index t = 0; t < nw; t += 8) HSK_PREFETCH_W(w + t);
  }

  Index m_, n_, rank_, oversample_;
  double core_factor_, c_factor_;
  std::uint64_t seed_;
  Index l_, s_;
  HybridSketch om_, psi_;
  MatrixXd Yt_, W_;
};

}  // namespace hsk
