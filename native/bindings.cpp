// native/bindings.cpp : pybind11 module `hsk_native`.
//
// NOT COMPILED. The sandbox could not reach Eigen or pybind11.
//
// hybrid_sketch.hpp's StreamingLowRank owns its Yt_/W_ MatrixXd, so it cannot attach to external
// memory. This file therefore defines `Sketch` (exported to Python as `StreamingLowRank`). It reuses
// the public HybridSketch / detail:: helpers from the header and replicates the StreamingLowRank
// math, with storage as raw double* that is either owned (std::vector) or borrowed from a NumPy array.
// It must stay in sync with the header on these points:
//   * Omega = HybridSketch(n, l, c_factor, seed); Psi = HybridSketch(m, s, c_factor, seed ^ 0xA5A5A5A5DEADBEEF)
//   * Yt is l x m and W is s x n, both column-major, serialized as Yt then W after the 88-byte HSK1 header.
// Blobs are byte-compatible with the C++ StreamingLowRank and the DuckDB extension (same header,
// same checksum stream order).
//
// Zero-copy mode: pass `yt` and `w` as float64 arrays of shape (l, m) and (s, n), Fortran-ordered and
// writeable (e.g. np.memmap(..., order="F")). Updates then write directly into that memory.
// The arrays are validated, never copied, and kept alive by the sketch.
//
// Thread-safety: not thread-safe. Do not call mutating methods concurrently on one sketch. Long
// operations release the GIL.
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "hybrid_sketch.hpp"

namespace py = pybind11;
namespace hd = hsk::detail;
using hsk::Index;
using hsk::MatrixXd;
using hsk::SpMat;
using hsk::SvdResult;

using IdxArr = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
using ValArr = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

constexpr std::uint64_t kPsiSalt = 0xA5A5A5A5DEADBEEFULL;

// ---------------------------------------------------------------------------
// Process-wide cache of immutable test matrices, keyed by (n, l, c_factor, seed).
// Sketches with equal parameters share one HybridSketch (and its sign table).
// ---------------------------------------------------------------------------
using SketchPtr = std::shared_ptr<const hsk::HybridSketch>;

SketchPtr cached_sketch(Index n, Index l, double c_factor, std::uint64_t seed) {
  static std::mutex mu;
  static std::map<std::tuple<Index, Index, double, std::uint64_t>, SketchPtr> cache;
  std::lock_guard<std::mutex> guard(mu);
  const auto key = std::make_tuple(n, l, c_factor, seed);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  if (cache.size() >= 8) cache.clear();  // live sketches keep their own shared_ptr
  SketchPtr sp = std::make_shared<const hsk::HybridSketch>(n, l, c_factor, seed);
  cache.emplace(key, sp);
  return sp;
}

// Read-only/writable view of a 1-D contiguous byte buffer (bytes, memoryview, mmap, uint8 array).
struct BufView {
  py::buffer_info info;
  const std::uint8_t* data = nullptr;
  std::size_t size = 0;
  explicit BufView(const py::buffer& b, bool writable = false) : info(b.request(writable)) {
    if (info.ndim != 1 || info.strides[0] != info.itemsize)
      throw std::invalid_argument("expected a contiguous 1-D byte buffer (bytes, memoryview, uint8 array)");
    data = static_cast<const std::uint8_t*>(info.ptr);
    size = static_cast<std::size_t>(info.size) * static_cast<std::size_t>(info.itemsize);
  }
};

py::array view2d(py::object base, double* p, Index rows, Index cols) {
  py::array a(py::dtype::of<double>(),
              std::vector<py::ssize_t>{static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)},
              std::vector<py::ssize_t>{static_cast<py::ssize_t>(sizeof(double)),
                                       static_cast<py::ssize_t>(rows * sizeof(double))},
              p, base);
  a.attr("setflags")(py::arg("write") = false);  // state is mutated only through the sketch API
  return a;
}

// ---------------------------------------------------------------------------
// Sketch : single-pass, turnstile, mergeable low-rank sketch with optional external storage.
// ---------------------------------------------------------------------------
class Sketch {
 public:
  Sketch(Index m, Index n, Index rank, Index oversample, double core_factor, double c_factor,
         std::uint64_t seed, py::object yt, py::object w)
      : m_(m), n_(n), rank_(rank), oversample_(oversample), core_(core_factor), cf_(c_factor), seed_(seed) {
    const hd::Dims d = hd::make_dims(m, n, rank, oversample, core_factor, c_factor);
    l_ = static_cast<Index>(d.l);
    s_ = static_cast<Index>(d.s);
    (void)hd::blob_bytes(header(0));  // throws std::overflow_error on absurd sizes
    om_ = cached_sketch(n_, l_, cf_, seed_);
    psi_ = cached_sketch(m_, s_, cf_, seed_ ^ kPsiSalt);

    const std::size_t ny = static_cast<std::size_t>(l_) * static_cast<std::size_t>(m_);
    const std::size_t nw = static_cast<std::size_t>(s_) * static_cast<std::size_t>(n_);
    const bool has_yt = !yt.is_none(), has_w = !w.is_none();
    if (has_yt != has_w) throw std::invalid_argument("pass both yt and w, or neither");
    if (has_yt) {
      yt_ = attach(yt, l_, m_, "yt");
      w_ = attach(w, s_, n_, "w");
      const auto a0 = reinterpret_cast<std::uintptr_t>(yt_), a1 = a0 + ny * 8;
      const auto b0 = reinterpret_cast<std::uintptr_t>(w_), b1 = b0 + nw * 8;
      if (a0 < b1 && b0 < a1) throw std::invalid_argument("yt and w must not overlap in memory");
      yt_ref_ = yt;
      w_ref_ = w;
      attached_ = true;
    } else {
      own_.assign(ny + nw, 0.0);
      yt_ = own_.data();
      w_ = own_.data() + ny;
    }
  }
  Sketch(const Sketch&) = delete;
  Sketch& operator=(const Sketch&) = delete;

  // ---- parameters ----
  Index m() const { return m_; }
  Index n() const { return n_; }
  Index rank() const { return rank_; }
  Index oversample() const { return oversample_; }
  double core_factor() const { return core_; }
  double c_factor() const { return cf_; }
  std::uint64_t seed() const { return seed_; }
  Index sketch_rank() const { return l_; }
  Index core_size() const { return s_; }
  bool attached() const { return attached_; }
  std::size_t blob_size() const { return hd::blob_bytes(header(0)); }

  double* yt_ptr() { return yt_; }
  double* w_ptr() { return w_; }

  // ---- updates ----
  void update(Index i, Index j, double v) {
    if (i < 0 || i >= m_ || j < 0 || j >= n_) throw std::out_of_range("update: index outside [0,m) x [0,n)");
    if (!std::isfinite(v)) throw std::invalid_argument("update: value must be finite");
    apply_unchecked(i, j, v);
  }

  // Atomic with respect to bad input: every entry is validated before any is applied.
  void update_coo(const IdxArr& r, const IdxArr& c, const ValArr& v) {
    if (r.ndim() != 1 || c.ndim() != 1 || v.ndim() != 1 || r.shape(0) != c.shape(0) || r.shape(0) != v.shape(0))
      throw std::invalid_argument("rows/cols/vals must be 1-D arrays of equal length");
    const std::size_t cnt = static_cast<std::size_t>(v.shape(0));
    const std::int64_t* R = r.data();
    const std::int64_t* C = c.data();
    const double* V = v.data();
    for (std::size_t k = 0; k < cnt; ++k) {
      if (R[k] < 0 || R[k] >= m_ || C[k] < 0 || C[k] >= n_)
        throw std::out_of_range("update_coo: index outside [0,m) x [0,n)");
      if (!std::isfinite(V[k])) throw std::invalid_argument("update_coo: value must be finite");
    }
    std::unique_ptr<py::gil_scoped_release> release;
    if (cnt >= 256) release.reset(new py::gil_scoped_release());  // not worth it for tiny batches
    for (std::size_t k = 0; k < cnt; ++k) apply_unchecked(R[k], C[k], V[k]);
  }

  void update_dense(const Eigen::Ref<const MatrixXd>& D) {
    if (D.rows() != m_ || D.cols() != n_) throw std::invalid_argument("update_dense: shape mismatch");
    if (!D.allFinite()) throw std::invalid_argument("update_dense: non-finite values");
    MatrixXd dy, dw;
    {
      py::gil_scoped_release release;
      dy = om_->apply(D);                  // m x l
      MatrixXd Dt = D.transpose();         // n x m
      dw = psi_->apply(Dt);                // n x s
    }
    yt_map() += dy.transpose();
    w_map() += dw.transpose();
  }

  void clear() {
    yt_map().setZero();
    w_map().setZero();
  }

  void set_state(const Eigen::Ref<const MatrixXd>& yt, const Eigen::Ref<const MatrixXd>& w) {
    if (yt.rows() != l_ || yt.cols() != m_)
      throw std::invalid_argument("set_state: yt must be " + std::to_string(l_) + " x " + std::to_string(m_));
    if (w.rows() != s_ || w.cols() != n_)
      throw std::invalid_argument("set_state: w must be " + std::to_string(s_) + " x " + std::to_string(n_));
    if (!yt.allFinite() || !w.allFinite()) throw std::invalid_argument("set_state: non-finite values");
    yt_map() = yt;
    w_map() = w;
  }

  // ---- merging ----
  void merge(const Sketch& o) {
    if (const char* bad = hd::first_param_mismatch(header(0), o.header(0)))
      throw std::invalid_argument(std::string("incompatible sketches: parameter '") + bad + "' differs");
    yt_map() += o.yt_c();
    w_map() += o.w_c();
  }

  // Adds a serialized sketch straight from bytes. Atomic: checksum and params are checked first.
  void merge_from_buffer(const py::buffer& b) {
    BufView v(b);
    const hd::BlobHeader h = verify(v.data, v.size);
    const std::size_t ny = static_cast<std::size_t>(l_) * static_cast<std::size_t>(m_);
    hd::add_from_bytes(yt_, v.data + hd::kHeaderBytes, ny);
    hd::add_from_bytes(w_, v.data + hd::kHeaderBytes + ny * 8, static_cast<std::size_t>(s_) * static_cast<std::size_t>(n_));
    (void)h;
  }

  // ---- serialization (HSK1 blob) ----
  py::bytes serialize() const {
    const std::size_t need = blob_size();
    PyObject* o = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(need));
    if (o == nullptr) throw py::error_already_set();
    py::bytes out = py::reinterpret_steal<py::bytes>(o);
    write_blob(reinterpret_cast<std::uint8_t*>(PyBytes_AS_STRING(o)));
    return out;
  }

  std::size_t serialize_into(const py::buffer& b) const {
    BufView v(b, /*writable=*/true);
    const std::size_t need = blob_size();
    if (v.size < need) throw std::length_error("serialize_into: buffer too small (need " + std::to_string(need) + " bytes)");
    write_blob(const_cast<std::uint8_t*>(v.data));
    return need;
  }

  // Replaces this sketch's state with the blob's. Params must match; the checksum is verified first.
  void load_from_buffer(const py::buffer& b) {
    BufView v(b);
    verify(v.data, v.size);
    const std::size_t ny = static_cast<std::size_t>(l_) * static_cast<std::size_t>(m_);
    std::memcpy(yt_, v.data + hd::kHeaderBytes, ny * 8);
    std::memcpy(w_, v.data + hd::kHeaderBytes + ny * 8, static_cast<std::size_t>(s_) * static_cast<std::size_t>(n_) * 8);
  }

  static std::unique_ptr<Sketch> from_buffer(const py::buffer& b) {
    BufView v(b);
    const hd::BlobHeader h = hd::read_header(v.data, v.size);  // checks exact length => bounded allocation
    auto s = std::make_unique<Sketch>(h.m, h.n, h.rank, h.oversample, h.core_factor, h.c_factor, h.seed,
                                      py::none(), py::none());
    s->load_from_buffer(b);
    return s;
  }

  static py::dict inspect(const py::buffer& b) {
    BufView v(b);
    const hd::BlobHeader h = hd::read_header(v.data, v.size);
    py::dict d;
    d["m"] = h.m; d["n"] = h.n; d["rank"] = h.rank; d["oversample"] = h.oversample;
    d["l"] = h.l; d["s"] = h.s; d["core_factor"] = h.core_factor; d["c_factor"] = h.c_factor;
    d["seed"] = h.seed; d["checksum"] = h.checksum;
    return d;
  }

  // ---- factorization ----
  SvdResult finalize(Index k) const {
    if (k < 1) throw std::invalid_argument("finalize: k must be >= 1");
    k = std::min<Index>(k, std::min(rank_, l_));  // components past `rank` are mostly noise
    MatrixXd Q = hd::orth(MatrixXd(yt_c().transpose()));          // m x l
    MatrixXd PQ = psi_->apply(Q.transpose()).transpose();         // s x l  (= Psi Q)
    Eigen::ColPivHouseholderQR<MatrixXd> qr(PQ);
    MatrixXd X = qr.solve(MatrixXd(w_c()));                       // l x n
    return hd::svd_from_Bt(Q, MatrixXd(X.transpose()), k);
  }

 private:
  Eigen::Map<MatrixXd> yt_map() { return Eigen::Map<MatrixXd>(yt_, l_, m_); }
  Eigen::Map<MatrixXd> w_map() { return Eigen::Map<MatrixXd>(w_, s_, n_); }
  Eigen::Map<const MatrixXd> yt_c() const { return Eigen::Map<const MatrixXd>(yt_, l_, m_); }
  Eigen::Map<const MatrixXd> w_c() const { return Eigen::Map<const MatrixXd>(w_, s_, n_); }

  void apply_unchecked(Index i, Index j, double v) {
    om_->add_scaled_bucket(om_->locate(j), v, yt_ + i * l_);
    psi_->add_scaled_bucket(psi_->locate(i), v, w_ + j * s_);
  }

  hd::BlobHeader header(std::uint64_t checksum) const {
    hd::BlobHeader h;
    h.m = m_; h.n = n_; h.rank = rank_; h.oversample = oversample_;
    h.l = l_; h.s = s_;
    h.core_factor = core_; h.c_factor = cf_;
    h.seed = seed_; h.checksum = checksum;
    return h;
  }

  void write_blob(std::uint8_t* dst) const {
    const std::size_t ny = static_cast<std::size_t>(l_) * static_cast<std::size_t>(m_);
    const std::size_t nw = static_cast<std::size_t>(s_) * static_cast<std::size_t>(n_);
    std::memcpy(dst + hd::kHeaderBytes, yt_, ny * 8);
    std::memcpy(dst + hd::kHeaderBytes + ny * 8, w_, nw * 8);
    hd::Checksum cs;
    cs.doubles(yt_, ny);
    cs.doubles(w_, nw);
    hd::write_header(dst, header(cs.digest()));
  }

  // Structure + params + checksum.
  hd::BlobHeader verify(const std::uint8_t* buf, std::size_t size) const {
    const hd::BlobHeader h = hd::read_header(buf, size);
    if (const char* bad = hd::first_param_mismatch(header(0), h))
      throw std::invalid_argument(std::string("incompatible sketches: parameter '") + bad + "' differs");
    if (hd::checksum_bytes(buf + hd::kHeaderBytes, hd::payload_doubles(h)) != h.checksum)
      throw std::runtime_error("hsk blob: corrupt (checksum mismatch)");
    return h;
  }

  static double* attach(const py::object& obj, Index rows, Index cols, const char* name) {
    const std::string nm(name);
    const std::string want = "(" + std::to_string(rows) + ", " + std::to_string(cols) + ")";
    if (!py::isinstance<py::array_t<double>>(obj)) throw std::invalid_argument(nm + " must be a float64 numpy array");
    py::array a = py::reinterpret_borrow<py::array>(obj);
    if (a.ndim() != 2 || a.shape(0) != rows || a.shape(1) != cols)
      throw std::invalid_argument(nm + " must have shape " + want);
    if (!(a.flags() & py::array::f_style))
      throw std::invalid_argument(nm + " must be Fortran-ordered, e.g. np.memmap(..., shape=" + want + ", order='F')");
    if (!a.writeable()) throw std::invalid_argument(nm + " must be writeable (use mode 'r+' or 'w+')");
    double* p = static_cast<double*>(a.mutable_data());
    if (reinterpret_cast<std::uintptr_t>(p) % alignof(double) != 0)
      throw std::invalid_argument(nm + " must be 8-byte aligned");
    return p;
  }

  Index m_, n_, rank_, oversample_;
  double core_, cf_;
  std::uint64_t seed_;
  Index l_ = 0, s_ = 0;
  SketchPtr om_, psi_;
  std::vector<double> own_;
  py::object yt_ref_, w_ref_;
  double* yt_ = nullptr;
  double* w_ = nullptr;
  bool attached_ = false;
};

// ---------------------------------------------------------------------------
// Two-pass randomized SVD helpers (unchanged behaviour from the original bindings).
// ---------------------------------------------------------------------------
void check_coo(Index m, Index n, const IdxArr& r, const IdxArr& c, const ValArr& v) {
  if (r.ndim() != 1 || c.ndim() != 1 || v.ndim() != 1 || r.shape(0) != c.shape(0) || r.shape(0) != v.shape(0))
    throw std::invalid_argument("rows/cols/vals must be 1-D arrays of equal length");
  auto R = r.unchecked<1>();
  auto C = c.unchecked<1>();
  for (py::ssize_t k = 0; k < r.shape(0); ++k)
    if (R(k) < 0 || R(k) >= m || C(k) < 0 || C(k) >= n) throw std::out_of_range("COO index outside matrix shape");
}

SpMat coo_to_sparse(Index m, Index n, const IdxArr& r, const IdxArr& c, const ValArr& v) {
  check_coo(m, n, r, c, v);
  auto R = r.unchecked<1>();
  auto C = c.unchecked<1>();
  auto V = v.unchecked<1>();
  std::vector<Eigen::Triplet<double>> t;
  t.reserve(static_cast<std::size_t>(v.shape(0)));
  for (py::ssize_t k = 0; k < v.shape(0); ++k) t.emplace_back(static_cast<int>(R(k)), static_cast<int>(C(k)), V(k));
  SpMat A(m, n);
  A.setFromTriplets(t.begin(), t.end());  // duplicates are summed
  return A;
}

hsk::RsvdOptions make_opts(Index rank, Index oversample, int q, double cf, std::uint64_t seed) {
  hsk::RsvdOptions o;
  o.rank = rank; o.oversample = oversample; o.power_iters = q; o.c_factor = cf; o.seed = seed;
  return o;
}

}  // namespace

PYBIND11_MODULE(hsk_native, mod) {
  mod.doc() = "Eigen CountSketch+SRHT low-rank engine (HSK1 blobs, zero-copy memmap state)";
  mod.attr("BLOB_HEADER_BYTES") = static_cast<std::size_t>(hd::kHeaderBytes);

  mod.def("sketch_dims",
          [](Index m, Index n, Index rank, Index oversample, double core_factor, double c_factor) {
            const hd::Dims d = hd::make_dims(m, n, rank, oversample, core_factor, c_factor);
            return py::make_tuple(d.l, d.s);
          },
          py::arg("m"), py::arg("n"), py::arg("rank"), py::arg("oversample") = 10, py::arg("core_factor") = 4.0,
          py::arg("c_factor") = 8.0);

  py::class_<Sketch>(mod, "StreamingLowRank")
      .def(py::init([](Index m, Index n, Index rank, Index oversample, double core_factor, double c_factor,
                       std::uint64_t seed, py::object yt, py::object w) {
             return std::make_unique<Sketch>(m, n, rank, oversample, core_factor, c_factor, seed, yt, w);
           }),
           py::arg("m"), py::arg("n"), py::arg("rank"), py::arg("oversample") = 10, py::arg("core_factor") = 4.0,
           py::arg("c_factor") = 8.0, py::arg("seed") = static_cast<std::uint64_t>(0xC0FFEEULL),
           py::arg("yt") = py::none(), py::arg("w") = py::none())
      .def_property_readonly("m", &Sketch::m)
      .def_property_readonly("n", &Sketch::n)
      .def_property_readonly("rank", &Sketch::rank)
      .def_property_readonly("oversample", &Sketch::oversample)
      .def_property_readonly("core_factor", &Sketch::core_factor)
      .def_property_readonly("c_factor", &Sketch::c_factor)
      .def_property_readonly("seed", &Sketch::seed)
      .def_property_readonly("sketch_rank", &Sketch::sketch_rank)
      .def_property_readonly("core_size", &Sketch::core_size)
      .def_property_readonly("attached", &Sketch::attached)
      .def_property_readonly("blob_size", &Sketch::blob_size)
      // Read-only zero-copy views of the live state: yt is (l x m), w is (s x n), Fortran-ordered.
      .def_property_readonly("yt", [](py::object self) {
        Sketch& s = self.cast<Sketch&>();
        return view2d(self, s.yt_ptr(), s.sketch_rank(), s.m());
      })
      .def_property_readonly("w", [](py::object self) {
        Sketch& s = self.cast<Sketch&>();
        return view2d(self, s.w_ptr(), s.core_size(), s.n());
      })
      .def("update", &Sketch::update, py::arg("i"), py::arg("j"), py::arg("v"))
      .def("update_coo", &Sketch::update_coo, py::arg("rows"), py::arg("cols"), py::arg("vals"))
      .def("update_dense", &Sketch::update_dense, py::arg("D"))
      .def("merge", &Sketch::merge, py::arg("other"))
      .def("merge_from_buffer", &Sketch::merge_from_buffer, py::arg("buf"))
      .def("clear", &Sketch::clear)
      .def("set_state", &Sketch::set_state, py::arg("yt"), py::arg("w"))
      .def("finalize",
           [](const Sketch& s, Index k) {
             SvdResult r;
             {
               py::gil_scoped_release release;
               r = s.finalize(k);
             }
             return py::make_tuple(r.U, r.S, r.V);
           },
           py::arg("k"))
      .def("serialize", &Sketch::serialize)
      .def("serialize_into", &Sketch::serialize_into, py::arg("buf"))
      .def("load_from_buffer", &Sketch::load_from_buffer, py::arg("buf"))
      .def_static("deserialize_from_buffer", &Sketch::from_buffer, py::arg("buf"))
      .def_static("inspect_buffer", &Sketch::inspect, py::arg("buf"));

  mod.def("rsvd_coo",
          [](Index m, Index n, const IdxArr& r, const IdxArr& c, const ValArr& v, Index rank, Index oversample,
             int power_iters, double c_factor, std::uint64_t seed) {
            SpMat A = coo_to_sparse(m, n, r, c, v);
            SvdResult res;
            double cert;
            {
              py::gil_scoped_release release;
              res = hsk::rsvd(A, make_opts(rank, oversample, power_iters, c_factor, seed));
              cert = hsk::residual_norm_estimate(A, res.U);
            }
            return py::make_tuple(res.U, res.S, res.V, cert);
          });

  mod.def("rsvd_dense",
          [](const MatrixXd& A, Index rank, Index oversample, int power_iters, double c_factor, std::uint64_t seed) {
            SvdResult res;
            double cert;
            {
              py::gil_scoped_release release;
              res = hsk::rsvd(A, make_opts(rank, oversample, power_iters, c_factor, seed));
              cert = hsk::residual_norm_estimate(A, res.U);
            }
            return py::make_tuple(res.U, res.S, res.V, cert);
          });
}