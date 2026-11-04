// hsk_duckdb_extension.cpp : DuckDB extension exposing the hybrid CountSketch+SRHT streaming
// low-rank sketch (hsk::StreamingLowRank) as SQL.
//
//   hsk_sketch_agg(row_idx BIGINT, col_idx BIGINT, val DOUBLE, m BIGINT, n BIGINT,
//                  rank INTEGER, seed UBIGINT) -> BLOB            (aggregate)
//   hsk_sketch_merge(a BLOB, b BLOB) -> BLOB                      (scalar)
//   hsk_sketch_svd(sketch BLOB, k INTEGER)                        (table function)
//        -> (component_idx INTEGER, singular_value DOUBLE, u_vec DOUBLE[], v_vec DOUBLE[])
//
// Semantics
//   * m, n, rank, seed must be constant expressions (checked at bind time); oversample, core
//     factor and CountSketch width use the library defaults (10, 4.0, 8.0) and are recorded in the blob.
//   * Rows with a NULL argument are skipped. NaN/Inf values and out-of-range indices are errors.
//   * Entries are additive: a negative `val` subtracts, so deletion/unlearning is exact (turnstile).
//     Duplicate (row_idx, col_idx) pairs sum.
//   * An aggregate over zero (non-NULL) rows returns NULL.
//   * Sketches from different groups/threads/tables merge iff (m, n, rank, seed) match.
//   * Memory: every aggregate state (per thread, per group) holds a full sketch of
//     8*(l*m + s*n) bytes. The per-sketch cap defaults to 1 GiB, is configurable with the
//     environment variable HSK_MAX_SKETCH_BYTES, and is hard-limited below 4 GiB (BLOB limit).
//     Prefer few groups with big sketches over many groups; SET threads=N bounds parallel copies.
//   * hsk_sketch_svd is an in/out table function, so it takes blob literals and column references
//     from other tables via (implicit or explicit) LATERAL joins:
//         SELECT s.* FROM t, LATERAL hsk_sketch_svd(t.sk, 5) s;
//     Passing a bare scalar subquery as the argument may be rejected by DuckDB's binder; use the
//     LATERAL form above. k larger than the sketch rank returns all available components.
//
// Targets DuckDB's C++ extension API: the DatabaseInstance/ExtensionUtil entry point (v1.1-v1.3)
// and, when "duckdb/main/extension/extension_loader.hpp" exists, the ExtensionLoader entry
// point (v1.4+). Pin your DuckDB submodule to a release tag and build via CMakeLists.txt.
#ifndef DUCKDB_EXTENSION_MAIN
#define DUCKDB_EXTENSION_MAIN
#endif

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/vector_operations/binary_executor.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "hsk_extension.hpp"
#if !HSK_USE_LOADER_API
#include "duckdb/main/extension_util.hpp"
#endif

#include "hybrid_sketch.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace duckdb {

namespace {

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
constexpr idx_t kHskDefaultMaxBytes = static_cast<idx_t>(1) << 30;           // 1 GiB per sketch
constexpr idx_t kHskHardMaxBytes = static_cast<idx_t>(3500) << 20;           // string_t holds < 4 GiB
constexpr idx_t kHskMaxListElementsPerChunk = static_cast<idx_t>(1) << 24;   // bound table-function output

idx_t HskMaxSketchBytes() {
	idx_t cap = kHskDefaultMaxBytes;
	const char *env = std::getenv("HSK_MAX_SKETCH_BYTES");
	if (env != nullptr && *env != '\0') {
		char *end = nullptr;
		const unsigned long long v = std::strtoull(env, &end, 10);
		if (end != nullptr && *end == '\0' && v > 0) {
			cap = static_cast<idx_t>(v);
		}
	}
	return std::min<idx_t>(cap, kHskHardMaxBytes);
}

// Runs f(); converts library exceptions into DuckDB exceptions with single-string messages
// (DuckDB's variadic exception constructors interpret format specifiers; ours never do).
template <class F>
auto HskGuard(const char *where, F &&f) -> decltype(f()) {
	try {
		return f();
	} catch (const Exception &) {
		throw;
	} catch (const std::bad_alloc &) {
		throw OutOfMemoryException(std::string(where) + ": out of memory while handling a sketch");
	} catch (const std::exception &e) {
		throw InvalidInputException(std::string(where) + ": " + e.what());
	}
}

// ---------------------------------------------------------------------------
// Aggregate: hsk_sketch_agg
// ---------------------------------------------------------------------------
struct HskAggBindData : public FunctionData {
	HskAggBindData(int64_t m_p, int64_t n_p, int32_t rank_p, uint64_t seed_p)
	    : m(m_p), n(n_p), rank(rank_p), seed(seed_p) {
	}
	int64_t m;
	int64_t n;
	int32_t rank;
	uint64_t seed;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<HskAggBindData>(m, n, rank, seed);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<HskAggBindData>();
		return m == o.m && n == o.n && rank == o.rank && seed == o.seed;
	}
};

struct HskState {
	hsk::StreamingLowRank *sketch;
};

unique_ptr<FunctionData> HskAggBind(ClientContext &context, AggregateFunction &, vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() != 7) {
		throw BinderException("hsk_sketch_agg expects 7 arguments");
	}
	auto constant = [&](idx_t i, const char *name) -> Value {
		if (!arguments[i]->IsFoldable()) {
			throw BinderException(std::string("hsk_sketch_agg: argument '") + name + "' must be a constant expression");
		}
		Value v = ExpressionExecutor::EvaluateScalar(context, *arguments[i]);
		if (v.IsNull()) {
			throw BinderException(std::string("hsk_sketch_agg: argument '") + name + "' must not be NULL");
		}
		return v;
	};
	const int64_t m = constant(3, "m").GetValue<int64_t>();
	const int64_t n = constant(4, "n").GetValue<int64_t>();
	const int32_t rank = constant(5, "rank").GetValue<int32_t>();
	const uint64_t seed = constant(6, "seed").GetValue<uint64_t>();

	std::size_t bytes = 0;
	try {
		bytes = hsk::StreamingLowRank::state_bytes(m, n, rank);
	} catch (const std::exception &e) {
		throw BinderException(std::string("hsk_sketch_agg: ") + e.what());
	}
	const idx_t cap = HskMaxSketchBytes();
	if (bytes > cap) {
		throw BinderException("hsk_sketch_agg: sketch would need " + std::to_string(bytes) + " bytes, above the limit of " +
		                      std::to_string(cap) + " bytes (raise HSK_MAX_SKETCH_BYTES or reduce m, n, rank)");
	}
	return make_uniq<HskAggBindData>(m, n, rank, seed);
}

idx_t HskAggStateSize(const AggregateFunction &) {
	return sizeof(HskState);
}

void HskAggInitialize(const AggregateFunction &, data_ptr_t state) {
	new (state) HskState {nullptr};
}

void HskAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = UnifiedVectorFormat::GetData<HskState *>(sdata);
	for (idx_t i = 0; i < count; i++) {
		HskState *st = states[sdata.sel->get_index(i)];
		delete st->sketch;
		st->sketch = nullptr;
	}
}

// Flushes a run of entries that all belong to the same aggregate state.
void HskApplyBatch(HskState &state, const HskAggBindData &bind, const std::vector<int64_t> &rows,
                   const std::vector<int64_t> &cols, const std::vector<double> &vals) {
	if (rows.empty()) {
		return;
	}
	HskGuard("hsk_sketch_agg", [&]() {
		if (state.sketch == nullptr) {
			state.sketch = new hsk::StreamingLowRank(bind.m, bind.n, bind.rank, hsk::StreamingLowRank::kDefaultOversample,
			                                         hsk::StreamingLowRank::kDefaultCoreFactor,
			                                         hsk::StreamingLowRank::kDefaultCFactor, bind.seed);
		}
		state.sketch->update_coo(rows.data(), cols.data(), vals.data(), rows.size());
	});
}

void HskAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, Vector &state_vector, idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<HskAggBindData>();
	UnifiedVectorFormat rf, cf, vf, sf;
	inputs[0].ToUnifiedFormat(count, rf);
	inputs[1].ToUnifiedFormat(count, cf);
	inputs[2].ToUnifiedFormat(count, vf);
	state_vector.ToUnifiedFormat(count, sf);
	auto rows = UnifiedVectorFormat::GetData<int64_t>(rf);
	auto cols = UnifiedVectorFormat::GetData<int64_t>(cf);
	auto vals = UnifiedVectorFormat::GetData<double>(vf);
	auto states = UnifiedVectorFormat::GetData<HskState *>(sf);

	std::vector<int64_t> rbuf, cbuf;
	std::vector<double> vbuf;
	rbuf.reserve(count);
	cbuf.reserve(count);
	vbuf.reserve(count);
	HskState *current = nullptr;

	for (idx_t i = 0; i < count; i++) {
		const idx_t ri = rf.sel->get_index(i);
		const idx_t ci = cf.sel->get_index(i);
		const idx_t vi = vf.sel->get_index(i);
		if (!rf.validity.RowIsValid(ri) || !cf.validity.RowIsValid(ci) || !vf.validity.RowIsValid(vi)) {
			continue; // NULL argument: skip the row
		}
		HskState *st = states[sf.sel->get_index(i)];
		if (st != current) { // ungrouped aggregates and sorted groups collapse into one batch
			if (current != nullptr) {
				HskApplyBatch(*current, bind, rbuf, cbuf, vbuf);
				rbuf.clear();
				cbuf.clear();
				vbuf.clear();
			}
			current = st;
		}
		rbuf.push_back(rows[ri]);
		cbuf.push_back(cols[ci]);
		vbuf.push_back(vals[vi]);
	}
	if (current != nullptr) {
		HskApplyBatch(*current, bind, rbuf, cbuf, vbuf);
	}
}

// source -> target. Source is copied, never consumed: DuckDB may reuse source states
// (e.g. window segment trees), so stealing the pointer would be unsafe.
void HskAggCombine(Vector &source, Vector &target, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sf, tf;
	source.ToUnifiedFormat(count, sf);
	target.ToUnifiedFormat(count, tf);
	auto src = UnifiedVectorFormat::GetData<HskState *>(sf);
	auto tgt = UnifiedVectorFormat::GetData<HskState *>(tf);
	HskGuard("hsk_sketch_agg", [&]() {
		for (idx_t i = 0; i < count; i++) {
			HskState *s = src[sf.sel->get_index(i)];
			HskState *t = tgt[tf.sel->get_index(i)];
			if (s->sketch == nullptr) {
				continue;
			}
			if (t->sketch == nullptr) {
				t->sketch = new hsk::StreamingLowRank(*s->sketch);
			} else {
				t->sketch->merge(*s->sketch); // parallel thread-local shard merge
			}
		}
	});
}

void HskAggFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
	UnifiedVectorFormat sf;
	state_vector.ToUnifiedFormat(count, sf);
	auto states = UnifiedVectorFormat::GetData<HskState *>(sf);
	auto out = FlatVector::GetData<string_t>(result);
	HskGuard("hsk_sketch_agg", [&]() {
		for (idx_t i = 0; i < count; i++) {
			HskState *st = states[sf.sel->get_index(i)];
			if (st->sketch == nullptr) {
				FlatVector::SetNull(result, i + offset, true);
				continue;
			}
			const std::size_t size = st->sketch->serialized_size();
			string_t blob = StringVector::EmptyString(result, size);
			st->sketch->serialize_to_buffer(reinterpret_cast<uint8_t *>(blob.GetDataWriteable()), size);
			blob.Finalize();
			out[i + offset] = blob;
		}
	});
}

AggregateFunctionSet MakeSketchAggregate() {
	AggregateFunction fn("hsk_sketch_agg",
	                     {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::DOUBLE, LogicalType::BIGINT,
	                      LogicalType::BIGINT, LogicalType::INTEGER, LogicalType::UBIGINT},
	                     LogicalType::BLOB, HskAggStateSize, HskAggInitialize, HskAggUpdate, HskAggCombine,
	                     HskAggFinalize, FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr, HskAggBind,
	                     HskAggDestroy);
	AggregateFunctionSet set("hsk_sketch_agg");
	set.AddFunction(fn);
	return set;
}

// ---------------------------------------------------------------------------
// Scalar: hsk_sketch_merge(blob, blob) -> blob
//   Byte-level merge: one fused pass verifies both checksums, adds, and writes the result.
//   No sketch objects are constructed.
// ---------------------------------------------------------------------------
void HskMergeFunction(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::Execute<string_t, string_t, string_t>(
	    args.data[0], args.data[1], result, args.size(), [&](string_t a, string_t b) {
		    return HskGuard("hsk_sketch_merge", [&]() {
			    const std::size_t na = a.GetSize();
			    string_t out = StringVector::EmptyString(result, na);
			    hsk::StreamingLowRank::merge_buffers(reinterpret_cast<const uint8_t *>(a.GetData()), na,
			                                         reinterpret_cast<const uint8_t *>(b.GetData()), b.GetSize(),
			                                         reinterpret_cast<uint8_t *>(out.GetDataWriteable()), na);
			    out.Finalize();
			    return out;
		    });
	    });
}

ScalarFunction MakeSketchMerge() {
	return ScalarFunction("hsk_sketch_merge", {LogicalType::BLOB, LogicalType::BLOB}, LogicalType::BLOB, HskMergeFunction);
}

// ---------------------------------------------------------------------------
// Table function: hsk_sketch_svd(blob, k)   (in/out function: works with LATERAL)
// ---------------------------------------------------------------------------
struct HskSvdBindData : public TableFunctionData {};

struct HskSvdLocalState : public LocalTableFunctionState {
	idx_t row = 0;            // current input row
	bool loaded = false;      // result holds the factors of `row`
	idx_t next_component = 0; // next component of `result` to emit
	hsk::SvdResult result;
};

unique_ptr<FunctionData> HskSvdBind(ClientContext &, TableFunctionBindInput &, vector<LogicalType> &return_types,
                                    vector<string> &names) {
	names = {"component_idx", "singular_value", "u_vec", "v_vec"};
	return_types = {LogicalType::INTEGER, LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE),
	                LogicalType::LIST(LogicalType::DOUBLE)};
	return make_uniq<HskSvdBindData>();
}

unique_ptr<GlobalTableFunctionState> HskSvdInitGlobal(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<GlobalTableFunctionState>();
}

unique_ptr<LocalTableFunctionState> HskSvdInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                    GlobalTableFunctionState *) {
	return make_uniq<HskSvdLocalState>();
}

OperatorResultType HskSvdInOut(ExecutionContext &, TableFunctionInput &data, DataChunk &input, DataChunk &output) {
	auto &st = data.local_state->Cast<HskSvdLocalState>();
	const idx_t in_count = input.size();

	UnifiedVectorFormat bf, kf;
	input.data[0].ToUnifiedFormat(in_count, bf);
	input.data[1].ToUnifiedFormat(in_count, kf);
	auto blobs = UnifiedVectorFormat::GetData<string_t>(bf);
	auto ks = UnifiedVectorFormat::GetData<int32_t>(kf);

	auto comp_data = FlatVector::GetData<int32_t>(output.data[0]);
	auto sv_data = FlatVector::GetData<double>(output.data[1]);
	Vector &u_vec = output.data[2];
	Vector &v_vec = output.data[3];
	auto u_entries = FlatVector::GetData<list_entry_t>(u_vec);
	auto v_entries = FlatVector::GetData<list_entry_t>(v_vec);
	idx_t u_size = ListVector::GetListSize(u_vec);
	idx_t v_size = ListVector::GetListSize(v_vec);
	idx_t produced = 0;

	while (produced < STANDARD_VECTOR_SIZE) {
		if (!st.loaded) {
			while (st.row < in_count) { // skip rows with a NULL blob or NULL k
				if (bf.validity.RowIsValid(bf.sel->get_index(st.row)) && kf.validity.RowIsValid(kf.sel->get_index(st.row))) {
					break;
				}
				st.row++;
			}
			if (st.row >= in_count) {
				break;
			}
			const int32_t k = ks[kf.sel->get_index(st.row)];
			if (k < 1) {
				throw InvalidInputException("hsk_sketch_svd: k must be >= 1");
			}
			const string_t blob = blobs[bf.sel->get_index(st.row)];
			HskGuard("hsk_sketch_svd", [&]() {
				hsk::StreamingLowRank sk = hsk::StreamingLowRank::deserialize_from_buffer(
				    reinterpret_cast<const uint8_t *>(blob.GetData()), blob.GetSize());
				st.result = sk.finalize(k);
			});
			st.next_component = 0;
			st.loaded = true;
		}

		const idx_t total = static_cast<idx_t>(st.result.S.size());
		const idx_t m = static_cast<idx_t>(st.result.U.rows());
		const idx_t n = static_cast<idx_t>(st.result.V.rows());
		while (st.next_component < total && produced < STANDARD_VECTOR_SIZE) {
			if (produced > 0 && (u_size + m > kHskMaxListElementsPerChunk || v_size + n > kHskMaxListElementsPerChunk)) {
				break; // keep per-chunk memory bounded; resume on the next call
			}
			const Eigen::Index c = static_cast<Eigen::Index>(st.next_component);

			comp_data[produced] = static_cast<int32_t>(st.next_component);
			sv_data[produced] = st.result.S(c);

			ListVector::Reserve(u_vec, u_size + m);
			double *u_child = FlatVector::GetData<double>(ListVector::GetEntry(u_vec));
			std::memcpy(u_child + u_size, st.result.U.col(c).data(), m * sizeof(double));
			u_entries[produced] = list_entry_t(u_size, m);
			u_size += m;

			ListVector::Reserve(v_vec, v_size + n);
			double *v_child = FlatVector::GetData<double>(ListVector::GetEntry(v_vec));
			std::memcpy(v_child + v_size, st.result.V.col(c).data(), n * sizeof(double));
			v_entries[produced] = list_entry_t(v_size, n);
			v_size += n;

			produced++;
			st.next_component++;
		}
		if (st.next_component >= total) {
			st.loaded = false;
			st.result = hsk::SvdResult(); // release factor memory
			st.row++;
		} else {
			break; // output chunk full
		}
	}

	ListVector::SetListSize(u_vec, u_size);
	ListVector::SetListSize(v_vec, v_size);
	output.SetCardinality(produced);

	if (!st.loaded && st.row >= in_count) {
		st.row = 0;
		return OperatorResultType::NEED_MORE_INPUT;
	}
	return OperatorResultType::HAVE_MORE_OUTPUT;
}

TableFunction MakeSketchSvd() {
	TableFunction fn("hsk_sketch_svd", {LogicalType::BLOB, LogicalType::INTEGER}, nullptr, HskSvdBind, HskSvdInitGlobal,
	                 HskSvdInitLocal);
	fn.in_out_function = HskSvdInOut;
	return fn;
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------
#if HSK_USE_LOADER_API
using HskRegistrar = ExtensionLoader;
template <class F>
void HskRegister(HskRegistrar &registrar, F fn) {
	registrar.RegisterFunction(std::move(fn));
}
#else
using HskRegistrar = DatabaseInstance;
template <class F>
void HskRegister(HskRegistrar &registrar, F fn) {
	ExtensionUtil::RegisterFunction(registrar, std::move(fn));
}
#endif

void HskLoadInternal(HskRegistrar &registrar) {
	HskRegister(registrar, MakeSketchAggregate());
	HskRegister(registrar, MakeSketchMerge());
	HskRegister(registrar, MakeSketchSvd());
}

} // namespace

#if HSK_USE_LOADER_API
void HskExtension::Load(ExtensionLoader &loader) {
	HskLoadInternal(loader);
}
#else
void HskExtension::Load(DuckDB &db) {
	HskLoadInternal(*db.instance);
}
#endif

std::string HskExtension::Name() {
	return "hsk";
}

std::string HskExtension::Version() const {
#ifdef EXT_VERSION_HSK
	return EXT_VERSION_HSK;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

#if HSK_USE_LOADER_API
DUCKDB_CPP_EXTENSION_ENTRY(hsk, loader) {
	duckdb::HskExtension ext;
	ext.Load(loader);
}
#else
DUCKDB_EXTENSION_API void hsk_init(duckdb::DatabaseInstance &db) {
	duckdb::DuckDB db_wrapper(db);
	db_wrapper.LoadExtension<duckdb::HskExtension>();
}

DUCKDB_EXTENSION_API const char *hsk_version() {
	return duckdb::DuckDB::LibraryVersion();
}
#endif
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
