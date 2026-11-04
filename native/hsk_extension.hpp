// native/hsk_extension.hpp : class the DuckDB generated static loader includes as "hsk_extension.hpp".
// Implementations live in hsk_duckdb_extension.cpp. CMake must pass INCLUDE_DIR .../native.
#pragma once

#include "duckdb.hpp"

#ifndef HSK_USE_LOADER_API
#if __has_include("duckdb/main/extension/extension_loader.hpp")
#include "duckdb/main/extension/extension_loader.hpp"
#define HSK_USE_LOADER_API 1
#else
#define HSK_USE_LOADER_API 0
#endif
#endif

namespace duckdb {

class HskExtension : public Extension {
public:
#if HSK_USE_LOADER_API
	void Load(ExtensionLoader &loader) override;
#else
	void Load(DuckDB &db) override;
#endif
	std::string Name() override;
	std::string Version() const override;
};

} // namespace duckdb