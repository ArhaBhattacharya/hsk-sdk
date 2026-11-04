# hsk: CountSketch+SRHT streaming low-rank engine.
#
#   hsk::sdk          header-only C++17 SDK  (native/hybrid_sketch.hpp, needs Eigen >= 3.3)
#   hsk (extension)   DuckDB loadable + static extension (native/hsk_duckdb_extension.cpp)
#
# ---------------------------------------------------------------------------------------
# A) SDK only (no DuckDB):
#      cmake -S . -B build && cmake --build build
#      target_link_libraries(my_app PRIVATE hsk::sdk)
#
# B) DuckDB extension, driven from this project (recommended):
#      git clone --branch <release-tag> https://github.com/duckdb/duckdb.git ~/duckdb
#      cmake -S . -B build -DHSK_DUCKDB_SOURCE_DIR=$HOME/duckdb
#      cmake --build build --target duckdb_hsk
#      -> build/duckdb-build/extension/hsk/hsk.duckdb_extension   (+ a duckdb CLI with hsk linked in)
#      Load an unsigned local build with:   duckdb -unsigned   then   LOAD '/path/to/hsk.duckdb_extension';
#
# C) DuckDB extension, driven by DuckDB's own build (extension-template style). In a config file:
#        duckdb_extension_load(hsk SOURCE_DIR /path/to/this/dir)
#      and run DuckDB's cmake with -DDUCKDB_EXTENSION_CONFIGS=<that file>. This CMakeLists.txt
#      detects DuckDB's build_loadable_extension() and builds the extension targets.
#
# Eigen is taken from find_package(Eigen3), else -DHSK_EIGEN_INCLUDE_DIR=<dir>, else fetched
# (set -DHSK_FETCH_EIGEN=OFF to forbid network access).
cmake_minimum_required(VERSION 3.15)

set(TARGET_NAME hsk)
project(${TARGET_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(HSK_FETCH_EIGEN "Download Eigen 3.4.0 if it cannot be found" ON)
option(HSK_BUILD_DUCKDB_EXTENSION "Build the DuckDB extension when DuckDB's build system is present" ON)
option(HSK_BUILD_PYTHON "Build the hsk_native Python module" OFF)
set(HSK_EIGEN_INCLUDE_DIR "" CACHE PATH "Directory containing Eigen/Dense (skips find_package)")
set(HSK_DUCKDB_SOURCE_DIR "" CACHE PATH "DuckDB source tree; enables the 'duckdb_hsk' target")

# ---------------------------------------------------------------------------------------
# Eigen (header-only)
# ---------------------------------------------------------------------------------------
if(HSK_EIGEN_INCLUDE_DIR)
  set(HSK_EIGEN_INCLUDE_DIRS "${HSK_EIGEN_INCLUDE_DIR}")
else()
  find_package(Eigen3 3.3 QUIET NO_MODULE)
  if(TARGET Eigen3::Eigen)
    get_target_property(HSK_EIGEN_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
  elseif(HSK_FETCH_EIGEN)
    include(FetchContent)
    FetchContent_Declare(hsk_eigen
      GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
      GIT_TAG        3.4.0
      GIT_SHALLOW    TRUE
      SOURCE_SUBDIR  hsk-header-only-no-cmake)   # populate only; do not add Eigen's own targets
    FetchContent_MakeAvailable(hsk_eigen)
    set(HSK_EIGEN_INCLUDE_DIRS "${hsk_eigen_SOURCE_DIR}")
  else()
    message(FATAL_ERROR "Eigen >= 3.3 not found. Install it, or pass -DHSK_EIGEN_INCLUDE_DIR=<dir>.")
  endif()
endif()
message(STATUS "hsk: Eigen include dirs: ${HSK_EIGEN_INCLUDE_DIRS}")

set(HSK_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/native")

# Flags shared by every hsk consumer. -fopenmp-simd enables '#pragma omp simd' (vectorized hot
# loops) without linking an OpenMP runtime or creating threads.
set(HSK_DEFINITIONS HSK_OMP_SIMD=1)
set(HSK_SIMD_FLAG "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fopenmp-simd>")

# ---------------------------------------------------------------------------------------
# Header-only SDK
# ---------------------------------------------------------------------------------------
add_library(hsk_sdk INTERFACE)
add_library(hsk::sdk ALIAS hsk_sdk)
target_include_directories(hsk_sdk INTERFACE "$<BUILD_INTERFACE:${HSK_NATIVE_DIR}>")
target_include_directories(hsk_sdk SYSTEM INTERFACE ${HSK_EIGEN_INCLUDE_DIRS})
target_compile_features(hsk_sdk INTERFACE cxx_std_17)
target_compile_definitions(hsk_sdk INTERFACE ${HSK_DEFINITIONS})
target_compile_options(hsk_sdk INTERFACE ${HSK_SIMD_FLAG})

if(NOT HSK_BUILD_PYTHON)
  install(FILES "${HSK_NATIVE_DIR}/hybrid_sketch.hpp" DESTINATION include/hsk)
endif()

# ---------------------------------------------------------------------------------------
# Python module (scikit-build-core / cibuildwheel). Driven by pyproject.toml.
# ---------------------------------------------------------------------------------------
if(HSK_BUILD_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(hsk_native MODULE "${HSK_NATIVE_DIR}/bindings.cpp")
  target_link_libraries(hsk_native PRIVATE hsk::sdk)
  target_compile_definitions(hsk_native PRIVATE EIGEN_MPL2_ONLY)
  install(TARGETS hsk_native LIBRARY DESTINATION hsk RUNTIME DESTINATION hsk)
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/service/engine.py" DESTINATION hsk)
  file(WRITE "${CMAKE_BINARY_DIR}/hsk_init.py" "from .engine import *  # noqa\n")
  install(FILES "${CMAKE_BINARY_DIR}/hsk_init.py" DESTINATION hsk RENAME __init__.py)
endif()

# ---------------------------------------------------------------------------------------
# DuckDB extension, inside DuckDB's build (modes B and C)
# ---------------------------------------------------------------------------------------
if(COMMAND build_loadable_extension AND HSK_BUILD_DUCKDB_EXTENSION)
  set(EXTENSION_NAME ${TARGET_NAME}_extension)
  set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)
  set(EXTENSION_SOURCES "${HSK_NATIVE_DIR}/hsk_duckdb_extension.cpp")

  include_directories("${HSK_NATIVE_DIR}")
  include_directories(SYSTEM ${HSK_EIGEN_INCLUDE_DIRS})

  # DuckDB owns the thread pool: Eigen must not spawn its own OpenMP threads, and we avoid
  # shipping a libgomp dependency inside a loadable extension.
  add_compile_definitions(EIGEN_DONT_PARALLELIZE EIGEN_MPL2_ONLY ${HSK_DEFINITIONS})
  # DuckDB builds with -Werror /WX in CI; third-party Eigen code must not break that build.
  add_compile_options(
    "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wno-error>"
    "$<$<CXX_COMPILER_ID:MSVC>:/WX->"
    ${HSK_SIMD_FLAG})

  build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
  build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

  set_target_properties(${EXTENSION_NAME} ${LOADABLE_EXTENSION_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)

  install(
    TARGETS ${EXTENSION_NAME}
    EXPORT "${DUCKDB_EXPORT_SET}"
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")

# ---------------------------------------------------------------------------------------
# Standalone project: optionally drive a DuckDB build that loads this extension (mode B)
# ---------------------------------------------------------------------------------------
elseif(HSK_BUILD_DUCKDB_EXTENSION AND HSK_DUCKDB_SOURCE_DIR)
  if(NOT EXISTS "${HSK_DUCKDB_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "HSK_DUCKDB_SOURCE_DIR='${HSK_DUCKDB_SOURCE_DIR}' is not a DuckDB source tree")
  endif()
  list(GET HSK_EIGEN_INCLUDE_DIRS 0 HSK_EIGEN_FIRST_DIR)

  set(HSK_DUCKDB_BUILD_DIR "${CMAKE_BINARY_DIR}/duckdb-build")
  set(HSK_DUCKDB_EXT_CONFIG "${CMAKE_BINARY_DIR}/hsk_extension_config.cmake")
  file(WRITE "${HSK_DUCKDB_EXT_CONFIG}"
       "duckdb_extension_load(hsk SOURCE_DIR \"${CMAKE_CURRENT_SOURCE_DIR}\" INCLUDE_DIR \"${CMAKE_CURRENT_SOURCE_DIR}/native\")\n")

  add_custom_target(duckdb_hsk
    COMMAND ${CMAKE_COMMAND}
            -S "${HSK_DUCKDB_SOURCE_DIR}" -B "${HSK_DUCKDB_BUILD_DIR}"
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_CXX_STANDARD=17
            -DBUILD_UNITTESTS=OFF
            -DDUCKDB_EXTENSION_CONFIGS=${HSK_DUCKDB_EXT_CONFIG}
            -DHSK_EIGEN_INCLUDE_DIR=${HSK_EIGEN_FIRST_DIR}
            -DHSK_BUILD_DUCKDB_EXTENSION=ON
    COMMAND ${CMAKE_COMMAND} --build "${HSK_DUCKDB_BUILD_DIR}" --config Release --target hsk_extension --target hsk_loadable_extension
    COMMENT "Building DuckDB with the hsk extension (output: ${HSK_DUCKDB_BUILD_DIR}/extension/hsk/hsk.duckdb_extension)"
    VERBATIM)
else()
  message(STATUS "hsk: SDK only. Pass -DHSK_DUCKDB_SOURCE_DIR=<duckdb checkout> to enable the 'duckdb_hsk' target.")
endif()
