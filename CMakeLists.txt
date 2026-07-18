cmake_minimum_required(VERSION 3.18)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

project(BOLT_LMM VERSION 2.5 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are not supported. Use: cmake -S . -B build")
endif()

include(CheckCXXCompilerFlag)
include(CTest)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BOLT_BLAS "AUTO" CACHE STRING
  "Linear algebra backend: AUTO, MKL, OPENBLAS, ACCELERATE, or SYSTEM")
set_property(CACHE BOLT_BLAS PROPERTY STRINGS
  AUTO MKL OPENBLAS ACCELERATE SYSTEM)

set(BOLT_OPENMP "AUTO" CACHE STRING
  "OpenMP support: AUTO, ON, or OFF")
set_property(CACHE BOLT_OPENMP PROPERTY STRINGS AUTO ON OFF)

set(BOLT_SIMD "AUTO" CACHE STRING
  "Genotype decoding implementation: AUTO, SSE2, NEON, or SCALAR")
set_property(CACHE BOLT_SIMD PROPERTY STRINGS AUTO SSE2 NEON SCALAR)

set(BOLT_LIBDEFLATE "AUTO" CACHE STRING
  "libdeflate acceleration for zlib-compressed BGEN: AUTO, ON, or OFF")
set_property(CACHE BOLT_LIBDEFLATE PROPERTY STRINGS AUTO ON OFF)

set(BOLT_CPU_TARGET "portable" CACHE STRING
  "CPU target: portable, native, or a compiler target such as x86-64-v3 or apple-m1")
set(BOLT_CPU_TUNE "" CACHE STRING
  "Optional x86 compiler tuning target, for example generic or native")

option(BOLT_CUDA "Build optional CUDA acceleration" OFF)
set(BOLT_CUDA_ARCHITECTURES "75;80" CACHE STRING
  "CUDA architectures to compile when BOLT_CUDA is enabled")

set(BOLT_OPENMP_ROOT "" CACHE PATH "Root of an OpenMP installation")
set(BOLT_MKL_ROOT "" CACHE PATH "Root of an Intel oneMKL installation")
set(BOLT_OPENBLAS_ROOT "" CACHE PATH "Root of an OpenBLAS installation")
set(BOLT_NLOPT_ROOT "" CACHE PATH "Root of an NLopt installation")
set(BOLT_ZSTD_ROOT "" CACHE PATH "Root of a zstd installation")
set(BOLT_LIBDEFLATE_ROOT "" CACHE PATH "Root of a libdeflate installation")

option(BOLT_USE_MKL_MALLOC "Use MKL's aligned allocator with the MKL backend" OFF)
option(BOLT_WRAP_LEGACY_MEMCPY
  "Use BOLT's old-glibc memcpy symbol wrapper (Linux release compatibility only)" OFF)

foreach(option_name IN ITEMS BOLT_BLAS BOLT_OPENMP BOLT_SIMD BOLT_LIBDEFLATE)
  string(TOUPPER "${${option_name}}" normalized_value)
  set(${option_name} "${normalized_value}")
endforeach()

if(NOT BOLT_BLAS MATCHES "^(AUTO|MKL|OPENBLAS|ACCELERATE|SYSTEM)$")
  message(FATAL_ERROR "Unsupported BOLT_BLAS='${BOLT_BLAS}'")
endif()
if(NOT BOLT_OPENMP MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "Unsupported BOLT_OPENMP='${BOLT_OPENMP}'")
endif()
if(NOT BOLT_SIMD MATCHES "^(AUTO|SSE2|NEON|SCALAR)$")
  message(FATAL_ERROR "Unsupported BOLT_SIMD='${BOLT_SIMD}'")
endif()
if(NOT BOLT_LIBDEFLATE MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "Unsupported BOLT_LIBDEFLATE='${BOLT_LIBDEFLATE}'")
endif()

set(BOLT_TARGET_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
  list(LENGTH CMAKE_OSX_ARCHITECTURES bolt_osx_arch_count)
  if(bolt_osx_arch_count EQUAL 1)
    list(GET CMAKE_OSX_ARCHITECTURES 0 BOLT_TARGET_PROCESSOR)
  else()
    set(BOLT_TARGET_PROCESSOR "universal")
  endif()
endif()
string(TOLOWER "${BOLT_TARGET_PROCESSOR}" BOLT_SYSTEM_PROCESSOR)
set(BOLT_X86 FALSE)
set(BOLT_ARM FALSE)
if(BOLT_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|i[3-6]86)$")
  set(BOLT_X86 TRUE)
elseif(BOLT_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|armv[0-9].*)$")
  set(BOLT_ARM TRUE)
endif()

add_library(bolt_build_options INTERFACE)
target_compile_features(bolt_build_options INTERFACE cxx_std_14)
target_compile_definitions(bolt_build_options INTERFACE MEASURE_DGEMM VERBOSE)
target_compile_options(bolt_build_options INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-Wextra>
  # BOLT performs checked binary reads inside assert(), so Release builds must
  # retain assertions until those legacy checks are converted to runtime errors.
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-UNDEBUG>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Clang,AppleClang>>:-Wno-deprecated-copy>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Clang,AppleClang>>:-Wno-vla-cxx-extension>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:AppleClang>>:-Wno-deprecated-declarations>)

if(BOLT_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(BOLT_SIMD STREQUAL "AUTO")
  if(BOLT_X86)
    set(BOLT_SIMD "SSE2")
  elseif(BOLT_ARM AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(BOLT_SIMD "NEON")
  else()
    set(BOLT_SIMD "SCALAR")
  endif()
endif()
if(BOLT_SIMD STREQUAL "SSE2")
  if(NOT BOLT_X86)
    message(FATAL_ERROR "BOLT_SIMD=SSE2 is only valid for x86 processors")
  endif()
  target_compile_definitions(bolt_build_options INTERFACE USE_SSE)
  if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    target_compile_options(bolt_build_options INTERFACE -msse2)
  endif()
elseif(BOLT_SIMD STREQUAL "NEON")
  if(NOT BOLT_ARM OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "BOLT_SIMD=NEON is only valid for 64-bit ARM processors")
  endif()
  target_compile_definitions(bolt_build_options INTERFACE USE_NEON)
endif()

if(NOT BOLT_CPU_TARGET STREQUAL "portable")
  if(BOLT_CPU_TARGET STREQUAL "native")
    if(BOLT_ARM)
      set(BOLT_CPU_FLAG "-mcpu=native")
    else()
      set(BOLT_CPU_FLAG "-march=native")
    endif()
  elseif(BOLT_ARM)
    set(BOLT_CPU_FLAG "-mcpu=${BOLT_CPU_TARGET}")
  elseif(BOLT_X86)
    set(BOLT_CPU_FLAG "-march=${BOLT_CPU_TARGET}")
  else()
    message(FATAL_ERROR
      "BOLT_CPU_TARGET is not supported for processor '${CMAKE_SYSTEM_PROCESSOR}'")
  endif()
  check_cxx_compiler_flag("${BOLT_CPU_FLAG}" BOLT_CPU_FLAG_SUPPORTED)
  if(NOT BOLT_CPU_FLAG_SUPPORTED)
    message(FATAL_ERROR
      "Compiler '${CMAKE_CXX_COMPILER_ID}' does not support ${BOLT_CPU_FLAG}")
  endif()
  target_compile_options(bolt_build_options INTERFACE "${BOLT_CPU_FLAG}")
endif()

if(NOT BOLT_CPU_TUNE STREQUAL "")
  if(NOT BOLT_X86)
    message(FATAL_ERROR "BOLT_CPU_TUNE is currently supported only on x86")
  endif()
  set(BOLT_TUNE_FLAG "-mtune=${BOLT_CPU_TUNE}")
  check_cxx_compiler_flag("${BOLT_TUNE_FLAG}" BOLT_TUNE_FLAG_SUPPORTED)
  if(NOT BOLT_TUNE_FLAG_SUPPORTED)
    message(FATAL_ERROR
      "Compiler '${CMAKE_CXX_COMPILER_ID}' does not support ${BOLT_TUNE_FLAG}")
  endif()
  target_compile_options(bolt_build_options INTERFACE "${BOLT_TUNE_FLAG}")
endif()

find_package(Threads REQUIRED)
find_package(Boost 1.58 REQUIRED COMPONENTS program_options iostreams)
find_package(ZLIB REQUIRED)

set(BOLT_PGENLIB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pgenlib")
add_library(bolt_pgenlib STATIC
  ${BOLT_PGENLIB_ROOT}/pgenlibr.cpp
  ${BOLT_PGENLIB_ROOT}/pgenlib_ffi_support.cpp
  ${BOLT_PGENLIB_ROOT}/pvar_ffi_support.cc
  ${BOLT_PGENLIB_ROOT}/include/pgenlib_misc.cc
  ${BOLT_PGENLIB_ROOT}/include/pgenlib_read.cc
  ${BOLT_PGENLIB_ROOT}/include/plink2_base.cc
  ${BOLT_PGENLIB_ROOT}/include/plink2_bits.cc)
target_compile_features(bolt_pgenlib PUBLIC cxx_std_11)
target_include_directories(bolt_pgenlib PUBLIC
  "${BOLT_PGENLIB_ROOT}"
  "${BOLT_PGENLIB_ROOT}/include"
  "${BOLT_PGENLIB_ROOT}/simde")
# pgenlib is vendored third-party code and has its own warning policy.
target_compile_options(bolt_pgenlib PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-w>)

add_library(bolt_zstd INTERFACE)
find_package(zstd CONFIG QUIET)
if(TARGET zstd::libzstd_shared)
  target_link_libraries(bolt_zstd INTERFACE zstd::libzstd_shared)
elseif(TARGET zstd::libzstd_static)
  target_link_libraries(bolt_zstd INTERFACE zstd::libzstd_static)
else()
  set(zstd_hints "${BOLT_ZSTD_ROOT}" "$ENV{ZSTD_ROOT}" "$ENV{ZSTD_DIR}")
  find_path(BOLT_ZSTD_INCLUDE_DIR zstd.h HINTS ${zstd_hints}
    PATH_SUFFIXES include lib)
  find_library(BOLT_ZSTD_LIBRARY NAMES zstd HINTS ${zstd_hints}
    PATH_SUFFIXES lib lib64)
  if(NOT BOLT_ZSTD_INCLUDE_DIR OR NOT BOLT_ZSTD_LIBRARY)
    message(FATAL_ERROR
      "zstd was not found. Set BOLT_ZSTD_ROOT or CMAKE_PREFIX_PATH.")
  endif()
  target_include_directories(bolt_zstd INTERFACE "${BOLT_ZSTD_INCLUDE_DIR}")
  target_link_libraries(bolt_zstd INTERFACE "${BOLT_ZSTD_LIBRARY}")
endif()

add_library(bolt_libdeflate INTERFACE)
set(BOLT_LIBDEFLATE_ENABLED FALSE)
if(NOT BOLT_LIBDEFLATE STREQUAL "OFF")
  set(libdeflate_hints "${BOLT_LIBDEFLATE_ROOT}" "$ENV{LIBDEFLATE_ROOT}")
  find_path(BOLT_LIBDEFLATE_INCLUDE_DIR libdeflate.h HINTS ${libdeflate_hints}
    PATH_SUFFIXES include)
  find_library(BOLT_LIBDEFLATE_LIBRARY NAMES deflate libdeflate
    HINTS ${libdeflate_hints} PATH_SUFFIXES lib lib64)
  if(BOLT_LIBDEFLATE_INCLUDE_DIR AND BOLT_LIBDEFLATE_LIBRARY)
    target_include_directories(bolt_libdeflate INTERFACE
      "${BOLT_LIBDEFLATE_INCLUDE_DIR}")
    target_link_libraries(bolt_libdeflate INTERFACE "${BOLT_LIBDEFLATE_LIBRARY}")
    target_compile_definitions(bolt_libdeflate INTERFACE BOLT_USE_LIBDEFLATE)
    set(BOLT_LIBDEFLATE_ENABLED TRUE)
    message(STATUS "Using libdeflate for zlib-compressed BGEN")
  elseif(BOLT_LIBDEFLATE STREQUAL "ON")
    message(FATAL_ERROR
      "libdeflate was not found. Set BOLT_LIBDEFLATE_ROOT or CMAKE_PREFIX_PATH, "
      "or use BOLT_LIBDEFLATE=OFF.")
  else()
    message(STATUS "libdeflate not found; using zlib for compressed BGEN")
  endif()
endif()

add_library(bolt_nlopt INTERFACE)
find_package(NLopt CONFIG QUIET)
if(TARGET NLopt::nlopt)
  target_link_libraries(bolt_nlopt INTERFACE NLopt::nlopt)
elseif(TARGET nlopt)
  target_link_libraries(bolt_nlopt INTERFACE nlopt)
else()
  set(nlopt_hints "${BOLT_NLOPT_ROOT}" "$ENV{NLOPT_ROOT}"
    "$ENV{NLOPT_INSTALL_DIR}")
  find_path(BOLT_NLOPT_INCLUDE_DIR nlopt.hpp HINTS ${nlopt_hints}
    PATH_SUFFIXES include api)
  find_library(BOLT_NLOPT_LIBRARY NAMES nlopt HINTS ${nlopt_hints}
    PATH_SUFFIXES lib lib64 .libs)
  if(NOT BOLT_NLOPT_INCLUDE_DIR OR NOT BOLT_NLOPT_LIBRARY)
    message(FATAL_ERROR
      "NLopt with its C++ header was not found. Set BOLT_NLOPT_ROOT or CMAKE_PREFIX_PATH.")
  endif()
  target_include_directories(bolt_nlopt INTERFACE "${BOLT_NLOPT_INCLUDE_DIR}")
  target_link_libraries(bolt_nlopt INTERFACE "${BOLT_NLOPT_LIBRARY}")
endif()

add_library(bolt_openmp INTERFACE)
set(BOLT_OPENMP_ENABLED FALSE)
if(NOT BOLT_OPENMP STREQUAL "OFF")
  if(NOT BOLT_OPENMP_ROOT STREQUAL "")
    list(PREPEND CMAKE_PREFIX_PATH "${BOLT_OPENMP_ROOT}")
  endif()
  find_package(OpenMP QUIET COMPONENTS CXX)
  if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(bolt_openmp INTERFACE OpenMP::OpenMP_CXX)
    set(BOLT_OPENMP_ENABLED TRUE)
  elseif(APPLE)
    set(openmp_hints "${BOLT_OPENMP_ROOT}" "$ENV{OpenMP_ROOT}"
      "$ENV{OPENMP_ROOT}")
    find_path(BOLT_OPENMP_INCLUDE_DIR omp.h HINTS ${openmp_hints}
      PATH_SUFFIXES include)
    find_library(BOLT_OPENMP_LIBRARY NAMES omp HINTS ${openmp_hints}
      PATH_SUFFIXES lib)
    if(BOLT_OPENMP_INCLUDE_DIR AND BOLT_OPENMP_LIBRARY)
      target_compile_options(bolt_openmp INTERFACE -Xpreprocessor -fopenmp)
      target_include_directories(bolt_openmp INTERFACE "${BOLT_OPENMP_INCLUDE_DIR}")
      target_link_libraries(bolt_openmp INTERFACE "${BOLT_OPENMP_LIBRARY}")
      set(BOLT_OPENMP_ENABLED TRUE)
    endif()
  endif()
endif()
if(BOLT_OPENMP_ENABLED)
  target_compile_definitions(bolt_openmp INTERFACE BOLT_USE_OPENMP)
elseif(BOLT_OPENMP STREQUAL "ON")
  message(FATAL_ERROR
    "OpenMP was requested but not found. Set BOLT_OPENMP_ROOT or OpenMP_ROOT.")
elseif(BOLT_OPENMP STREQUAL "AUTO")
  message(WARNING "OpenMP was not found; BOLT will be built for serial execution")
endif()

add_library(bolt_linear_algebra INTERFACE)
set(BOLT_BLAS_SELECTED "${BOLT_BLAS}")
if(BOLT_BLAS_SELECTED STREQUAL "AUTO")
  if((NOT BOLT_MKL_ROOT STREQUAL "" OR NOT "$ENV{MKLROOT}" STREQUAL "")
      AND BOLT_X86)
    set(BOLT_BLAS_SELECTED "MKL")
  elseif(APPLE)
    set(BOLT_BLAS_SELECTED "ACCELERATE")
  else()
    set(BOLT_BLAS_SELECTED "SYSTEM")
  endif()
endif()

if(BOLT_BLAS_SELECTED STREQUAL "MKL")
  if(NOT BOLT_X86)
    message(FATAL_ERROR "The MKL backend is supported only on x86 processors")
  endif()
  set(mkl_hints "${BOLT_MKL_ROOT}" "$ENV{MKLROOT}")
  # BOLT's Fortran BLAS/LAPACK declarations use 32-bit integers.  Recent
  # oneMKL CMake packages default to ILP64, so select the matching ABI unless
  # the caller explicitly requested an interface.
  if(NOT DEFINED MKL_INTERFACE AND NOT DEFINED MKL_INTERFACE_FULL)
    set(MKL_INTERFACE "lp64")
  endif()
  find_package(MKL CONFIG QUIET HINTS ${mkl_hints}
    PATH_SUFFIXES lib/cmake/mkl)
  if(TARGET MKL::MKL)
    target_link_libraries(bolt_linear_algebra INTERFACE MKL::MKL)
  else()
    find_path(BOLT_MKL_INCLUDE_DIR mkl.h HINTS ${mkl_hints}
      PATH_SUFFIXES include)
    find_library(BOLT_MKL_LIBRARY NAMES mkl_rt HINTS ${mkl_hints}
      PATH_SUFFIXES lib lib64 lib/intel64)
    if(NOT BOLT_MKL_INCLUDE_DIR OR NOT BOLT_MKL_LIBRARY)
      message(FATAL_ERROR
        "oneMKL was not found. Set BOLT_MKL_ROOT, MKLROOT, or CMAKE_PREFIX_PATH.")
    endif()
    target_include_directories(bolt_linear_algebra INTERFACE
      "${BOLT_MKL_INCLUDE_DIR}")
    target_link_libraries(bolt_linear_algebra INTERFACE
      "${BOLT_MKL_LIBRARY}" Threads::Threads ${CMAKE_DL_LIBS})
  endif()
  target_compile_definitions(bolt_linear_algebra INTERFACE USE_MKL)
  if(BOLT_USE_MKL_MALLOC)
    target_compile_definitions(bolt_linear_algebra INTERFACE USE_MKL_MALLOC)
  endif()
elseif(BOLT_BLAS_SELECTED STREQUAL "OPENBLAS")
  set(openblas_hints "${BOLT_OPENBLAS_ROOT}" "$ENV{OPENBLAS_ROOT}")
  find_package(OpenBLAS CONFIG QUIET HINTS ${openblas_hints}
    PATH_SUFFIXES lib/cmake/openblas)
  if(OpenBLAS_FOUND)
    target_include_directories(bolt_linear_algebra INTERFACE
      ${OpenBLAS_INCLUDE_DIRS})
    target_link_libraries(bolt_linear_algebra INTERFACE ${OpenBLAS_LIBRARIES})
  else()
    find_path(BOLT_OPENBLAS_INCLUDE_DIR cblas.h HINTS ${openblas_hints}
      PATH_SUFFIXES include include/openblas)
    find_library(BOLT_OPENBLAS_LIBRARY NAMES openblas HINTS ${openblas_hints}
      PATH_SUFFIXES lib lib64)
    if(NOT BOLT_OPENBLAS_INCLUDE_DIR OR NOT BOLT_OPENBLAS_LIBRARY)
      message(FATAL_ERROR
        "OpenBLAS was not found. Set BOLT_OPENBLAS_ROOT, OPENBLAS_ROOT, or CMAKE_PREFIX_PATH.")
    endif()
    target_include_directories(bolt_linear_algebra INTERFACE
      "${BOLT_OPENBLAS_INCLUDE_DIR}")
    target_link_libraries(bolt_linear_algebra INTERFACE
      "${BOLT_OPENBLAS_LIBRARY}")
  endif()
elseif(BOLT_BLAS_SELECTED STREQUAL "ACCELERATE")
  if(NOT APPLE)
    message(FATAL_ERROR "The Accelerate backend is available only on Apple platforms")
  endif()
  find_library(BOLT_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  target_link_libraries(bolt_linear_algebra INTERFACE
    "${BOLT_ACCELERATE_FRAMEWORK}")
elseif(BOLT_BLAS_SELECTED STREQUAL "SYSTEM")
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  target_link_libraries(bolt_linear_algebra INTERFACE BLAS::BLAS LAPACK::LAPACK)
endif()

set(BOLT_SOURCES
  src/Bolt.cpp
  src/BoltMain.cpp
  src/BoltParams.cpp
  src/BoltParEstCV.cpp
  src/BoltReml.cpp
  src/CovariateBasis.cpp
  src/DataMatrix.cpp
  src/FileUtils.cpp
  src/Jackknife.cpp
  src/LDscoreCalibration.cpp
  src/MapInterpolater.cpp
  src/MatrixUtils.cpp
  src/MemoryUtils.cpp
  src/NonlinearOptMulti.cpp
  src/NumericUtils.cpp
  src/PhenoBuilder.cpp
  src/PgenUtils.cpp
  src/RestrictSnpSet.cpp
  src/SnpData.cpp
  src/SnpInfo.cpp
  src/SpectrumTools.cpp
  src/Stage1Model.cpp
  src/StatsUtils.cpp
  src/StringUtils.cpp
  src/Timer.cpp)

if(BOLT_CUDA)
  list(APPEND BOLT_SOURCES
    src/CudaStep1.cu
    src/CudaStage2.cu
    src/ParallelMemCopy.cpp)
endif()

if(BOLT_WRAP_LEGACY_MEMCPY)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "BOLT_WRAP_LEGACY_MEMCPY is supported only on Linux")
  endif()
  list(APPEND BOLT_SOURCES src/memcpy.cpp)
endif()

add_executable(bolt ${BOLT_SOURCES})
target_include_directories(bolt PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(bolt PRIVATE
  bolt_build_options
  bolt_pgenlib
  bolt_linear_algebra
  bolt_nlopt
  bolt_openmp
  bolt_zstd
  bolt_libdeflate
  Boost::program_options
  Boost::iostreams
  ZLIB::ZLIB
  Threads::Threads
  ${CMAKE_DL_LIBS})

if(BOLT_CUDA)
  target_compile_definitions(bolt PRIVATE BOLT_USE_CUDA)
  target_link_libraries(bolt PRIVATE CUDA::cudart CUDA::cublas)
  set_target_properties(bolt PROPERTIES
    CUDA_ARCHITECTURES "${BOLT_CUDA_ARCHITECTURES}")
endif()

if(BOLT_WRAP_LEGACY_MEMCPY)
  target_link_options(bolt PRIVATE "-Wl,--wrap=memcpy")
endif()

install(TARGETS bolt RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

if(BUILD_TESTING)
  add_test(NAME bolt_help COMMAND bolt --help)

  add_executable(pgen_stage1_parity_test
    tests/PgenStage1Parity.cpp
    src/FileUtils.cpp
    src/MapInterpolater.cpp
    src/MemoryUtils.cpp
    src/NumericUtils.cpp
    src/PgenUtils.cpp
    src/RestrictSnpSet.cpp
    src/SnpData.cpp
    src/SnpInfo.cpp
    src/StringUtils.cpp)
  target_include_directories(pgen_stage1_parity_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(pgen_stage1_parity_test PRIVATE
    bolt_build_options
    bolt_pgenlib
    bolt_linear_algebra
    bolt_openmp
    bolt_zstd
    Boost::iostreams
    ZLIB::ZLIB
    Threads::Threads
    ${CMAKE_DL_LIBS})
  add_test(NAME pgen_stage1_hardcall_parity
    COMMAND pgen_stage1_parity_test
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/pgen_parity"
      "${CMAKE_CURRENT_BINARY_DIR}")

  set(BOLT_STAGE2_TEST_SOURCES ${BOLT_SOURCES})
  list(REMOVE_ITEM BOLT_STAGE2_TEST_SOURCES
    src/BoltMain.cpp
    src/BoltParams.cpp)
  add_executable(pgen_stage2_parity_test
    tests/PgenStage2Parity.cpp
    ${BOLT_STAGE2_TEST_SOURCES})
  target_include_directories(pgen_stage2_parity_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(pgen_stage2_parity_test PRIVATE
    bolt_build_options
    bolt_pgenlib
    bolt_linear_algebra
    bolt_nlopt
    bolt_openmp
    bolt_zstd
    bolt_libdeflate
    Boost::iostreams
    ZLIB::ZLIB
    Threads::Threads
    ${CMAKE_DL_LIBS})
  if(BOLT_CUDA)
    target_compile_definitions(pgen_stage2_parity_test PRIVATE BOLT_USE_CUDA)
    target_link_libraries(pgen_stage2_parity_test PRIVATE CUDA::cudart CUDA::cublas)
    set_target_properties(pgen_stage2_parity_test PROPERTIES
      CUDA_ARCHITECTURES "${BOLT_CUDA_ARCHITECTURES}")
  endif()
  add_test(NAME pgen_stage2_hardcall_parity
    COMMAND pgen_stage2_parity_test
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/pgen_parity"
      "${CMAKE_CURRENT_BINARY_DIR}")

  if(BOLT_CUDA)
    add_executable(cuda_step1_parity_test
      tests/CudaStep1Parity.cpp
      src/CudaStep1.cu
      src/ParallelMemCopy.cpp)
    target_include_directories(cuda_step1_parity_test PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(cuda_step1_parity_test PRIVATE
      bolt_openmp CUDA::cudart CUDA::cublas)
    set_target_properties(cuda_step1_parity_test PROPERTIES
      CUDA_ARCHITECTURES "${BOLT_CUDA_ARCHITECTURES}")
    add_test(NAME cuda_step1_matrix_parity COMMAND cuda_step1_parity_test)
    add_test(NAME cuda_step1_streaming_parity
      COMMAND cuda_step1_parity_test --no-cache)
    add_test(NAME cuda_step1_host_cache_parity
      COMMAND cuda_step1_parity_test --host-cache)

    add_executable(cuda_step1_stream_benchmark EXCLUDE_FROM_ALL
      benchmarks/CudaStep1StreamingBenchmark.cpp
      src/CudaStep1.cu
      src/ParallelMemCopy.cpp)
    target_include_directories(cuda_step1_stream_benchmark PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(cuda_step1_stream_benchmark PRIVATE
      bolt_openmp CUDA::cudart CUDA::cublas)
    set_target_properties(cuda_step1_stream_benchmark PROPERTIES
      CUDA_ARCHITECTURES "${BOLT_CUDA_ARCHITECTURES}")

    add_executable(cuda_stage2_parity_test
      tests/CudaStage2Parity.cpp
      src/CudaStage2.cu)
    target_include_directories(cuda_stage2_parity_test PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(cuda_stage2_parity_test PRIVATE CUDA::cudart)
    set_target_properties(cuda_stage2_parity_test PROPERTIES
      CUDA_ARCHITECTURES "${BOLT_CUDA_ARCHITECTURES}")
    add_test(NAME cuda_stage2_packed_parity COMMAND cuda_stage2_parity_test)
  endif()
endif()

message(STATUS "BOLT platform: ${CMAKE_SYSTEM_NAME}/${BOLT_TARGET_PROCESSOR}")
message(STATUS "BOLT BLAS backend: ${BOLT_BLAS_SELECTED}")
message(STATUS "BOLT SIMD implementation: ${BOLT_SIMD}")
message(STATUS "BOLT CPU target: ${BOLT_CPU_TARGET}")
if(BOLT_OPENMP_ENABLED)
  message(STATUS "BOLT OpenMP: enabled")
else()
  message(STATUS "BOLT OpenMP: serial fallback")
endif()
if(BOLT_LIBDEFLATE_ENABLED)
  message(STATUS "BOLT libdeflate BGEN acceleration: enabled")
else()
  message(STATUS "BOLT libdeflate BGEN acceleration: disabled")
endif()
if(BOLT_CUDA)
  message(STATUS "BOLT CUDA acceleration: enabled (${BOLT_CUDA_ARCHITECTURES})")
else()
  message(STATUS "BOLT CUDA acceleration: disabled")
endif()
