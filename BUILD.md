# Building BOLT-LMM

BOLT-LMM uses CMake for portable Linux and macOS builds. The Makefile in the
repository root is a convenience wrapper; `src/Makefile` remains available for
legacy Intel/Linux release environments.

## Dependencies

- A C++14 compiler and CMake 3.18 or newer
- The bundled pgenlib sources (no separate PLINK 2 installation is required)
- Boost.Program_options and Boost.Iostreams
- zlib and zstd
- NLopt, including its C++ header (`nlopt.hpp`)
- OpenMP for multithreading; a serial build is available when it is absent
- One of Intel oneMKL, OpenBLAS, Apple Accelerate, or system BLAS/LAPACK

## Standard build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The executable is written to `build/bolt`. The equivalent convenience command
is `make`; additional CMake options can be supplied through `CMAKE_ARGS`.

## macOS on Apple Silicon

Apple Accelerate is the default linear-algebra backend. Homebrew's OpenBLAS can
be selected explicitly when reproducibility with Linux BLAS builds is useful:

```sh
brew install cmake boost nlopt zstd libomp openblas

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(brew --prefix)" \
  -DBOLT_OPENMP_ROOT="$(brew --prefix libomp)" \
  -DBOLT_BLAS=OPENBLAS \
  -DBOLT_OPENBLAS_ROOT="$(brew --prefix openblas)" \
  -DBOLT_ZSTD_ROOT="$(brew --prefix zstd)"
cmake --build build --parallel
```

Use `-DBOLT_BLAS=ACCELERATE` to select Accelerate explicitly. The default
portable build emits no `-mcpu` flag and runs on Apple Silicon generally;
`-DBOLT_CPU_TARGET=apple-m1` sets an M1 floor, while
`-DBOLT_CPU_TARGET=native` optimizes for the build machine.

## Linux on x86-64

With distribution BLAS/LAPACK libraries installed, `BOLT_BLAS=AUTO` selects the
system backend. If `MKLROOT` or `BOLT_MKL_ROOT` is set, AUTO selects oneMKL.
Backends can also be selected explicitly:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DBOLT_BLAS=MKL -DBOLT_MKL_ROOT=/opt/intel/oneapi/mkl/latest

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DBOLT_BLAS=OPENBLAS -DBOLT_OPENBLAS_ROOT=/opt/OpenBLAS
```

Portable builds use the compiler's baseline architecture. A redistributable
modern x86-64 build can use
`-DBOLT_CPU_TARGET=x86-64-v3 -DBOLT_CPU_TUNE=generic`; use `native` only for a
binary that will stay on the build machine.

## Configuration options

- `BOLT_BLAS=AUTO|MKL|OPENBLAS|ACCELERATE|SYSTEM`
- `BOLT_OPENMP=AUTO|ON|OFF`; AUTO falls back to serial execution
- `BOLT_SIMD=AUTO|SSE2|SCALAR`; AUTO uses SSE2 only on x86
- `BOLT_CPU_TARGET=portable|native|<compiler-target>`
- `BOLT_CPU_TUNE=<x86-tuning-target>`
- `BOLT_NLOPT_ROOT`, `BOLT_ZSTD_ROOT`, and backend-specific root paths for
  dependencies outside standard search locations

The legacy glibc memcpy wrapper is excluded by default. It can be enabled for
specialized old-Linux release builds with `BOLT_WRAP_LEGACY_MEMCPY=ON`.
Optimized CMake builds retain BOLT's existing safety assertions because some
legacy binary-input checks still rely on them.
