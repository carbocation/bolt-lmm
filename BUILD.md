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
- Optional: CUDA Toolkit with cuBLAS for Step 1 GPU acceleration

## Standard build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The executable is written to `build/bolt`. The equivalent convenience command
is `make`; additional CMake options can be supplied through `CMAKE_ARGS`.

## PLINK 2 hardcall input

Use `--pfile PREFIX` in either Stage 1 or Stage 2 to read
`PREFIX.pgen`, `PREFIX.pvar` (or `.pvar.gz`), and `PREFIX.psam` (or `.psam.gz`).
BOLT accepts biallelic variants and intentionally reads the PGEN hardcall
component; dosage overrides in the PGEN are reported and ignored.

Stage 1 keeps a BED-coded 2-bit cache because its numerical kernels revisit
variants many times. When this cache would exceed half of physical RAM, BOLT
automatically creates an unlinked, file-backed cache under `TMPDIR` (or `/tmp`).
Use `--pgenCacheDir DIR` to choose local scratch storage explicitly; specifying
the option also forces file-backed storage, which is useful for testing and for
leaving RAM available to operating-system and model working sets. BOLT reports
the required cache size and checks available scratch space before loading.
Packed-code conversion, allele/missingness accumulation, sample-missingness
tracking, and the final cache write are fused into one host pass during PGEN
ingestion.

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

## CUDA Step 1 acceleration

CUDA support is opt-in and leaves the default CPU build unchanged. The GPU path
accelerates the repeated projected `X X'` products used by variance estimation
and the infinitesimal model, along with the block matrix operations in the
Bayesian iterations and the `X beta` products used by Monte Carlo phenotype
generation and cross-validation prediction. Transposed products used by
variance estimation, LINREG, and retrospective association scoring are also
computed on the GPU. Initial SNP normalization and covariate projection for the
main analysis and cross-validation folds also run on the GPU. Packed 2-bit
genotypes are transferred to the device and decoded there; expanded genotype
matrices do not cross PCIe.

The CUDA backend uses otherwise-free device memory to cache a shared prefix of
the packed genotype matrix. It reserves one additional decoded SNP block plus
4 GiB for a simultaneous cross-validation fold and working buffers. Main and
fold-specific `Bolt` instances share the packed cache while retaining their own
sample masks and covariate projections.

For an NVIDIA A100 (compute capability 8.0):

```sh
cmake -S . -B build/cuda -DCMAKE_BUILD_TYPE=Release \
  -DBOLT_CUDA=ON \
  -DBOLT_CUDA_ARCHITECTURES=80 \
  -DBOLT_BLAS=MKL \
  -DBOLT_MKL_ROOT=/opt/intel/oneapi/mkl/latest
cmake --build build/cuda --parallel
ctest --test-dir build/cuda --output-on-failure
```

If `nvcc` is not on `PATH`, also set
`-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`. CUDA-enabled binaries use the
GPU path by default for Stage 1. Pass `--no-cuda` to force the CPU path for
testing or comparison. The existing `--cuda` flag remains accepted for command
line compatibility. Builds made without `BOLT_CUDA=ON` retain the CPU-only
default and have no CUDA runtime dependency.

## Configuration options

- `BOLT_BLAS=AUTO|MKL|OPENBLAS|ACCELERATE|SYSTEM`
- `BOLT_OPENMP=AUTO|ON|OFF`; AUTO falls back to serial execution
- `BOLT_SIMD=AUTO|SSE2|SCALAR`; AUTO uses SSE2 only on x86
- `BOLT_CPU_TARGET=portable|native|<compiler-target>`
- `BOLT_CPU_TUNE=<x86-tuning-target>`
- `BOLT_CUDA=ON|OFF` and `BOLT_CUDA_ARCHITECTURES=<CMake architecture list>`
- `BOLT_NLOPT_ROOT`, `BOLT_ZSTD_ROOT`, and backend-specific root paths for
  dependencies outside standard search locations

The legacy glibc memcpy wrapper is excluded by default. It can be enabled for
specialized old-Linux release builds with `BOLT_WRAP_LEGACY_MEMCPY=ON`.
Optimized CMake builds retain BOLT's existing safety assertions because some
legacy binary-input checks still rely on them.
