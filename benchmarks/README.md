# Step 1 single-thread benchmark

`generate_plink_benchmark.py` creates a deterministic PLINK 1 dataset with a
polygenic quantitative phenotype. It requires Python 3 and NumPy.

The x86 benchmark used for the `perf/x86` audit was generated with:

```sh
python3 benchmarks/generate_plink_benchmark.py /tmp/bolt-step1 \
  --samples 32768 --variants 16384 --causal-variants 1024 \
  --heritability 0.5 --batch-variants 256
```

For a strictly single-thread oneMKL build and run:

```sh
cmake -S . -B build/step1-mkl \
  -DCMAKE_BUILD_TYPE=Release \
  -DBOLT_OPENMP=OFF \
  -DBOLT_BLAS=MKL \
  -DBOLT_MKL_ROOT=/opt/intel/oneapi/mkl/latest \
  -DMKL_THREADING=sequential \
  -DBOLT_SIMD=SSE2 \
  -DBOLT_CPU_TARGET=portable
cmake --build build/step1-mkl -j

MKL_NUM_THREADS=1 taskset -c 0 build/step1-mkl/bolt \
  --stage=1 \
  --stage1Model=/tmp/bolt-step1.model \
  --bfile=/tmp/bolt-step1 \
  --phenoFile=/tmp/bolt-step1.pheno \
  --phenoCol=PHENO \
  --lmmForceNonInf \
  --LDscoresUseChip \
  --CVfoldsCompute=1 \
  --numThreads=1
```

Use a fresh model output path for each timed repetition.

The CUDA build described in `BUILD.md` uses the GPU automatically for Stage 1.
Pass `--no-cuda` to benchmark its CPU path instead. The legacy `--cuda` flag is
still accepted.

On an NVIDIA A100-SXM4-40GB, the workload above took 3.3 seconds versus 111.8
seconds for the portable single-thread oneMKL build, a 33.69x speedup. Reported
heritability, cross-validation choice, prediction errors, convergence
trajectories, and association summaries matched the CPU run at their displayed
precision.

A larger 131,072-sample, 16,384-variant fixture took 8.4 seconds with CUDA.
Moving the two SNP normalization/projection passes for the main and CV-fold
models to the GPU reduced that fixture from 26.2 seconds with the otherwise
same CUDA backend.

The same 131,072-sample fixture converted to hardcall PGEN was used to measure
single-threaded input setup. Fusing packed-code conversion, QC statistics, and
missing-sample collection reduced median `SnpData` setup time from 2.947 to
1.320 seconds across three pinned runs (55.2%). Stage 1 model artifacts from all
baseline and optimized runs were byte-identical.

Populating the shared device hardcall cache during the main marker-
initialization scan, and reusing it during fold initialization, reduced median
main initialization from 0.926 to 0.799 seconds (13.7%) and the one-fold
mixture-parameter phase from 1.022 to 0.907 seconds (11.2%) on that fixture.
This also avoids redundant scratch-disk reads when the packed host matrix is
file-backed and larger than RAM. Model artifacts again remained byte-identical.

## Uncached CUDA streaming

The 131,072-sample PGEN fixture was also run with `--cudaCacheGiB=0` to exercise
the host-to-device path for every genotype block. Two pinned buffers and a
nonblocking transfer stream reduced median phase times as follows across three
pinned runs:

| Phase | Synchronous H2D | Double-buffered H2D | Reduction |
| --- | ---: | ---: | ---: |
| Variance fitting | 2.673 s | 1.981 s | 25.9% |
| Mixture estimation (one fold) | 1.985 s | 1.519 s | 23.5% |
| Bayesian association scoring | 1.649 s | 1.329 s | 19.4% |

Main/fold marker setup remained neutral (0.793 versus 0.789 seconds). The
normal fully cached path stayed within 1.2% of its prior timings. A half-cached
run exercised the cache-to-stream transition, and all cached, partially cached,
and uncached runs produced the same byte-identical Stage 1 model artifact.
