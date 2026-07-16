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
