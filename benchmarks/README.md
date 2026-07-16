# Step 1 single-thread benchmark

`generate_plink_benchmark.py` creates a deterministic PLINK 1 dataset with a
polygenic quantitative phenotype. It requires Python 3 and NumPy.

For storage and streaming measurements at the intended 500,000-sample by
1,000,000-variant scale, template mode avoids drawing 500 billion genotypes in
Python while preserving the exact packed-matrix dimensions:

```sh
python3 benchmarks/generate_plink_benchmark.py /tmp/bolt-target \
  --samples 500000 --variants 1000000 --template-variants 2048 \
  --causal-variants 256 --batch-variants 32 --min-maf 0.01

plink2 --bfile /tmp/bolt-target --make-pgen --out /tmp/bolt-target-pgen \
  --threads 6
```

Each template is independently generated, and subsequent copies rotate packed
four-sample blocks. This is an exact physical-scale I/O and memory benchmark,
not a simulation of one million independent markers. The command above wrote a
116.415 GiB BED in 5:15 on the A100 VM. PLINK 2 converted it to a 93.466 GiB
PGEN in 7:30.

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

## File-backed CUDA host cache

At the target 500,000-sample stride, a 131,072-SNP packed matrix occupies
16.384 GB. On the A100 VM's 440 MB/s persistent disk, a cold uncached matrix
pass took 37.264 seconds, while the same pass with warm pages took 4.047
seconds. A separate background-copy worker changed the cold result by only
0.2% and the warm median by 0.9%, confirming that the existing two-buffer
pipeline already overlaps the prior block's GPU work with mmap faults and host
copies.

Retaining a bounded part of the device-uncached range in ordinary host RAM
reduces the bytes reread on every subsequent pass:

| Retained host cache | Repeated cold-file pass | Reduction |
| ---: | ---: | ---: |
| 0 GiB | 37.264 s | — |
| 3.934 GiB | 28.444 s | 23.7% |
| 7.987 GiB | 19.368 s | 48.0% |

The retained range is filled during the first marker-initialization scan, so
its disk read is a one-time cost. The automatic limit is one half of physical
RAM (about 41.7 GiB on the 83 GiB A100 VM); use
`--cudaHostCacheGiB=0` for the previous minimum-memory behavior. An end-to-end
file-backed PGEN run with half of its matrix retained produced the same
byte-identical Stage 1 model as both the no-host-cache run and the prior CUDA
streaming implementation.

On the exact 500,000-sample by 1,000,000-variant fixture, increasing retained
host genotypes from 20.86 to 39.94 GiB reduced the CUDA LINREG pass from 226.0
to 180.6 seconds (20.1%). PGEN setup was unchanged (528.7 versus 528.9 seconds),
as was the first marker-initialization pass (396.3 versus 397.8 seconds). The
larger run retained about 40 GiB of reclaimable file pages in addition to the
host cache, still had about 40 GiB available, and used no swap.

LINREG is used here only as a controlled one-traversal timer. The intended
production output is `P_BOLT_LMM`; the same retained device and host ranges are
reused by variance fitting, cross-validation, and final spike-and-slab scoring.
The optional Stage 1 `--noLinreg` flag removes this traversal while preserving
the default behavior when absent. On the medium full-model fixture it reduced
the model artifact from 51,945,840 to 50,897,210 bytes, and Stage 2 output
including `P_BOLT_LMM` remained byte-identical to the default path.

## Larger CUDA device cache

Reducing the decoded CUDA block from 1,024 to 512 SNPs lowers the target-scale
block allocation from 3.815 to 1.907 GiB. Cache sizing keeps an additional
3 GiB free, as well as one decoded block, because the main and CV-fold CUDA
objects coexist. At the default largest batch of 52 vectors, the target-scale
batch allocations occupy about 0.44 GiB. This leaves roughly 2.5 GiB beyond the
main batch and second decoded block during a fold.

Based on the measured 27.299 GiB device cache with 1,024-SNP blocks, the smaller
block raises automatic device retention to about 32.114 GiB on this A100. With
the VM's 41.732 GiB automatic host cache, the disk-backed suffix of the exact
116.415 GiB packed matrix falls from 47.384 to 42.569 GiB: 10.2% fewer disk
bytes per repeated genotype traversal. These are memory-accounting projections,
not another full target-scale timing run.

Six interleaved runs on the fully cached 131,072-sample by 16,384-variant PGEN
fixture checked the launch-overhead tradeoff on the complete spike-and-slab
path:

| Measurement (median) | 1,024-SNP blocks | 512-SNP blocks | Change |
| --- | ---: | ---: | ---: |
| End-to-end wall time | 7.10 s | 7.12 s | +0.3% |
| Variance fitting | 0.819 s | 0.826 s | +0.9% |
| Mixture estimation | 0.910 s | 0.901 s | -1.0% |
| Bayesian association scoring | 0.847 s | 0.840 s | -0.8% |

Stage 2 statistics, including `P_BOLT_LMM`, were byte-identical between the two
Step 1 models. This synthetic fixture exercised 6- and 7-iteration variance CG
solves, a 9-iteration infinitesimal solve, a 6-iteration CV fit, and a
5-iteration final Bayesian fit. Its h2 secant search nevertheless converged in
zero steps, so it is a regression and overhead benchmark rather than a proxy
for the difficulty of variance fitting in real biobank data.
