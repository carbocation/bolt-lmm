# BOLT-LMM-XL performance benchmark record

## Current headline snapshot

Last updated 2026-07-18 for the codebase through `acbb21c`. This is the
maintained summary; the remainder of this file is the detailed, chronological
benchmark log. A commit named in a row is the code actually measured. An older
measurement is never silently relabeled as "current."

The Stage 1 and Stage 2 baseline is `a58e7c3`, the first split-stage commit. It
is essentially upstream/v2.5 with the Stage 1 artifact boundary added, so both
sides perform the same split workflow. The REML baseline is upstream/v2.5
itself at `fa732f8`.

CPU runs used all six physical Xeon cores, threaded oneMKL, and
`--numThreads=6`. CUDA runs used the same six host cores plus the A100. SMT was
disabled. Except for the marked target-scale row, values are medians of three
external end-to-end wall measurements in seconds; lower is better. The two
speedup columns use the upstream-equivalent six-core time as their denominator.

| Analysis and representative workload | Upstream-equivalent CPU, 6 cores | Fork CPU, 6 cores | Fork CUDA, A100 | Fork CPU speedup | A100 speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 1, synthetic N=32,768, M=16,384, default LINREG | 38.47 | 28.59 | 3.47 | 1.35x | 11.09x |
| Stage 1, real-LD-derived direct PGEN, N=500,000, M=700,000 | — | — | 4,553* | — | — |
| Stage 2, BED, N=131,072, M=16,384, 1 basis + 2 statistics | 28.05 | 1.13 | 1.18 | 24.82x | 23.77x |
| Stage 2, BED, N=131,072, M=16,384, 21 bases + 2 statistics | 33.08 | 2.94 | 1.78 | 11.25x | 18.58x |
| REML, real-LD N=8,192, M=16,384, default refinement | 20.38 | 18.67 | 2.33 | 1.09x | 8.75x |

On the small direct Stage 2 workload, the fork's six-core CPU path is 1.04x
faster end-to-end than A100 because CUDA context startup dominates; A100 is
1.65x faster than six-core CPU on the denser 21-basis workload. A100 is 8.24x
faster than fork CPU for Stage 1 and 8.01x faster for REML on these fixtures.
These ratios are workload-specific and must not be extrapolated to the target
row. The starred value is one cold-cache end-to-end run, not a three-run
median. It used six physical host cores, direct PGEN input, ephemeral scratch,
default scientific convergence, default LINREG, and no persistent Step 0
cache. A matched pre-parallel-copy A100 run took 4,846 seconds. The two target
models were byte-identical. Full target CPU baselines were not run.

All Stage 1 models produced the same byte-identical final Stage 2 statistics.
All direct and dense Stage 2 output files were also byte-identical. REML used
the same CG convergence sequence and printed identical variance estimates and
standard errors in all three configurations. The matched raw repetitions are
in [`a100_production_headline.tsv`](results/a100_production_headline.tsv).
The target Stage 1 measurement is in
[`a100_stage1_target_700k.tsv`](results/a100_stage1_target_700k.tsv).
Single-thread diagnostics, additional formats, target-stride probes, and
historical comparison points remain below.

## Benchmark setup and historical log

Everything below is retained as a performance notebook: it records individual
optimization checkpoints, realistic and synthetic fixtures, negative results,
and target-scale constraints. Within this historical log, words such as
"current" describe the comparison point at that location; the accompanying
commit hash and raw result file are authoritative.

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

A binary compiled with CUDA as described in [`BUILD.md`](../BUILD.md) uses the
GPU automatically for Stage 1. Pass `--no-cuda` to benchmark its CPU path
instead; [`USAGE.md`](../USAGE.md) documents the runtime behavior.

## Historical end-to-end Stage 1 comparison at `f337b92`

The first split-Stage-1 commit, `a58e7c3d2f3ca651ba16a39063f25b8e449087e0`,
provides the original baseline. It is essentially upstream BOLT-LMM v2.5 with
the Stage 1/2 boundary already present, so its timing includes the same model
fit and Stage 1 artifact write as the current code.

The original baseline and fork CPU-only and CUDA binaries at `f337b92` were built with
GCC 11.4, `-O3 -march=native`, LP64 sequential oneMKL, and one analysis thread.
The fork CPU build disabled OpenMP; the original OpenMP build was restricted
to one thread. The CUDA build targeted compute capability 8.0 and used the same
native host settings. Three repetitions of each binary were interleaved and
pinned to the same vCPU on the Xeon/A100 VM. All used the 32,768-sample by
16,384-SNP BED fixture and the runtime arguments above, including default
LINREG.

| Stage 1 implementation | Median wall time | Speedup vs original | Time reduction |
| --- | ---: | ---: | ---: |
| Original split baseline (`a58e7c3`) | 129.06 s | 1.00x | — |
| Fork CPU-only (`f337b92`) | 112.67 s | 1.15x | 12.7% |
| Fork CUDA (`f337b92`) | 3.55 s | 36.35x | 97.2% |

CUDA was 31.74x faster than the optimized CPU-only build. Median phase times
show where the end-to-end changes came from. The nine raw measurements are in
`results/a100_stage1_headline.tsv`.

| Phase | Original | Fork CPU at `f337b92` | Fork CUDA at `f337b92` |
| --- | ---: | ---: | ---: |
| Genotype input/QC | 3.730 s | 0.683 s | 0.654 s |
| Marker/covariate initialization | 2.921 s | 2.581 s | 0.457 s |
| LINREG | 1.040 s | 0.911 s | 0.010 s |
| Variance fitting | 37.524 s | 28.266 s | 0.444 s |
| Infinitesimal association scoring | 44.418 s | 41.294 s | 0.426 s |
| Mixture estimation | 24.304 s | 24.413 s | 0.586 s |
| Bayesian association scoring | 14.955 s | 14.506 s | 0.526 s |

All three runs followed identical convergence paths: the variance estimate
used two secant steps, the infinitesimal solve used 10 CG iterations, CV used
9 iterations, and final Bayesian scoring used 6 iterations. Loading each model
with the same Stage 2 binary produced byte-identical statistics files,
including `P_BOLT_LMM`.

This deterministic synthetic workload is small enough to fit completely in GPU
memory. It is useful for an end-to-end compute comparison but does not model
the difficulty of variance fitting in real biobank phenotypes or the I/O costs
of 500,000 samples by one million SNPs. The target-scale sections below report
those storage and streaming constraints separately without extrapolating this
36.35x figure to them.

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

The optional `cuda_step1_stream_benchmark` build target isolates the steady
Bayesian block API at the production 500,000-sample stride without fitting a
model. Its default device-resident run and `--stream` run use the same inputs
and checksum; `--variants` and `--iterations` bound the matrix and repetition
counts:

```sh
cmake --build build/cuda --target cuda_step1_stream_benchmark -j
build/cuda/cuda_step1_stream_benchmark --variants 8192 --iterations 20
build/cuda/cuda_step1_stream_benchmark --stream \
  --variants 8192 --iterations 20
```

## Target-scale Stage 1 A100 checkpoint

The target checkpoint used N=500,000 and M=700,000, matching the scale quoted
for UK Biobank in the upstream manual. Genotypes were block-resampled from the
1KG Phase 3 DRAGEN-derived panel with 32-variant donor blocks, so the fixture
retains local LD, allele-frequency variation, missingness, and population
structure without representing the synthetic individuals as real people. The
phenotype used h2=0.3 and 8,192 causal variants, and `SUPERPOP` was fitted as a
categorical covariate.

Both A100 runs read hardcall PGEN directly and built only ephemeral scratch.
They used six physical Xeon cores, default LINREG, default convergence values,
`--lmmForceNonInf`, `--LDscoresUseChip`, and no scientific shortcut flags. The
device cache held 275,968 variants (32.13 GiB); ordinary host RAM retained the
remaining 424,032 variants (49.36 GiB). No iterative genotype pass reread disk.

| Phase | Before parallel host copy (`c4c5670`) | Parallel host copy (`b47cfbb`) | Reduction |
| --- | ---: | ---: | ---: |
| PGEN setup | 176.719 s | 177.173 s | -0.3% |
| Marker/covariate initialization | 292.510 s | 289.578 s | 1.0% |
| LINREG | 15.409 s | 7.187 s | 53.4% |
| Variance fitting | 471.867 s | 271.735 s | 42.4% |
| Infinitesimal association statistics | 195.370 s | 127.557 s | 34.7% |
| Chip LD scores | 259.252 s | 246.676 s | 4.9% |
| Mixture-parameter CV | 1,668.130 s | 1,669.160 s | -0.1% |
| Final spike-and-slab fit | 1,756.210 s | 1,752.950 s | 0.2% |
| Complete analysis | 4,841.940 s | 4,548.480 s | 6.1% |
| External wall time | 4,846 s | 4,553 s | 6.0% |

Both runs used the same 8/15/9 variance-component CG counts, 13 infinitesimal
CG iterations, 75 CV iterations, 76 final iterations, and selected f2=0.1 and
p=0.02. Their 205 MiB Stage 1 models were byte-identical, with SHA-256
`d2cae1cd5b8d50b99e4e4f40aab61a4bba6c2ff6ce14428edbef94d01c58cd73`.

A matched 300-second CV trace explains the intermittent utilization seen in
`nvidia-smi`. Before parallel host copies, mean/median GPU utilization was
46.28%/37%, with 239 of 300 samples below 50%. At `b47cfbb` it was
45.95%/36%, with 240 samples below 50%. Each iteration alternated between a
short 85-88% device-cached prefix and a long 36-38% host-streamed suffix.
Parallel copying helps the smaller-batch phases above but does not change this
CV pattern.

Four follow-ups were measured and removed. A persistent background worker that
overlapped pageable host copy and H2D with GPU work made the six-thread A100
streaming microbenchmark 5.4% slower and the four-thread result 8.6% slower.
Reusable pinned bounce buffers for the small Bayesian products and updates
changed the six-thread median by only 0.2%. The background-worker prototype
passed all CUDA tests and produced a byte-identical complete real-LD Stage 1
model on T4 before removal.

The device-update alternative was also implemented and tested rather than
inferred. Keeping coefficients and Bayesian factors on the GPU and doing the
ordered 512-marker spike-and-slab recurrence there was 44.1% slower with one
CUDA block handling all candidates. Assigning one CUDA block to each of the 18
independent candidates reduced that penalty, but was still 15.2% slower than
the retained CPU-update round trip (0.195092 versus 0.169298 seconds per
iteration). A complete real-LD T4 run was 5.8% slower end to end and introduced
floating-point differences in the model artifact despite matching printed
calibration and convergence counts. The prototype was therefore removed.

The remaining CV cost is structural: a 700,000-variant traversal has 1,368
512-variant blocks, and the spike-and-slab update returns products to the CPU
and sends coefficient updates back for every block. The recurrence is ordered
within each candidate, while the usual batch has only 18 independent
candidates—too little parallel work for the tested A100 device formulation.
Staging changes and this direct device port did not improve that path. Raw
target timings and rejected experiments are in
[`a100_stage1_target_700k.tsv`](results/a100_stage1_target_700k.tsv) and
[`a100_cuda_staging_experiments.tsv`](results/a100_cuda_staging_experiments.tsv).

At `acbb21c`, the retained ordinary-RAM genotype cache is page-aligned and
registered with CUDA. Blocks inside that cache transfer directly to the
device, avoiding the pageable-to-pinned copy; sources outside it retain the
existing two-buffer fallback. If a driver or host-memory policy refuses
registration, execution also falls back to the previous staged path.

On A100, the production-stride Bayesian streaming microbenchmark fell from
0.169298 to 0.132460 seconds per iteration (21.8%) with identical checksums.
The matched T4 result was effectively neutral at 4.753720 versus 4.710844
seconds (0.9% faster). The complete 49.36 GiB target host suffix registered
successfully. On the N=500,000, M=700,000 PGEN fixture, LINREG fell from
7.18732 to 6.34372 seconds (11.7%), while initialization was 291.501 versus
289.578 seconds, max RSS remained about 85.0 GB, and neither run used swap.
The calibration mean and lambdaGC were identical. The target CV/final and
end-to-end effect require a new complete run; these phase measurements are not
used to revise the 4,553-second headline above. Raw measurements are in
[`a100_cuda_registered_host_cache.tsv`](results/a100_cuda_registered_host_cache.tsv).

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
its disk read is a one-time cost. The automatic limit is 72% of physical RAM
(about 60.1 GiB on the 83 GiB A100 VM); use
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

After the reusable Step 0 artifact and larger device cache were in place, a
paired current-binary run isolated a further host-cache increase on the same
exact target fixture. Retaining 41.723 GiB on the host left a 42.565 GiB disk
suffix and took 152.343 seconds for the controlled LINREG traversal. Retaining
59.962 GiB left 24.326 GiB on disk and took 98.362 seconds, a 35.4% reduction;
marker initialization was neutral at 392.9 versus 398.9 seconds. The larger
run had about 20 GiB available at peak and used no swap. The 72% automatic
limit matches this stress-tested point, leaving a roughly 24.2 GiB suffix. Its
98.4-second traversal is about 11.6% faster than the 111-second traversal
projected for the previous two-thirds policy and 35.4% faster than the paired
half-RAM control. Raw measurements are in
`results/a100_stage1_target_cache.tsv`.

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
the VM's roughly 60.1 GiB automatic host cache, the disk-backed suffix of the
exact 116.415 GiB packed matrix falls to about 24.2 GiB. The preceding section
reports the paired target-scale traversal measurements.

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

## Real-reference benchmark ladder

`generate_real_plink_benchmark.py` builds deterministic PLINK 1 performance
fixtures from an external real-reference prefix. The source and generated
genotypes stay outside the repository. Native-sample mode preserves the source
sample IDs and genotypes. Expansion mode assigns samples to the source
population groups and draws a new donor within the same group every short
variant block. It preserves real allele frequencies, missingness, population
structure, and local LD, but the resulting people are explicitly synthetic.

For a small correctness and convergence fixture using the real samples:

```sh
python3 benchmarks/generate_real_plink_benchmark.py \
  /data/real-reference /scratch/bolt-real-native \
  --source-psam /data/real-reference.psam \
  --identity-samples --variants 16384 \
  --causal-variants 1024 --heritability 0.5
```

For a target-stride compute fixture with a diffuse polygenic phenotype:

```sh
python3 benchmarks/generate_real_plink_benchmark.py \
  /data/real-reference /scratch/bolt-real-expanded \
  --source-psam /data/real-reference.psam \
  --samples 131072 --variants 16384 --ld-block-variants 32 \
  --causal-variants 8192 --heritability 0.3 --threads 6
```

Expansion workers gather and pack independent variant blocks concurrently.
Donor draws, output writes, and phenotype accumulation remain in their original
order, so one- and multi-threaded fixtures are byte-identical. On the A100
host, a 500,000-sample by 4,096-variant target-stride slice fell from 35.13 to
18.75 seconds with five workers (1.87x); the BED and phenotype hashes matched.

The reference used on the A100 was the 1KG Phase 3 DRAGEN-derived HapMap3
panel: 2,573 samples and 1,107,021 variants. The 16,384-variant native fixture
needed 8/12/11 variance-component CG iterations, 16 infinitesimal CG
iterations, 21 cross-validation iterations, and 14 final spike-and-slab
iterations. It completed Stage 1 in 2.47 seconds. This is materially harder
than the earlier independent synthetic fixture and confirms that real
genotypes change convergence even at small sample count. Loading its
`--noLinreg` model in Stage 2 produced both `P_BOLT_LMM_INF` and
`P_BOLT_LMM` for the real DRAGEN-style variant IDs.

The 131,072-sample block-bootstrap fixture converged cleanly in 32.94 seconds,
with variance CG counts of 23/41/24, 35 infinitesimal CG iterations, 90 CV
iterations, and 72 final spike-and-slab iterations. It is a useful
throughput/convergence stress test, not a replacement for a genuinely large
cohort. A 256-variant bootstrap-block experiment was rejected as a benchmark:
it made the GRM poorly conditioned and cross-validation reached its 250-
iteration limit.

## BOLT-REML benchmark ladder

BOLT-REML is benchmarked separately from the Stage 1 association-model fit.
The REML ladder uses the real-reference fixtures above, not independent random
genotypes. The 32,768-sample rung retains 32-variant donor blocks from the 1KG
Phase 3 DRAGEN-derived reference. The smaller refined and N=500,000 rungs
resample those real-derived blocks again; they retain local LD but are not
substitutes for a genuinely observed biobank cohort.

`prepare_reml_benchmark.py` layers correlated traits and chromosome-partitioned
variance-component labels onto an existing PLINK 1 prefix without touching the
genotypes:

```sh
python3 benchmarks/prepare_reml_benchmark.py \
  /scratch/bolt-real-expanded /scratch/bolt-reml-d2-vc2 \
  --traits 2 --trait-correlation 0.4 --variance-components 2
```

The generated traits and VC split are deterministic performance/convergence
inputs, not a claim that the added traits follow a realistic genetic-correlation
model. The source phenotype remains diffuse and polygenic, and all timing rungs
exercise genotype LD inherited from the real reference.

Final A100 measurements use GCC 11.4, native x86 code, sequential oneMKL, and
one pinned host CPU (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`--numThreads=1`). "Coarse" rows specify `--remlNoRefine`; the refined row uses
the scientific default of 15 coarse plus 100 refinement Monte Carlo trials.
Times are BOLT's analysis timer:

| Workload | upstream/v2.5 CPU | fork CPU | fork CUDA | CUDA vs fork CPU | CUDA vs upstream |
| --- | ---: | ---: | ---: | ---: | ---: |
| N=32,768, M=16,384, D=1, one VC, coarse 15 trials | 118.81 s | 96.55 s | 4.30 s | 22.4x | 27.6x |
| N=8,192, M=16,384, D=1, one VC, default refinement | 58.90 s | 53.75 s | 3.95 s | 13.6x | 14.9x |
| N=500,000, M=1,024, D=1, one VC, coarse 3 trials | -- | 74.09 s | 7.13 s | 10.4x | -- |
| N=32,768, M=16,384, D=2, two VCs, coarse 3 trials | -- | 251.83 s | 7.57 s | 33.3x | -- |

Converting the first rung to PGEN and running it directly, without a Stage 0
cache, took 3.85 seconds with CUDA on warm filesystem pages. Its estimates and
CG counts matched the BED and upstream rows at every printed digit. This small
matrix fit entirely in the A100 device cache; the low-level suite separately
checks uncached streaming and retained-host-cache execution.

The N=500,000 row validates target sample stride, vector transfers, and memory
behavior. Its 1,024-SNP GRM is deliberately bounded and low-rank, so it must not
be presented as an end-to-end N=500,000 by M=1,000,000 scientific analysis.
Raw measurements and provenance are recorded in
`results/a100_reml.tsv`.

CUDA now covers the main AI-REML conjugate-gradient matrix products, derivative
products, arbitrary trait/VC coefficient matrices, and genetic Monte Carlo
pseudo-phenotype generation. It retains Boost's upstream RNG and draw order on
the CPU. CPU-only builds and CUDA builds run with `--no-cuda` keep the upstream
CPU REML implementation; no estimator, starting values, tolerance, Monte Carlo
count, refinement default, or trust-region logic changed.

CUDA changes floating-point reduction order. On every rung above, CG iteration
counts and all reported variance matrices, standard errors, sigma2 values, h2
values, and genetic/residual correlations matched the CPU result at every
printed digit. Low-level CUDA tests independently cover cached, streamed, and
host-cache paths and require absolute matrix error at most 1e-11.

The full reference panel was also used for bounded ingest/QC measurements.
With warm filesystem pages, BOLT's setup timer was 13.652 seconds for the
620.5 MB PGEN plus its uncompressed PVAR, versus 5.521 seconds for the
equivalent 712.9 MB BED plus BIM. Two guard-bounded repetitions were stable at
13.85/13.88 seconds wall time for PGEN and 5.73/5.74 seconds for BED.

That comparison includes metadata parsing: the DRAGEN PVAR is 1.386 GB
uncompressed because of its INFO field. Repeating the PGEN measurement with a
45.7 MB PVAR containing only BIM-equivalent required columns took 4.51/4.50
seconds. The 9.36-second penalty on this small panel was therefore attributable
to copying and tokenizing unused metadata, not hardcall conversion.

The selected-field PVAR parser removes that penalty without changing parsed
variant metadata or skipping malformed-row validation. On the original 1.386
GB PVAR, paired guard-bounded old-parser runs took 17.07/15.45 seconds and the
new parser took 5.79/5.76 seconds: medians of 16.26 and 5.775 seconds, or a
64.5% reduction. The optimized full-metadata path is now close to the compact
metadata lower bound. A convergent 131,072-sample real-genotype Stage 1 run
also produced a model byte-for-byte identical to the pre-change model.

Metadata parsing is O(M), while hardcall conversion is O(NM). The separate
exact target-shape benchmark's 528.8-second PGEN setup still motivates a
persistent reusable converted-PGEN cache for repeated biobank analyses when
that cache is already colocated with the compute job. The source `.pvar.zst`
was decompressed on the benchmark host because BOLT currently accepts `.pvar`
or `.pvar.gz`, not `.pvar.zst`.

The full 1.107-million-marker panel is intentionally not an end-to-end model
fixture with only 2,573 samples: after raising `--maxModelSnps`, its
variational-Bayes fit diverged in the extreme M/N regime. Full-model timings
therefore use the convergent 16,384-variant fixtures, while the full panel is
used only for parser, hardcall-conversion, QC, and storage measurements. Raw
measurements are in `results/a100_stage1_real_genotypes.tsv`.

## Native x86 PGEN hardcall conversion

A pinned-core profile at the production 500,000-sample stride attributed
84.5% of direct-PGEN setup samples to BOLT's byte-at-a-time conversion, allele
sum, and missing-call loop; pgenlib decompression was the much smaller part.
Native AVX2 builds now convert 32 packed bytes (128 genotypes) at once with an
exact bitwise PGEN-to-BED mapping. Nibble shuffles accumulate allele and
missing counts, while the existing scalar path handles tails, missing-index
collection, and builds without AVX2.

The bounded 500,000-sample by 100,000-variant container keeps the target sample
stride and avoids repeatedly profiling the full million-marker input. Across
two order-reversed pairs, guard-bounded direct ingestion fell from a median of
28.145 to 15.035 seconds, a 46.6% reduction. Durable Step 0 construction fell
from 46.60 to 39.54 seconds (15.2%) after including the unchanged 12.5 GB write
and flush. The complete 12,501,765,536-byte baseline and optimized Step 0
artifacts compared byte-for-byte identical. An exhaustive unit test also
covers every packed input byte, including missing calls. On the 131,072-sample
real-genotype PGEN fixture, the reverse-order warm-page pair fell from 1.41 to
0.84 seconds (40.4%).

A follow-up encoded each two-genotype nibble's allele sum and missing count in
one byte. This replaced two lookup tables and four shuffles with one table and
two shuffles. Across an order-reversed pair, the AVX2-v1 median was 15.355
seconds and the combined-statistics median was 14.88 seconds, a further 3.1%
reduction. It is retained despite the small incremental result because the
implementation and dependency chain are simpler.

This speedup applies to direct ingestion on every run and does not assume that
a large converted cache has been transferred to the compute machine. The
conversion work is O(NM); the bounded target-stride result is reported instead
of extrapolating it into an unmeasured full-scale wall time.

## Persistent CPU Step 0 PGEN cache

`--stage=0 --pfile PREFIX --pgenCacheFile FILE` performs PGEN hardcall
conversion and genotype QC once, writing a durable packed artifact directly
without a second full-size copy. Stage 1 accepts the artifact only when a
fingerprint of the PGEN source, parsed PVAR/PSAM identities, sample removal,
variant/model selection, packing dimensions, map-derived metadata, and QC
thresholds matches. It never silently builds or refreshes the artifact. A
CUDA-enabled binary still executes Step 0 entirely on the CPU, allowing
production caches to be prepared away from accelerator nodes when both see the
same high-throughput storage.

On the 131,072-sample by 16,384-variant real-reference fixture, Step 0 took
2.97 seconds once. Reuse reduced Stage 1 `SnpData` setup from 1.593 to 0.213
seconds (86.6%) and end-to-end CUDA wall time from 31.95 to 30.22 seconds
(5.4%). The direct-PGEN and persistent-cache Stage 1 model artifacts were
byte-identical.

The exact 500,000-sample by 1,000,000-variant artifact contains 116.4 GiB of
packed hardcalls. Building and durably flushing it took 522.53 seconds on one
pinned Xeon core, compared with the prior 528.8-second per-run PGEN conversion.
Two guard-bounded Stage 1 loads took 2.50 and 2.47 seconds with about 605 MiB
peak RSS: a 213x setup speedup and roughly 8 minutes 46 seconds saved on every
reuse. One-MiB regions at the beginning, middle, and end of the payload matched
the original BED-equivalent hardcalls byte-for-byte. These target-shape runs
measure setup only; they do not attempt a full million-variant model fit.

The artifact is not intended to be copied from a separate machine for every
run. At 116.4 GiB, a one-way transfer must exceed about 226 MiB/s to finish
within the 526-second conversion saving; a build-host upload followed by a
compute-host download must exceed about 453 MiB/s across both sequential
copies. These break-even rates omit transfer setup and storage costs. Direct
PGEN ingestion is therefore faster whenever the cache is not already local or
available through shared high-throughput storage.

## Variance-component CG warm starts

This optimization is now explicitly enabled with `--warmStartVarianceCG`.
Cold starts matching the upstream v2.5 numerical path are the default because
the changed CG trajectory can change output at the existing residual tolerance.
The measurements below were collected with the warm start enabled.

The h2 secant search repeatedly solves closely related systems at nearby
log(delta) values. Retaining the previous solution as the next conjugate-
gradient start reduced the real 131,072-sample CUDA fixture's variance phase
from 4.871 to 4.249 seconds (12.8%) and end-to-end wall time from 32.63 to
31.95 seconds (2.1%). The logged iteration counts changed from 23/41/24 to
23/32/19; the latter two warm solves each require one additional, unlogged
matrix application to form their initial residual, so the net count is 88 to
76 genotype matrix passes.

A separate CPU-only measurement used one pinned Xeon core and a bounded
32,768-sample fixture generated from the same real reference. Variance fitting
fell from 47.09 to 43.07 seconds (8.5%) and end-to-end wall time from 246.47 to
241.63 seconds (2.0%). Its net genotype matrix-pass count fell from 47 to 43.
Reuse is limited to log(delta) changes of at most 2; larger secant jumps retain
the cold-start behavior because computing a warm residual itself costs a full
matrix application.

Both implementations fitted h2g=0.262 on the CUDA fixture. Across 16,384
Stage 2 variants, five printed `P_BOLT_LMM_INF` and three printed `P_BOLT_LMM`
values changed. The maximum absolute change was 0.001; the largest relative
change was from 1.3e-8 to 1.4e-8, and no genome-wide-significance classification
changed. Beta changes were at most 1e-6 and standard-error changes at most
1e-7. These are the numerical effects of reaching the existing CG residual
criterion through a different Krylov path; the model and output schema are
otherwise unchanged.

## Final variational-Bayes warm start

This optimization is now explicitly enabled with `--warmStartFinalVB`. The
upstream v2.5 zero initialization is the default because variational-Bayes
initialization can change `P_BOLT_LMM` at the existing approximate-likelihood
tolerance. The measurements below were collected with the warm start enabled.

When mixture parameters are selected by cross-validation and
`--warmStartFinalVB` is supplied, the fitted effects from the winning model are
reused to initialize the final LOCO spike-and-slab fit. Effects are retained in
raw per-allele units, averaged across every computed fold, and converted to the
full-cohort normalization.
Each LOCO model applies its own SNP mask, and BOLT explicitly constructs the
matching initial `y-X*beta` residual. This initialization costs one genotype
pass and is used only for variational Bayes, not MCMC. Fits with user-supplied
mixture parameters retain the cold start because no cross-validation effects
exist to reuse.

On the warmed 131,072-sample CUDA fixture, the final fit converged in 50 rather
than 72 iterations and took 6.49 rather than 9.37 seconds, a 30.7% phase
reduction including initialization. Paired wall time fell from 30.38 to 27.65
seconds (9.0%). With `--approxLLtol=0.001`, the final fit similarly fell from
130 to 95 iterations and from 16.64 to 11.45 seconds (31.2%).

The CPU-only fixture on one pinned Xeon core converged in 17 rather than 33
final iterations. Its final-fit time fell from 60.97 to 33.59 seconds (44.9%),
and end-to-end Stage 1 wall time fell from 241.63 to 214.06 seconds (11.4%).
Cross-validation time was unchanged within noise (61.59 versus 61.83 seconds).

At the default tolerance, both CUDA and CPU comparisons produced identical
printed beta, standard error, and `P_BOLT_LMM_INF` columns across all 16,384
Stage 2 variants. Printed `P_BOLT_LMM` values changed for 2,146 CUDA variants
and 1,163 CPU variants; the maximum absolute difference was 0.01, the largest
relative CUDA change was 10%, from 1.0e-7 to 1.1e-7, and no value crossed 5e-8.
These small differences reflect the existing approximate-likelihood stopping
criterion being reached from a different, much closer starting point.

## CUDA Bayesian residual transfers

The CUDA variational-Bayes loop keeps its residual vectors on the device.
Previously, every iteration downloaded all residuals and then uploaded the
same values again, even when the CPU had only read them to evaluate the
existing convergence criterion. It also continued downloading rows belonging
to models that had already converged. The optimized path skips an upload until
the CPU actually permutes residual rows and transfers only the still-active
prefix. The arithmetic and CPU convergence calculation are unchanged.

Across two order-reversed pairs on the 131,072-sample fixture, median mixture
fitting fell from 9.609 to 9.141 seconds (4.9%), final Bayesian fitting fell
from 6.477 to 6.208 seconds (4.2%), and wall time fell from 27.42 to 26.895
seconds (1.9%). All four Stage 1 model artifacts were byte-for-byte identical.
The small end-to-end gain is retained because the implementation removes only
redundant transfers and adds no alternate numerical path.

## Bayesian SNP-factor precomputation

The spike-and-slab scalar update previously recomputed two mixture standard
deviations, two shrinkage factors, two posterior variances, and two posterior-
variance log ratios for every SNP, model, and variational-Bayes iteration.
These values depend only on the fixed variance parameters and the SNP norm.
They are now evaluated once per fit with the same expressions and stored for
reuse; update order, convergence criteria, and all floating-point formulas are
unchanged.

Across two order-reversed pairs on top of the CUDA residual-transfer change,
median mixture fitting fell from 9.154 to 8.977 seconds (1.9%), final
spike-and-slab fitting fell from 6.228 to 6.003 seconds (3.6%), and wall time
fell from 26.855 to 26.335 seconds (1.9%). All four Stage 1 model artifacts
were byte-for-byte identical. A bounded 8,192-sample CPU-only pair was noisier
end to end, but its final spike-and-slab phase fell from 5.93 to 4.05 seconds
(31.7%) and its model was also byte-identical.

Precomputing the two log ratios on top of the first six factors reduced the
same current CUDA workload's mixture phase from 8.970 to 8.656 seconds (3.5%),
final fit from 6.003 to 5.773 seconds (3.8%), and wall time from 26.390 to
25.915 seconds (1.8%). The four incremental comparison models were again
byte-identical. Relative to recomputing all eight invariants, the cumulative
reductions are 5.4% for mixture fitting, 7.3% for final Bayes, and 3.5% for
wall time.

The factor table uses 64 bytes per SNP-model pair and exists only during a
fit. At one million Stage 1 SNPs, the default 18-pair first CV fold requires
about 1.07 GiB; a 22-chromosome final LOCO fit requires about 1.31 GiB. This
is host working memory, not a persistent artifact, and therefore introduces
no cache-transfer premise.

## Native x86 AVX genotype expansion

Native x86 builds now expand each four-sample packed genotype lookup with one
256-bit AVX load/store, and apply cross-validation sample masks with one AVX
multiply. Portable x86 builds retain the existing SSE2 path. On the pinned
32,768-sample CPU fixture, cross-validation fell from 61.83 to 60.50 seconds
(2.2%), final VB fell from 33.59 to 33.27 seconds (1.0%), and end-to-end Stage 1
fell from 214.06 to 212.75 seconds (0.6%). The Stage 1 model artifact was
byte-identical. This is retained despite the small total gain because the
implementation is a narrow intrinsic-width substitution and production runs
are expected to use AVX-capable x86 hosts.

## Current single-core CPU ceiling audit

After the PGEN, warm-start, and Bayesian-factor changes above, a fresh pinned
profile used the 8,192-sample by 16,384-variant real-reference fixture with
`--noLinreg`, `--warmStartVarianceCG`, and `--warmStartFinalVB`. End-to-end
Stage 1 took 35.26 seconds: 8.64 seconds for variance fitting, 9.86 seconds for
infinitesimal scoring, 11.72 seconds for mixture estimation, and 4.05 seconds
for final spike-and-slab scoring. This historical performance profile therefore
does not represent the now-default upstream-compatible cold-start path.

The main executable's profile attributed 1,425,498 calls to packed-genotype
expansion: 589,824 from the CG matrix multiply and 671,744 from variational
Bayes. Expansion accounted for 72.9% of samples collected in the executable,
but that percentage excludes time inside dynamically linked oneMKL and must
not be interpreted as its share of wall time. Four output-preservation
experiments tested the remaining local opportunities:

| Experiment | Bounded result | Decision |
| --- | --- | --- |
| Fuse CPU marker centering and normalization | Initialization 0.656 to 0.489 s; wall 35.01 to 34.81 s | Rejected: Stage 2 statistics were byte-identical, but the internal model differed and moderate duplicated decode logic bought only 0.6% wall time |
| Address active Bayesian models through their compacted indices | Wall 34.95 to 35.19 s | Rejected: byte-identical model, slightly slower |
| Manually unroll native AVX lookup expansion four ways | Wall 35.03 to 36.06 s | Rejected: byte-identical model, 2.9% slower |
| Decode with AVX-512 variable permutation instead of a per-SNP lookup table | Wall 34.90 to 33.62 s at N=8,192 | Rejected at target stride: the 500,000-sample initialization median changed from 9.937 to 9.990 s |

The AVX-512 check is important because it prevents a small-sample benchmark
from selecting the wrong production path. Its target-stride comparison used
4,096 model markers, so each initialization expanded 16.384 GB of doubles;
two order-reversed pairs were neutral to slightly unfavorable despite the
small-fixture win. Raw measurements are in
`results/a100_stage1_cpu_ceiling.tsv`.

The remaining substantial CPU opportunity is to avoid materializing dense
double genotype blocks and multiply directly from packed hardcalls. That is a
different matrix arithmetic and summation path, not an obvious local
optimization, and therefore needs a separate scientific-output validation
before it could become a default. Under the requirement that the current
numerical path and real-data output remain unchanged, this audit found no
untried single-core CPU change with a credible 10% end-to-end gain.

## Single-threaded Stage 2 comparison

The Stage 2 CPU audit used the 131,072-sample by 16,384-variant expanded
real-reference fixture. Its genotypes retain short-range LD, allele-frequency
structure, and population structure from the 1000 Genomes source. The BED and
PGEN contain identical hardcalls. PLINK 2 exported those hardcalls to an 8-bit,
layout-2, zlib-compressed BGEN, so this BGEN workload exercises the production
probability decoder and compression path but not uncertain imputation dosages.

Two Stage 1 model shapes cover the CPU dispatch points. The direct-packed case
contains one covariate-basis vector and the two mixed-model statistics. The
batched case contains 21 covariate-basis vectors and the same two statistics.
Neither contains LINREG, and all reported files contain the two requested
mixed-model p-values. For hardcalls, the current code selects direct packed
scoring through six combined basis/statistic vectors and dense DGEMM batching
above that threshold. Single-threaded BGEN retains fused scalar scoring below
five vectors and uses dense batching from five vectors onward.

The original executable is the exact first split-stage commit,
`a58e7c3d2f3ca651ba16a39063f25b8e449087e0`. Both it and the current executable
were built with GCC 11.4, `-O3 -march=native`, LP64 sequential oneMKL, no CUDA,
and no OpenMP execution. Runs were pinned to one Xeon vCPU. The current build
also had auto-detected libdeflate enabled for zlib BGEN. Current Stage 1 files
use a faster v2 checksum; for the old loader, the same serialized payload was
given a benchmark-only v1 header and recomputed FNV checksum. No sample,
covariate, residual, chunk, scale-factor, or statistic bytes were changed.

| Model and comparison | Original Stage 2 | Current Stage 2 | Speedup | Reduction |
| --- | ---: | ---: | ---: | ---: |
| 1 basis + 2 stats: a58 BED → current BED | 28.466 s | 3.448 s | 8.26x | 87.9% |
| 1 basis + 2 stats: a58 BED → current PGEN | 28.466 s | 3.261 s | 8.73x | 88.5% |
| 21 bases + 2 stats: a58 BED → current BED | 48.109 s | 9.841 s | 4.89x | 79.5% |
| 21 bases + 2 stats: a58 BED → current PGEN | 48.109 s | 8.167 s | 5.89x | 83.0% |
| 21 bases + 2 stats: a58 BGEN → current BGEN | 74.403 s | 22.107 s | 3.37x | 70.3% |

These are complete Stage 2 wall times, including the roughly 0.3-0.5-second
model load. The association-readout-only medians were 28.085→3.113 seconds
for direct BED, 47.679→9.438 seconds for batched BED, and 73.956→21.704
seconds for BGEN.

The PGEN rows compare current PGEN with the original BED route over identical
hardcalls because `a58e7c3` did not support PGEN. The BED and BGEN rows are
same-format comparisons. Every complete current output was byte-identical to
the corresponding original output; PGEN was also byte-identical to original
BED.

Within the current threaded-oneMKL build constrained to one thread, the
high-covariate BGEN work separated as follows: scalar scoring with zlib had a
54.659-second median; batching reduced it to 37.930 seconds; eliminating two
redundant sample scans reduced it to 30.362 seconds; and libdeflate reduced it
to 24.730 seconds. Switching the final controlled build to sequential oneMKL
gave the 21.704-second headline above. libdeflate is optional and auto-detected;
builds without it retain the zlib path and the same scientific output.

Raw timings and output hashes are in `results/a100_stage2_cpu.tsv`.

## CUDA Stage 2 hardcall scoring

CUDA-enabled Stage 2 now scores BED and hardcall-PGEN data directly from their
two-bit representation. Allele frequency, missingness, mean centering, raw
norms, covariate products, and mixed-model products are computed on device.
The kernel accepts the Stage 2 input-to-model sample map, so exclusions,
reordering, and phenotype/covariate masks do not require dense CPU genotype
expansion. Models with more than 32 combined basis and statistic vectors retain
the CPU path.

The A100 comparison reused the real-LD 131,072-sample by 16,384-variant fixture
and the same sequential oneMKL CPU control described above. CUDA and CPU were
run from the same CUDA-enabled executable, with `--no-cuda` selecting the CPU
control. Times are complete association readout phases; total Stage 2 also
includes model loading and CUDA context initialization. CUDA times are medians
of three final-code runs; the CPU controls are representative paired runs.

| Model and input | CPU readout | A100 readout | Readout speedup | CPU total | A100 total | Total speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 basis + 2 stats, BED | 3.137 s | 0.294 s | 10.67x | 3.477 s | 0.905 s | 3.84x |
| 21 bases + 2 stats, BED | 9.413 s | 0.821 s | 11.47x | 9.816 s | 1.482 s | 6.63x |
| 21 bases + 2 stats, PGEN | 7.726 s | 1.047 s | 7.38x | 8.153 s | 1.725 s | 4.73x |

Every A100 output was byte-identical to its CPU control. The integration test
also compares CUDA BED and PGEN with the scalar CPU reference after subsetting,
reordering, and masking samples. A lower-level test covers both encodings with
3 and 23 score vectors on A100 and T4.

An exact production-engine probe on the T4 used N=500,000, 256-variant pinned
batches, and identity sample order. After GPU clock warm-up, packed transfer,
allele-frequency and missingness calculation, centering, scoring, and result
transfer projected to 36.3-36.7 seconds per million variants with 3 vectors and
203.9 seconds with 23 vectors. Bitwise population counts compute identity-order
hardcall summaries directly from packed 32-bit words; compared with the prior
sample-wise summary kernel this reduced the warm 3-vector scorer by about 16%
and the 23-vector scorer by about 3.5%. These are scorer-only feasibility
measurements, not end-to-end file timings. A four-way transfer/compute pipeline
slowed the 3-vector case; a two-way pipeline was neutral, so neither was
retained.

Dense BGEN dosage scoring was also implemented and tested experimentally. On
the paired A100 real-LD workload it changed readout only from 24.061 to 23.052
seconds (4.2%). CPU probability decoding and dense host-to-device traffic
dominated, so the added CUDA and pinned-buffer ownership complexity was removed.
BGEN keeps the CPU path: its 3.37x single-thread acceleration is documented in
the preceding section, and production CPU scaling is documented below.

Raw A100 timings are in `results/a100_stage2_cuda.tsv`.

## Multithreaded Stage 2 BGEN scaling

The algorithmic Stage 2 improvements above were established with one thread
before adding production CPU parallelism. The high-dimensional layout-2 BGEN
path now decompresses and decodes independent variants with OpenMP, compacts
any variants surviving QC in input order, and scores each bounded dense batch
with threaded BLAS. Decode and BLAS regions do not overlap, avoiding nested
thread teams. Models below the batching crossover retain the existing fused
per-variant OpenMP path.

The scaling ladder used the same real-LD 131,072-sample by 16,384-variant BGEN
fixture and 21-basis plus two-statistic Stage 1 model as the single-core audit.
The A100 VM exposes six physical Xeon cores and 12 logical CPUs. All outputs
were byte-identical to the scalar and single-thread optimized references.

| CPU path | Threads | Readout | Speedup vs optimized 1-thread sequential build |
| --- | ---: | ---: | ---: |
| Optimized sequential-oneMKL control | 1 | 21.812 s | 1.00x |
| Previous per-variant threaded path | 2 | 21.741 s | 1.00x |
| Previous per-variant threaded path | 4 | 11.173 s | 1.95x |
| Previous per-variant threaded path | 6 | 7.902 s | 2.76x |
| Parallel decode + batched scoring | 2 | 13.668 s | 1.60x |
| Parallel decode + batched scoring | 4 | 6.998 s | 3.12x |
| Parallel decode + batched scoring | 6 | 4.973 s | 4.39x |
| Parallel decode + batched scoring | 12 | 6.076 s | 3.59x |

At two, four, and six threads, retaining batching reduced readout by about 37%
relative to the prior threaded path at the same thread count. Six physical
cores were 18% faster than all 12 SMT threads on this VM. Compared with the
original split-stage BGEN readout of 73.956 seconds, the six-core path is about
14.9x faster. Raw measurements are in
`results/a100_stage2_threads.tsv`.

## Production CPU thread scaling

The single-threaded audits above established the algorithmic improvements
before production parallelism. Stage 1 and REML already divide packed-genotype
blocks with OpenMP and use threaded BLAS. A real-LD 8,192-sample by
16,384-variant cold-start spike-and-slab Stage 1 run scaled from 36.937 seconds
on one thread to 11.944 seconds on the VM's six physical cores (3.09x). Its
variance, infinitesimal, mixture-estimation, and final Bayesian phases all
improved. Runs used `--noLinreg` only to exclude that independent traversal
from the scaling measurement. Thread-count-dependent BLAS reductions changed
some low-order model-artifact bits, but Stage 2 output generated from every
model was byte-identical. The scientific summaries and iteration counts also
matched.

The remaining default CPU marker passes were then made multithreaded without
changing their per-marker arithmetic. Each thread has private genotype and
lookup workspaces; variants still write to their original result indices. On a
bounded target-stride N=500,000 by M=4,096 fixture, the default LINREG pass fell
from 3.633 seconds to 0.637 seconds on six physical cores (5.70x). The new
one-thread path took 3.636 seconds, so the parallel-ready implementation added
no measurable serial overhead. This fixture reads 2.048 billion genotypes but
is deliberately only 1/244 of the target variant count.

A complete real-LD N=8,192 by M=16,384 control used fixed mixture parameters
to exercise LINREG, infinitesimal, and final spike-and-slab marker passes in one
artifact. Against the immediately preceding six-thread executable, LINREG fell
from 0.223 to 0.044 seconds and the complete infinitesimal phase fell from
2.881 to 2.709 seconds; a 283-iteration Bayesian fit was neutral at 35.975
versus 36.057 seconds. The complete Stage 1 model and verbose Stage 2 output,
including LINREG, INF, and BOLT-LMM columns, were byte-identical to the
pre-change six-thread control. The long fixed-parameter Bayesian run is a
parity stress test, not a representative headline workload.

Direct PGEN ingestion and persistent Step 0 construction now use independent
reader state per OpenMP worker. Variants are handled in bounded batches; QC
survivors are compacted in source order, and thread-local per-sample missingness
counts are reduced deterministically. This changes neither the packed cache
format nor its identity, and the same loader is used when Stage 1 reads PGEN
without a persistent cache.

On the real-LD 131,072-sample by 16,384-variant fixture, complete Step 0 time
fell from 2.191 to 1.637 seconds (25.3%). The 500,000-sample by 100,000-variant
target-stride template produced an 11.6 GiB cache. Across an order-reversed
pair, one thread took 39.607 and 43.539 seconds while six physical cores took
34.631 and 36.316 seconds: paired medians of 41.573 and 35.473 seconds, a 14.7%
end-to-end reduction. These totals include PVAR/PSAM processing, source
fingerprinting, cache metadata, and the final 11.6 GiB flush. Twelve logical
CPUs took 35.550 seconds and provided no improvement over six physical cores.
The one- and six-thread 11.6 GiB artifacts were byte-identical; the smaller
real-LD artifacts were also byte-identical. The target-stride fixture is one
tenth of the intended variant count and preserves exact storage dimensions,
but its template genotypes are not a replacement for a full-scale real-data
scientific run.

Default-refinement BOLT-REML on the same real-LD shape scaled from 54.333 to
18.770 seconds (2.89x) on six cores. Every thread count produced the same CG
convergence sequence and the same printed estimates: h2g 0.512 and sigma2
1.003198 (SE 0.016559). A bounded target-stride N=500,000 by M=1,024,
three-trial rung improved from 74.092 to 22.507 seconds (3.29x); this is a
low-rank throughput check, not a substitute for a full-rank real analysis.

BED and hardcall-PGEN Stage 2 now also use CPU threads outside BLAS. For models
through six combined basis/statistic vectors, bounded packed batches are scored
concurrently with thread-local workspaces and emitted in input order. Larger
models convert independent packed variants to centered dense columns in
parallel before threaded batched DGEMM. QC survivors are compacted in original
order. CUDA dispatch and scalar validation paths are unchanged.

The Stage 2 scaling ladder reused the real-LD 131,072-sample by 16,384-variant
fixture. The low-dimensional model had one basis and two statistics; the dense
model had 21 bases and two statistics.

| Model and input | 1-thread readout | 6-core readout | Readout speedup | 1-thread total | 6-core total | Total speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 basis + 2 stats, BED direct-packed | 4.831 s | 1.233 s | 3.92x | 5.175 s | 1.572 s | 3.29x |
| 1 basis + 2 stats, hardcall PGEN direct-packed | 4.466 s | 1.250 s | 3.57x | 4.836 s | 1.620 s | 2.98x |
| 21 bases + 2 stats, BED dense batch | 9.375 s | 2.478 s | 3.78x | 9.780 s | 2.887 s | 3.39x |
| 21 bases + 2 stats, hardcall PGEN dense batch | 7.648 s | 2.241 s | 3.41x | 8.077 s | 2.669 s | 3.03x |

The dense BED six-core readout was 2.75x faster than leaving conversion serial
and threading DGEMM alone (6.803 seconds). All BED/PGEN results were
byte-identical across thread counts and between the two formats. Relative to
the first split-stage commit's readout, the current six-core BED path is about
22.8x faster for the direct model and 19.2x faster for the dense model.

Across Stage 1, REML, BED/PGEN Stage 2, and BGEN Stage 2, all 12 logical CPUs
were slower than six physical cores on this six-core Xeon VM. Production
defaults should therefore start with the physical-core count and establish a
machine-specific scaling ladder before enabling SMT. Raw measurements and
compatibility checks are in `results/a100_cpu_threads.tsv`.

### CUDA host-thread scaling

The CUDA executable was also measured with one, two, four, and six physical
host cores and all 12 logical CPUs. Six physical cores cut REML wall time from
4.17 to 2.33 seconds. Stage 1 improved slightly, and Stage 2 was largely
insensitive to host-thread count. Twelve SMT threads were slower than six
physical cores for Stage 1 and REML. These results do not support a
CUDA-specific one-host-thread cap.

| CUDA workload | 1 host thread | 2 host threads | 4 host threads | 6 host threads | 12 logical threads |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 1 | 3.52 | 3.44 | 3.42 | 3.47 | 3.58 |
| Stage 2, 1 basis + 2 statistics | 1.22 | 1.21 | 1.19 | 1.18 | 1.26 |
| Stage 2, 21 bases + 2 statistics | 1.76 | 1.75 | 1.77 | 1.78 | 1.75 |
| REML, default refinement | 4.17 | 2.87 | 2.44 | 2.33 | 2.52 |

Every host-thread count produced the same byte-identical Stage 1 and Stage 2
association statistics. REML retained the same CG convergence sequence and
printed estimates. Raw measurements are in
[`a100_cuda_host_threads.tsv`](results/a100_cuda_host_threads.tsv).
