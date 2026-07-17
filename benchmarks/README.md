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

## End-to-end Stage 1 comparison

The first split-Stage-1 commit, `a58e7c3d2f3ca651ba16a39063f25b8e449087e0`,
provides the original baseline. It is essentially upstream BOLT-LMM v2.5 with
the Stage 1/2 boundary already present, so its timing includes the same model
fit and Stage 1 artifact write as the current code.

The original baseline and current CPU-only and CUDA binaries were built with
GCC 11.4, `-O3 -march=native`, LP64 sequential oneMKL, and one analysis thread.
The current CPU build disabled OpenMP; the original OpenMP build was restricted
to one thread. The CUDA build targeted compute capability 8.0 and used the same
native host settings. Three repetitions of each binary were interleaved and
pinned to the same vCPU on the Xeon/A100 VM. All used the 32,768-sample by
16,384-SNP BED fixture and the runtime arguments above, including default
LINREG.

| Stage 1 implementation | Median wall time | Speedup vs original | Time reduction |
| --- | ---: | ---: | ---: |
| Original split baseline (`a58e7c3`) | 129.06 s | 1.00x | — |
| Current CPU-only (`f337b92`) | 112.67 s | 1.15x | 12.7% |
| Current CUDA (`f337b92`) | 3.55 s | 36.35x | 97.2% |

CUDA was 31.74x faster than the optimized CPU-only build. Median phase times
show where the end-to-end changes came from. The nine raw measurements are in
`results/a100_stage1_headline.tsv`.

| Phase | Original | Current CPU | Current CUDA |
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
  --causal-variants 8192 --heritability 0.3
```

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

When mixture parameters are selected by cross-validation, the fitted effects
from the winning model are now reused to initialize the final LOCO
spike-and-slab fit. Effects are retained in raw per-allele units, averaged
across every computed fold, and converted to the full-cohort normalization.
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
`--noLinreg`. End-to-end Stage 1 took 35.26 seconds: 8.64 seconds for variance
fitting, 9.86 seconds for infinitesimal scoring, 11.72 seconds for mixture
estimation, and 4.05 seconds for final spike-and-slab scoring.

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
