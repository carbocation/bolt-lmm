# Running BOLT-LMM-XL

The upstream [BOLT-LMM v2.5 user
manual](https://alkesgroup.broadinstitute.org/BOLT-LMM/BOLT-LMM_manual.html) is
the authoritative reference for the scientific analysis options and output
fields. This document describes runtime behavior introduced by BOLT-LMM-XL:
split stages, PGEN input, optional PGEN caching, CUDA execution, and numerical
compatibility controls.

Print the common or complete command-line options with:

```sh
bolt --help
bolt --helpFull
```

## Split-stage association analysis

### Stage 1: fit the model

Stage 1 reads the model-marker genotypes, phenotype, and covariates; fits the
mixed model; and writes a reusable model artifact.

```sh
bolt \
  --stage=1 \
  --stage1Model=cohort.stage1.model \
  --bfile=data/model-markers \
  --phenoFile=data/phenotypes.tsv \
  --phenoCol=TRAIT \
  --covarFile=data/covariates.tsv \
  --qCovarCol=AGE \
  --qCovarCol=PC{1:10} \
  --lmm \
  --LDscoresFile=tables/LDSCORE.1000G_EUR.GRCh38.tab.gz
```

Stage 1 computes and stores LINREG statistics by default so split Stage 2 can
match upstream behavior: `CHISQ_LINREG` and `P_LINREG` are emitted only when
`--verboseStats` is passed. Analyses that do not need even optional LINREG
output can explicitly pass `--noLinreg` in Stage 1 to skip that computation.
The default Stage 2 output retains `P_BOLT_LMM_INF` and, when the spike-and-slab
model is fitted, `P_BOLT_LMM`.

In CPU OpenMP builds, `--numThreads` applies both to the main Stage 1 matrix
operations and to the marker-wise LINREG, infinitesimal, and spike-and-slab
association passes. The per-marker arithmetic and output order are unchanged.

For BOLT-REML, use `--stage=1 --reml`. No Stage 2 association readout is
required.

### Stage 2: test variants

Stage 2 reloads the fitted artifact and streams the variants to test.

```sh
bolt \
  --stage=2 \
  --stage1Model=cohort.stage1.model \
  --bfile=data/variants-to-test \
  --statsFile=cohort.stats.gz
```

The Stage 1 and Stage 2 genetic files do not need to be the same. Stage 1 needs
the model-marker set used for fitting; Stage 2 may independently read the
variant set to test from PLINK 1 BED, PLINK 2 PGEN, BGEN, IMPUTE2, or the dosage
formats supported by BOLT. Sample identities must still be compatible with the
fitted Stage 1 cohort.

The Stage 1 artifact makes it possible to run multiple independent Stage 2
jobs against different variant files without refitting the model.

BED and hardcall-PGEN inputs use bounded CPU batching or direct packed scoring
automatically. With `--numThreads` greater than one, low-dimensional models
score packed variants concurrently; larger models convert packed variants to
dense columns concurrently and then use threaded batched BLAS. In a CUDA build,
Stage 2 instead sends their two-bit packed data directly to the GPU and supports
sample subsetting and reordering without first expanding genotypes on the CPU.
Up to 32 combined covariate and statistic vectors use this path; larger models
fall back to CPU scoring. Single-threaded layout-2 BGEN readout retains the
optimized CPU path. With multiple threads, BGEN decompresses and decodes
variants in parallel and then uses the same batched scoring algorithm when the
number of covariate and statistic vectors is large enough to benefit. Using the
number of physical CPU cores is a good starting point; SMT may not improve these
memory-intensive pipelines.
`--stage2Scalar` selects the original per-variant arithmetic for these formats
as a validation/reference path; it does not change the statistical model or
output columns.

## Numerical compatibility

By default, Stage 1 retains the upstream v2.5 cold-start behavior for
variance-component conjugate-gradient solves and the final variational-Bayes
fit. This default applies to both CPU and CUDA builds.

Two output-changing performance experiments are available explicitly:

- `--warmStartVarianceCG` reuses solutions between nearby variance-component
  CG systems.
- `--warmStartFinalVB` initializes the final spike-and-slab fit from effects
  learned during cross-validation.

These options can change reported statistics at the existing convergence
tolerances. They are never enabled implicitly and should be used only after
that numerical-output tradeoff has been evaluated for the analysis.

## PLINK 2 PGEN hardcall input

Use `--pfile PREFIX` in Stage 1 or Stage 2 to read `PREFIX.pgen`,
`PREFIX.pvar[.gz]`, and `PREFIX.psam[.gz]`. BOLT-LMM-XL accepts biallelic PGEN
variants and reads their hardcalls. Dosage overrides in a PGEN file are
reported and ignored.

### Direct Stage 1 PGEN input

Stage 0 is not required. Without `--pgenCacheFile`, Stage 1 reads PGEN directly
and constructs its packed, BED-coded working matrix for that run. The matrix is
kept in RAM when it is no more than half of physical memory; otherwise BOLT
creates an unlinked, automatically deleted file under `TMPDIR` or `/tmp`.

Use `--pgenCacheDir DIR` to select local scratch storage explicitly. This also
forces file-backed working storage. It does not create a persistent or reusable
cache.

At 500,000 samples and one million Stage 1 variants, the packed working matrix
is about 116.4 GiB. The storage is two bits per genotype and exists for the
duration of the Stage 1 process even when no persistent cache is requested.

### Optional persistent Stage 0 cache

For repeated Stage 1 analyses using the same PGEN cohort and model-marker set,
Stage 0 can persist the converted packed representation:

```sh
bolt \
  --stage=0 \
  --pfile=data/cohort \
  --pgenCacheFile=/scratch/cohort.bolt-pgen-cache \
  --numThreads=6 \
  --remove=data/remove.txt \
  --modelSnps=data/model.snps

bolt \
  --stage=1 \
  --stage1Model=cohort.stage1.model \
  --pfile=data/cohort \
  --pgenCacheFile=/scratch/cohort.bolt-pgen-cache \
  --remove=data/remove.txt \
  --modelSnps=data/model.snps \
  --phenoFile=data/phenotypes.tsv \
  --phenoCol=TRAIT \
  --lmm
```

Stage 0 is CPU-only even when the binary contains CUDA support. OpenMP builds
use `--numThreads` to read, convert, and QC bounded PGEN variant batches with
independent reader state; direct PGEN Stage 1 ingestion uses the same path. It
writes the post-filter/QC packed hardcalls, variant mapping and MAFs, and sample
QC mask atomically. Stage 1 parses the named PGEN/PVAR/PSAM files and verifies
their identity together with the sample/variant selection, packing, and QC
settings before memory-mapping the cache. Phenotypes and covariates are
deliberately not part of the cache identity, allowing reuse across traits.

The persistent artifact is useful only when it remains on storage local to or
already shared with the Stage 1 environment. At the target shape it is about
116.4 GiB. The exact full-scale one-thread measurement required 8 minutes 46
seconds, so staging the cache afresh had to sustain more than about 226 MiB/s
one-way—or 453 MiB/s for two sequential copies—just to break even. Six-core
ingestion is 14.7% faster on the one-tenth target-stride rung; if that reduction
holds at one million variants, the corresponding thresholds rise to about 266
and 532 MiB/s. The full-scale threaded value has not been extrapolated into a
claimed measurement. If the cache must be built or transferred for each
analysis, use direct PGEN ingestion instead.

## CUDA Stage 1, packed Stage 2, and BOLT-REML execution

CUDA-enabled binaries accelerate the projected genotype matrix products used
by variance estimation and the infinitesimal model, the block operations in
variational Bayes, cross-validation prediction, LINREG, retrospective scoring,
and SNP normalization and projection. Packed two-bit genotypes cross PCIe and
are decoded on the device; dense genotype matrices do not cross PCIe.

With `--reml`, CUDA also accelerates the AI-REML conjugate-gradient and
derivative products and the multiplication used to construct genetic Monte
Carlo pseudo-phenotypes. This applies to univariate, multivariate, and multiple
variance-component REML. The estimator, CPU RNG and draw order, convergence
tolerances, trial counts, and refinement behavior are unchanged.

A CUDA-enabled binary uses the GPU automatically for Stages 1 and 2. In Stage 2,
this applies to BED and hardcall-PGEN association input; BGEN, IMPUTE2, and text
dosage formats stay on CPU. Pass `--no-cuda` to exercise the CPU path for
comparison or testing. The legacy `--cuda` flag is still accepted. A CPU-only
build has no CUDA runtime dependency.

### Device genotype cache

The CUDA backend uses otherwise-free device memory to cache a shared prefix of
the packed genotype matrix. It reserves one decoded SNP block plus 3 GiB for a
simultaneous cross-validation fold and working buffers. Main and fold-specific
models share the packed cache while retaining their own sample masks and
covariate projections.

Use `--cudaCacheGiB N` to cap the packed device cache. A value of `0` disables
it and exercises the streamed host-to-device path; the default `-1` selects the
size automatically.

### Retained host cache and streaming

When Stage 1 PGEN hardcalls are file-backed and do not all fit in the device
cache, CUDA can retain a bounded part of the remaining packed matrix in
ordinary host RAM. The retained range begins after the device cache and is
populated during marker initialization, avoiding another scratch-disk read.

Use `--cudaHostCacheGiB N` to set its limit. A value of `0` minimizes memory
use; the default `-1` uses at most 72% of physical RAM. This cache is not
page-locked and is temporary to the Stage 1 process.

Blocks outside the device and retained-host caches are double-buffered through
two pinned host buffers and two device buffers. A transfer stream preloads the
next block while the compute stream processes the current block. At 500,000
samples with the default 512-SNP CUDA block, the pinned host pair is bounded at
about 122 MiB. Large pageable-to-pinned staging copies use the CPU team selected
by `--numThreads`; `--numThreads=1` retains serial staging. This does not change
the genotype bytes or floating-point operations performed on the GPU.

See [`BUILD.md`](BUILD.md) for CUDA compilation instructions and
[`benchmarks/README.md`](benchmarks/README.md) for measured performance and
scientific-output comparisons.
