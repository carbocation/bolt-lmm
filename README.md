# BOLT-LMM-XL

BOLT-LMM is software for large-scale genetic analysis. It provides two main
methods:

- **BOLT-LMM** for linear mixed-model association testing.
- **BOLT-REML** for variance-components analysis, including SNP-heritability
  partitioning and genetic-correlation estimation.

BOLT-LMM is intended for human datasets with more than 5,000 samples. Its
association statistics are appropriate for quantitative traits and reasonably
balanced case-control traits; consult the
[BOLT-LMM v2.5 user manual](https://alkesgroup.broadinstitute.org/BOLT-LMM/BOLT-LMM_manual.html)
before analyzing smaller datasets or highly unbalanced binary traits.

## About this fork

BOLT-LMM-XL is a fork of BOLT-LMM that aims to accelerate workflows built on
BOLT-LMM. 

### Split-stage analyses
It separates association analysis into two phases: Stage 1 fits the phenotype,
covariates, and mixed model once and saves a reusable model artifact; Stage 2
reloads that artifact and streams the variants to test.

This design avoids repeating model fitting when the same model is used to test
multiple variant sets, and it allows association readout to be divided among
independent Stage 2 jobs. 

### PGEN (plink2 file format) partial support
BOLT-LMM-XL also adds PLINK 2 PGEN hardcall input. 

### Easier builds
Implements a portable CMake build for Linux and macOS.

## Documentation

- The [user
  manual](https://alkesgroup.broadinstitute.org/BOLT-LMM/BOLT-LMM_manual.html)
  is the authoritative reference for analysis options, output, computing
  requirements, recommendations, and troubleshooting.
- [`BUILD.md`](BUILD.md) covers dependencies, CMake configuration, supported
  BLAS/LAPACK backends, OpenMP, portable CPU targets, and platform-specific
  instructions for Linux and macOS.
- [`example/run_example.sh`](example/run_example.sh) demonstrates BOLT-LMM, and
  [`example/run_example_reml2.sh`](example/run_example_reml2.sh) demonstrates
  multi-trait BOLT-REML. The bundled dataset is only large enough to demonstrate
  command syntax, not to produce a robust analysis.

> **Compatibility note:** commands in the published manual describe the
> standard BOLT-LMM single-run release interface. BOLT-LMM-XL requires `--stage`
> and, for association analysis, `--stage1Model` as shown below.

## Build

You need a C++14 compiler, CMake 3.18 or newer, Boost.Program_options,
Boost.Iostreams, zlib, zstd, NLopt (including `nlopt.hpp`), and a supported
BLAS/LAPACK implementation. OpenMP is optional; builds fall back to serial
execution when it is unavailable.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The executable is written to `build/bolt`. `make` is a convenience wrapper for
the same CMake build. See [`BUILD.md`](BUILD.md) for package-install examples and
configuration options for Apple Accelerate, OpenBLAS, Intel oneMKL, OpenMP, and
CPU targeting.

## Reference data

The upstream `tables/` directory is about 200 MiB and is intentionally not
tracked in this repository. Download it from the
[official BOLT-LMM downloads page](https://alkesgroup.broadinstitute.org/BOLT-LMM/downloads/).
For BOLT-LMM v2.5, the reference files are included in the
[BOLT-LMM v2.5 release archive](https://storage.googleapis.com/broad-alkesgroup-public/BOLT-LMM/downloads/BOLT-LMM_v2.5.tar.gz):

```sh
curl -LO https://storage.googleapis.com/broad-alkesgroup-public/BOLT-LMM/downloads/BOLT-LMM_v2.5.tar.gz
tar -xzf BOLT-LMM_v2.5.tar.gz
```

After extracting the archive, copy or symlink its `tables/` directory into the
root of this checkout. The quick start below expects
`tables/LDSCORE.1000G_EUR.GRCh38.tab.gz`, which is also used by the bundled
BOLT-LMM example. The release includes hg19 and hg38 genetic maps for analyses
that use `--geneticMapFile`.

## Quick start

Print the common or complete command-line options with:

```sh
./build/bolt --help
./build/bolt --helpFull
```

An association analysis first fits the phenotype and covariates and saves a
model artifact:

```sh
./build/bolt \
  --stage=1 \
  --stage1Model=cohort.stage1.model \
  --bfile=data/cohort \
  --phenoFile=data/phenotypes.tsv \
  --phenoCol=TRAIT \
  --covarFile=data/covariates.tsv \
  --qCovarCol=AGE \
  --qCovarCol=PC{1:10} \
  --lmm \
  --LDscoresFile=tables/LDSCORE.1000G_EUR.GRCh38.tab.gz \
  --numThreads=8
```

Stage 1 computes and stores LINREG statistics by default for compatibility
with existing model artifacts and Stage 2 `--verboseStats` output. Analyses
that only need mixed-model results can pass `--noLinreg` in Stage 1 to avoid
that extra genotype traversal. Stage 2 will then omit `P_LINREG` while retaining
`P_BOLT_LMM_INF` and, when the spike-and-slab model is fitted, `P_BOLT_LMM`.

Stage 2 reloads that artifact and streams the variants to test:

```sh
./build/bolt \
  --stage=2 \
  --stage1Model=cohort.stage1.model \
  --bfile=data/cohort \
  --statsFile=cohort.stats.gz \
  --numThreads=8
```

Replace `--bfile PREFIX` with `--pfile PREFIX` to use PLINK 2
`PREFIX.pgen`, `PREFIX.pvar[.gz]`, and `PREFIX.psam[.gz]` files in either stage.
This repository accepts biallelic PGEN variants and reads their hardcalls;
dosage overrides in a PGEN file are reported and ignored. The manual documents
additional Stage 2 inputs for BGEN, IMPUTE2, and dosage formats.

For BOLT-REML, use `--stage=1 --reml`; no Stage 2 association readout is
required. See the bundled REML example for multi-trait syntax.

## License

BOLT-LMM-XL is distributed under the GNU General Public License v3.0. See
[`license.txt`](license.txt) for the full terms and third-party license notices.
