../bolt \
    --bfile=EUR_subset \
    --phenoFile=EUR_subset.pheno2.covars \
    --exclude=EUR_subset.exclude2 \
    --phenoCol=PHENO \
    --phenoCol=QCOV1 \
    --modelSnps=EUR_subset.modelSnps2 \
    --reml \
    --numThreads=2 \
    2>&1 | tee example_reml2.log # log output written to stdout and stderr

### NOTE: This example just demonstrates parameter usage.
###       The algorithm is not robust with so few samples; we recommend N>5000.

# --bfile: prefix of PLINK genotype files (bed/bim/fam)
# --phenoFile: phenotype file
# --phenoCol: column(s) of phenotype file containing phenotypes
# --modelSnps: subset of SNPs to use in GRMs
# --reml: flag to perform default BOLT-REML variance components analysis
# --numThreads: multi-threaded execution
