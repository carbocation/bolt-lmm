../bolt \
    --bfile=EUR_subset \
    --remove=EUR_subset.remove \
    --exclude=EUR_subset.exclude \
    --exclude=EUR_subset.exclude2 \
    --phenoFile=EUR_subset.pheno2.covars \
    --phenoCol=PHENO \
    --covarFile=EUR_subset.pheno2.covars \
    --covarCol=CAT_COV \
    --qCovarCol=QCOV{1:2} \
    --modelSnps=EUR_subset.modelSnps \
    --lmm \
    --LDscoresFile=../tables/LDSCORE.1000G_EUR.GRCh38.tab.gz \
    --numThreads=2 \
    --statsFile=example.stats \
    --dosageFile=EUR_subset.dosage.chr17first100 \
    --dosageFile=EUR_subset.dosage.chr22last100.gz \
    --dosageFidIidFile=EUR_subset.dosage.indivs \
    --statsFileDosageSnps=example.dosageSnps.stats \
    --impute2FileList=EUR_subset.impute2FileList.txt \
    --impute2FidIidFile=EUR_subset.impute2.indivs \
    --statsFileImpute2Snps=example.impute2Snps.stats \
    --dosage2FileList=EUR_subset.dosage2FileList.txt \
    --statsFileDosage2Snps=example.dosage2Snps.stats \
    2>&1 | tee example.log # log output written to stdout and stderr

# basic args:
# --bfile: prefix of PLINK genotype files (bed/bim/fam)
# --remove: list of individuals to remove (FID IID)
# --exclude: list of SNPs to exclude (rs###)
# --phenoFile: phenotype file
# --phenoCol: column of phenotype file containing phenotypes
# --covarFile: covariate file
# --covarCol: column(s) containing categorical covariate (multiple ok)
# --qCovarCol: column(s) containing quantitative covariates (array format)
# --modelSnps: subset of SNPs to use in GRMs
# --lmm: flag to perform default BOLT-LMM mixed model association
# --LDscoresFile: reference LD Scores (data table in separate download)
# --numThreads: multi-threaded execution

# additional args for association testing on imputed SNPs:
# --statsFile: output file for association statistics at PLINK-format SNPs
# --dosageFile: file(s) containing additional dosage-format SNPs (multiple ok)
# --dosageFidIidFile: file containing FIDs and IIDs for dosage-format SNPs
# --statsFileDosageSnps: output file for assoc stats at dosage-format SNPs
# --impute2FileList: file listing chroms and IMPUTE2-format additional SNPs
# --impute2FidIidFile: file containing FIDs and IIDs for IMPUTE2-format SNPs
# --impute2CallThresh: minimum pAA+pAB+pBB for calling IMPUTE2-format SNPs
# --statsFileImpute2Snps: output file for assoc stats at IMPUTE2-format SNPs
# --dosage2FileList: file listing map and 2-dosage format additional SNPs
# --statsFileDosage2Snps: output file for assoc stats at 2-dosage format SNPs
