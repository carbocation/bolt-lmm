# PGEN parity fixture

The `example` PLINK 1 and PLINK 2 file sets contain the same synthetic 500-sample,
1,000-variant hardcalls. They are copied from the regenie example data at commit
`d4c47804b097cd70760bd311866bd9c05c577813` and are used to verify byte-for-byte
parity after BOLT sample subsetting and hardcall conversion. The layout-2 BGEN
fixture was exported from the same PGEN with PLINK 2.0 alpha 7.1 using 8-bit
probabilities. The Stage 2 test also reorders and masks model samples before
comparing complete association output from the optimized and scalar paths.
