#!/usr/bin/env python3
"""Generate a deterministic PLINK 1 workload for Step 1 benchmarks."""

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=Path, help="output PLINK prefix")
    parser.add_argument("--samples", type=int, default=32768)
    parser.add_argument("--variants", type=int, default=16384)
    parser.add_argument("--causal-variants", type=int, default=1024)
    parser.add_argument("--heritability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--batch-variants", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.samples < 8 or args.variants < 22:
        raise SystemExit("--samples must be >= 8 and --variants must be >= 22")
    if args.causal_variants < 1:
        raise SystemExit("--causal-variants must be >= 1")
    if args.batch_variants < 1:
        raise SystemExit("--batch-variants must be >= 1")
    if not 0 < args.heritability < 1:
        raise SystemExit("--heritability must be between 0 and 1")

    prefix = args.prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    samples_padded = (args.samples + 3) & ~3
    bytes_per_variant = samples_padded // 4
    genetic_score = np.zeros(args.samples, dtype=np.float64)
    causal = min(args.causal_variants, args.variants)
    effects = rng.normal(size=causal)
    effects /= np.linalg.norm(effects)

    with prefix.with_suffix(".bed").open("wb") as bed:
        bed.write(b"\x6c\x1b\x01")
        for first in range(0, args.variants, args.batch_variants):
            count = min(args.batch_variants, args.variants - first)
            maf = rng.uniform(0.1, 0.5, size=(count, 1))
            geno = rng.binomial(2, maf, size=(count, args.samples)).astype(np.uint8)

            causal_count = max(0, min(first + count, causal) - first)
            if causal_count:
                centered = geno[:causal_count].astype(np.float64) - 2 * maf[:causal_count]
                genetic_score += effects[first:first + causal_count] @ centered

            if samples_padded != args.samples:
                geno = np.pad(geno, ((0, 0), (0, samples_padded - args.samples)))

            # PLINK BED codes: 00 -> 2, 10 -> 1, 11 -> 0 effect alleles.
            code = np.choose(geno, (3, 2, 0)).astype(np.uint8)
            packed = (code[:, 0::4] | (code[:, 1::4] << 2) |
                      (code[:, 2::4] << 4) | (code[:, 3::4] << 6))
            assert packed.shape == (count, bytes_per_variant)
            bed.write(packed.tobytes())

    with prefix.with_suffix(".bim").open("w") as bim:
        for variant in range(args.variants):
            chrom = 1 + (22 * variant) // args.variants
            chrom_first = ((chrom - 1) * args.variants + 21) // 22
            position = 1000 * (variant - chrom_first + 1)
            bim.write(f"{chrom}\trs{variant + 1}\t0\t{position}\tA\tG\n")

    with prefix.with_suffix(".fam").open("w") as fam:
        for sample in range(args.samples):
            fam.write(f"F{sample + 1}\tI{sample + 1}\t0\t0\t0\t-9\n")

    genetic_score -= genetic_score.mean()
    genetic_score /= genetic_score.std()
    noise = rng.normal(size=args.samples)
    noise -= noise.mean()
    noise /= noise.std()
    phenotype = (np.sqrt(args.heritability) * genetic_score +
                 np.sqrt(1 - args.heritability) * noise)
    with prefix.with_suffix(".pheno").open("w") as pheno:
        pheno.write("FID\tIID\tPHENO\n")
        for sample, value in enumerate(phenotype):
            pheno.write(f"F{sample + 1}\tI{sample + 1}\t{value:.12g}\n")

    gib = (3 + args.variants * bytes_per_variant) / (1024 ** 3)
    print(f"Wrote {prefix} ({args.samples} samples, {args.variants} variants, "
          f"{gib:.3f} GiB BED)")


if __name__ == "__main__":
    main()
