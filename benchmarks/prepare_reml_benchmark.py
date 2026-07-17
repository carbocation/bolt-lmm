#!/usr/bin/env python3
"""Prepare deterministic multi-trait and multi-VC inputs for BOLT-REML.

This script does not generate or copy genotypes.  It layers correlated traits
and chromosome-partitioned variance-component labels onto an existing PLINK 1
benchmark, so the genotype fixture's allele frequencies and LD are preserved.
The additional traits are intended for performance and parity testing, not as
a realistic generative model for genetic correlation.
"""

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("genotype_prefix", type=Path, help="input PLINK 1 prefix")
    parser.add_argument("output_prefix", type=Path, help="output phenotype/model-SNP prefix")
    parser.add_argument("--source-pheno", type=Path,
                        help="input phenotype table (default: GENOTYPE_PREFIX.pheno)")
    parser.add_argument("--traits", type=int, default=2)
    parser.add_argument("--trait-correlation", type=float, default=0.4)
    parser.add_argument("--variance-components", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def read_source_phenotype(path):
    with path.open() as stream:
        header = stream.readline().split()
        if len(header) < 3 or header[:2] != ["FID", "IID"]:
            raise SystemExit(f"Expected FID IID and a phenotype column in {path}")
        samples = []
        values = []
        for line_number, line in enumerate(stream, 2):
            fields = line.split()
            if len(fields) != len(header):
                raise SystemExit(f"Invalid phenotype line {line_number}: {path}")
            samples.append((fields[0], fields[1]))
            try:
                values.append(float(fields[2]))
            except ValueError as error:
                raise SystemExit(f"Invalid phenotype value on line {line_number}: {path}") from error
    if not samples:
        raise SystemExit(f"No phenotype samples found in {path}")
    return samples, np.asarray(values, dtype=np.float64)


def require_matching_fam(path, samples):
    fam_samples = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if len(fields) < 2:
                raise SystemExit(f"Invalid FAM line {line_number}: {path}")
            fam_samples.append((fields[0], fields[1]))
    if fam_samples != samples:
        raise SystemExit(f"Sample IDs/order differ between {path} and the phenotype table")


def standardized(values, label):
    values = values - values.mean()
    std = values.std()
    if not np.isfinite(std) or std == 0:
        raise SystemExit(f"{label} has zero or invalid variance")
    return values / std


def write_phenotypes(path, samples, source, traits, correlation, seed):
    rng = np.random.default_rng(seed)
    columns = [standardized(source, "source phenotype")]
    residual_scale = np.sqrt(1 - correlation * correlation)
    for trait in range(1, traits):
        noise = standardized(rng.normal(size=source.size), f"trait {trait + 1} noise")
        columns.append(standardized(correlation * columns[0] + residual_scale * noise,
                                    f"trait {trait + 1}"))
    with path.open("w") as stream:
        stream.write("FID\tIID\t" + "\t".join(
            f"PHENO{trait + 1}" for trait in range(traits)) + "\n")
        for sample_index, (fid, iid) in enumerate(samples):
            values = "\t".join(f"{column[sample_index]:.12g}" for column in columns)
            stream.write(f"{fid}\t{iid}\t{values}\n")


def write_model_snps(bim_path, output_path, variance_components):
    counts = np.zeros(variance_components, dtype=np.int64)
    with bim_path.open() as bim, output_path.open("w") as output:
        for line_number, line in enumerate(bim, 1):
            fields = line.split()
            if len(fields) < 2:
                raise SystemExit(f"Invalid BIM line {line_number}: {bim_path}")
            try:
                chromosome = int(fields[0].removeprefix("chr"))
            except ValueError:
                chromosome = line_number
            component = (chromosome - 1) % variance_components
            output.write(f"{fields[1]}\tvc{component + 1}\n")
            counts[component] += 1
    if np.any(counts == 0):
        raise SystemExit("At least one requested variance component has no SNPs")
    return counts


def main():
    args = parse_args()
    if args.traits < 1:
        raise SystemExit("--traits must be positive")
    if not -1 < args.trait_correlation < 1:
        raise SystemExit("--trait-correlation must be between -1 and 1")
    if not 1 <= args.variance_components <= 255:
        raise SystemExit("--variance-components must be between 1 and 255")

    source_pheno = args.source_pheno or args.genotype_prefix.with_suffix(".pheno")
    samples, source = read_source_phenotype(source_pheno)
    require_matching_fam(args.genotype_prefix.with_suffix(".fam"), samples)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pheno_path = args.output_prefix.with_suffix(".pheno")
    model_path = args.output_prefix.with_suffix(".modelSnps")
    write_phenotypes(pheno_path, samples, source, args.traits,
                     args.trait_correlation, args.seed)
    counts = write_model_snps(args.genotype_prefix.with_suffix(".bim"), model_path,
                              args.variance_components)
    print(f"Wrote {pheno_path} ({len(samples)} samples, {args.traits} traits)")
    print(f"Wrote {model_path} ({', '.join(str(count) for count in counts)} SNPs per VC)")


if __name__ == "__main__":
    main()
