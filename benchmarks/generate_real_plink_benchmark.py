#!/usr/bin/env python3
"""Build a deterministic PLINK benchmark from a real PLINK 1 reference panel.

The reference genotypes are never copied into the repository.  For an expanded
cohort, each output sample is assigned a reference-panel group and independently
draws a donor from that group at each LD-block boundary.  Genotypes within a
block therefore retain the reference panel's allele frequencies, missingness,
local LD, and population structure without pretending that the expanded
individuals are real people.
"""

import argparse
from pathlib import Path

import numpy as np


BED_MAGIC = b"\x6c\x1b\x01"
BED_CODE_TO_DOSAGE = np.array([2.0, np.nan, 1.0, 0.0])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_prefix", type=Path, help="input PLINK 1 prefix")
    parser.add_argument("output_prefix", type=Path, help="output PLINK 1 prefix")
    parser.add_argument("--source-psam", type=Path,
                        help="optional PSAM containing the population-group column")
    parser.add_argument("--group-column", default="SuperPop")
    parser.add_argument("--samples", type=int,
                        help="output samples (default: preserve the reference samples)")
    parser.add_argument("--variants", type=int, default=16384)
    parser.add_argument("--identity-samples", action="store_true",
                        help="preserve reference sample order instead of block bootstrapping")
    parser.add_argument("--ld-block-variants", type=int, default=32,
                        help="number of consecutive selected variants per donor draw")
    parser.add_argument("--batch-variants", type=int, default=32)
    parser.add_argument("--causal-variants", type=int, default=1024)
    parser.add_argument("--heritability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def read_fam(path):
    samples = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if len(fields) < 2:
                raise SystemExit(f"Invalid FAM line {line_number}: {path}")
            samples.append((fields[0], fields[1], fields))
    if not samples:
        raise SystemExit(f"No samples found in {path}")
    return samples


def read_groups(path, group_column, samples):
    if path is None:
        return np.array(["ALL"] * len(samples), dtype=object)

    with path.open() as stream:
        header = None
        iid_column = group_index = None
        groups_by_iid = {}
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if header is None:
                if not fields[0].startswith("#"):
                    raise SystemExit(f"PSAM header not found in {path}")
                fields[0] = fields[0].lstrip("#")
                header = fields
                try:
                    iid_column = header.index("IID")
                    group_index = header.index(group_column)
                except ValueError as error:
                    raise SystemExit(f"Missing PSAM column in {path}: {error}") from error
                continue
            if len(fields) < len(header):
                raise SystemExit(f"Invalid PSAM line {line_number}: {path}")
            groups_by_iid[fields[iid_column]] = fields[group_index]

    groups = []
    for _, iid, _ in samples:
        if iid not in groups_by_iid:
            raise SystemExit(f"Sample {iid} from FAM is absent from {path}")
        groups.append(groups_by_iid[iid])
    return np.asarray(groups, dtype=object)


def read_bim_and_select(path, variants):
    lines = []
    chromosomes = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if len(fields) < 6:
                raise SystemExit(f"Invalid BIM line {line_number}: {path}")
            try:
                chromosome = int(fields[0].removeprefix("chr"))
            except ValueError:
                chromosome = -1
            lines.append(line)
            chromosomes.append(chromosome)

    eligible_by_chromosome = [
        np.flatnonzero(np.asarray(chromosomes) == chromosome)
        for chromosome in range(1, 23)
    ]
    eligible_count = sum(indices.size for indices in eligible_by_chromosome)
    if variants < 22 or variants > eligible_count:
        raise SystemExit(
            f"--variants must be between 22 and {eligible_count} autosomal variants"
        )

    if variants == eligible_count:
        selected = np.concatenate(eligible_by_chromosome)
        return lines, selected

    sizes = np.asarray([indices.size for indices in eligible_by_chromosome])
    desired = variants * sizes / sizes.sum()
    counts = np.floor(desired).astype(int)
    counts = np.maximum(counts, 1)
    while counts.sum() < variants:
        choices = np.flatnonzero(counts < sizes)
        choice = choices[np.argmax(desired[choices] - counts[choices])]
        counts[choice] += 1
    while counts.sum() > variants:
        choices = np.flatnonzero(counts > 1)
        choice = choices[np.argmax(counts[choices] - desired[choices])]
        counts[choice] -= 1

    selected_chunks = []
    for indices, count in zip(eligible_by_chromosome, counts):
        first = (indices.size - count) // 2
        selected_chunks.append(indices[first:first + count])
    return lines, np.concatenate(selected_chunks)


def pack_bed_codes(codes, samples_padded):
    if samples_padded != codes.shape[1]:
        codes = np.pad(codes, ((0, 0), (0, samples_padded - codes.shape[1])))
    return (codes[:, 0::4] | (codes[:, 1::4] << 2) |
            (codes[:, 2::4] << 4) | (codes[:, 3::4] << 6))


def standardized_score_genotype(codes):
    dosage = BED_CODE_TO_DOSAGE[codes].astype(np.float64, copy=False)
    missing = np.isnan(dosage)
    if missing.any():
        observed = dosage[~missing]
        if observed.size == 0:
            return np.zeros(dosage.size)
        dosage = dosage.copy()
        dosage[missing] = observed.mean()
    dosage -= dosage.mean()
    std = dosage.std()
    if std == 0:
        return np.zeros(dosage.size)
    return dosage / std


def main():
    args = parse_args()
    if args.output_prefix.resolve() == args.source_prefix.resolve():
        raise SystemExit("The output prefix must differ from the source prefix")
    if args.variants < 22:
        raise SystemExit("--variants must be at least 22")
    if args.ld_block_variants < 1 or args.batch_variants < 1:
        raise SystemExit("block and batch sizes must be positive")
    if args.causal_variants < 1 or args.causal_variants > args.variants:
        raise SystemExit("--causal-variants must be between 1 and --variants")
    if not 0 < args.heritability < 1:
        raise SystemExit("--heritability must be between 0 and 1")

    source_fam = args.source_prefix.with_suffix(".fam")
    source_bim = args.source_prefix.with_suffix(".bim")
    source_bed = args.source_prefix.with_suffix(".bed")
    samples = read_fam(source_fam)
    source_groups = read_groups(args.source_psam, args.group_column, samples)
    bim_lines, selected_variants = read_bim_and_select(source_bim, args.variants)

    source_n = len(samples)
    output_n = args.samples if args.samples is not None else source_n
    if output_n < 8:
        raise SystemExit("--samples must be at least 8")
    if args.identity_samples and output_n != source_n:
        raise SystemExit("--identity-samples requires the reference sample count")

    source_bytes_per_variant = (source_n + 3) // 4
    source_m = len(bim_lines)
    expected_bed_bytes = 3 + source_m * source_bytes_per_variant
    if source_bed.stat().st_size != expected_bed_bytes:
        raise SystemExit(
            f"Unexpected BED size for {source_n} samples and {source_m} variants: {source_bed}"
        )
    with source_bed.open("rb") as stream:
        if stream.read(3) != BED_MAGIC:
            raise SystemExit(f"Unsupported BED header: {source_bed}")
    source_matrix = np.memmap(
        source_bed, dtype=np.uint8, mode="r", offset=3,
        shape=(source_m, source_bytes_per_variant)
    )

    seed_sequence = np.random.SeedSequence(args.seed)
    group_rng, donor_rng, effect_rng, noise_rng = [
        np.random.default_rng(seed) for seed in seed_sequence.spawn(4)
    ]
    group_names, source_group_codes = np.unique(source_groups, return_inverse=True)
    source_members = [np.flatnonzero(source_group_codes == group) for group in range(len(group_names))]

    if args.identity_samples:
        output_group_codes = source_group_codes
        output_ids = [(fid, iid) for fid, iid, _ in samples]
    else:
        group_probabilities = np.bincount(source_group_codes) / source_n
        output_group_codes = group_rng.choice(
            len(group_names), size=output_n, p=group_probabilities
        )
        output_ids = [(f"F{sample + 1}", f"I{sample + 1}") for sample in range(output_n)]
    output_group_positions = [
        np.flatnonzero(output_group_codes == group) for group in range(len(group_names))
    ]

    causal_indices = np.sort(effect_rng.choice(
        args.variants, size=args.causal_variants, replace=False
    ))
    causal_effects = effect_rng.normal(size=args.causal_variants)
    causal_effects /= np.linalg.norm(causal_effects)
    causal_lookup = dict(zip(causal_indices.tolist(), causal_effects.tolist()))
    genetic_score = np.zeros(output_n, dtype=np.float64)

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    samples_padded = (output_n + 3) & ~3
    with output_prefix.with_suffix(".bed").open("wb") as bed:
        bed.write(BED_MAGIC)
        for block_first in range(0, args.variants, args.ld_block_variants):
            block_count = min(args.ld_block_variants, args.variants - block_first)
            if args.identity_samples:
                donors = np.arange(source_n, dtype=np.int64)
            else:
                donors = np.empty(output_n, dtype=np.int64)
                for members, positions in zip(source_members, output_group_positions):
                    donors[positions] = donor_rng.choice(members, size=positions.size)
            donor_bytes = donors >> 2
            donor_shifts = ((donors & 3) << 1)[None, :]

            for batch_first in range(block_first, block_first + block_count,
                                     args.batch_variants):
                batch_count = min(
                    args.batch_variants, block_first + block_count - batch_first
                )
                source_indices = selected_variants[batch_first:batch_first + batch_count]
                source_packed = np.asarray(source_matrix[source_indices])
                codes = ((source_packed[:, donor_bytes] >> donor_shifts) & 3).astype(np.uint8)
                pack_bed_codes(codes, samples_padded).tofile(bed)

                for offset in range(batch_count):
                    output_variant = batch_first + offset
                    effect = causal_lookup.get(output_variant)
                    if effect is not None:
                        genetic_score += effect * standardized_score_genotype(codes[offset])

    with output_prefix.with_suffix(".bim").open("w") as bim:
        for source_index in selected_variants:
            bim.write(bim_lines[source_index])

    with output_prefix.with_suffix(".fam").open("w") as fam:
        if args.identity_samples:
            for (_, _, fields) in samples:
                fields = fields[:]
                while len(fields) < 6:
                    fields.append("0")
                fields[5] = "-9"
                fam.write("\t".join(fields[:6]) + "\n")
        else:
            for fid, iid in output_ids:
                fam.write(f"{fid}\t{iid}\t0\t0\t0\t-9\n")

    genetic_score -= genetic_score.mean()
    score_std = genetic_score.std()
    if score_std == 0:
        raise SystemExit("Generated genetic score has zero variance")
    genetic_score /= score_std
    noise = noise_rng.normal(size=output_n)
    noise -= noise.mean()
    noise /= noise.std()
    phenotype = (np.sqrt(args.heritability) * genetic_score +
                 np.sqrt(1 - args.heritability) * noise)

    with output_prefix.with_suffix(".pheno").open("w") as pheno:
        pheno.write("FID\tIID\tPHENO\n")
        for (fid, iid), value in zip(output_ids, phenotype):
            pheno.write(f"{fid}\t{iid}\t{value:.12g}\n")

    with output_prefix.with_suffix(".covar").open("w") as covar:
        covar.write("FID\tIID\tSUPERPOP\n")
        for (fid, iid), group in zip(output_ids, group_names[output_group_codes]):
            covar.write(f"{fid}\t{iid}\t{group}\n")

    gib = (3 + args.variants * (samples_padded // 4)) / (1024 ** 3)
    if args.identity_samples:
        mode = "identity"
    else:
        grouping = f"within-{args.group_column}" if args.source_psam else "single-group"
        mode = f"{grouping} {args.ld_block_variants}-variant block bootstrap"
    print(
        f"Wrote {output_prefix} ({output_n} samples, {args.variants} real variants, "
        f"{mode}, {gib:.3f} GiB BED)"
    )


if __name__ == "__main__":
    main()
