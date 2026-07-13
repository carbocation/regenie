#!/usr/bin/env python3
"""Generate a deterministic PLINK BED dataset for Step 1 benchmarking."""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path


BED_MAGIC = b"\x6c\x1b\x01"

# GRCh38 primary-assembly autosome lengths. Physical length is not a perfect
# model of array-marker density, but it captures the first-order chromosome
# size distribution without tying the benchmark to a proprietary array.
HUMAN_AUTOSOME_LENGTHS = (
    248956422,
    242193529,
    198295559,
    190214555,
    181538259,
    170805979,
    159345973,
    145138636,
    138394717,
    133797422,
    135086622,
    133275309,
    114364328,
    107043718,
    101991189,
    90338345,
    83257441,
    80373285,
    58617616,
    64444167,
    46709983,
    50818468,
)


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--samples", type=positive_int, default=20000)
    parser.add_argument("--variants", type=positive_int, default=20000)
    parser.add_argument("--phenotypes", type=positive_int, default=4)
    parser.add_argument("--chromosomes", type=positive_int, default=22)
    parser.add_argument(
        "--chromosome-layout",
        choices=("equal", "human"),
        default="equal",
        help=(
            "allocate variants equally or approximately in proportion to "
            "human autosome lengths"
        ),
    )
    parser.add_argument("--seed", type=positive_int, default=20260712)
    parser.add_argument("--variants-per-chunk", type=positive_int, default=256)
    parser.add_argument("--max-bed-gb", type=positive_float, default=4.0)
    parser.add_argument(
        "--trait-type",
        choices=("qt", "bt"),
        default="qt",
        help="generate quantitative (qt) or binary (bt) phenotypes",
    )
    parser.add_argument(
        "--missingness-profile",
        choices=("none", "incident"),
        default="none",
        help=(
            "missingness model; incident excludes deterministic prevalent "
            "binary-trait cases before analysis"
        ),
    )
    return parser.parse_args()


def genotype_translation_table():
    # PLINK BED two-bit codes: 00=A1/A1, 10=heterozygous, 11=A2/A2.
    # Map uniform two-bit random values to 1:2:1 genotype frequencies and no
    # missing values (01 is PLINK's missing code).
    mapped_codes = (0b00, 0b10, 0b10, 0b11)
    table = bytearray(256)
    for raw_byte in range(256):
        encoded = 0
        for sample_in_byte in range(4):
            raw_code = (raw_byte >> (2 * sample_in_byte)) & 0b11
            encoded |= mapped_codes[raw_code] << (2 * sample_in_byte)
        table[raw_byte] = encoded
    return bytes(table)


def random_bytes(rng, count):
    if hasattr(rng, "randbytes"):
        return rng.randbytes(count)
    return rng.getrandbits(count * 8).to_bytes(count, "little")


def write_bed(path, samples, variants, seed, variants_per_chunk):
    bytes_per_variant = (samples + 3) // 4
    remainder = samples % 4
    padding_mask = (1 << (2 * remainder)) - 1 if remainder else 0xFF
    translation = genotype_translation_table()
    rng = random.Random(seed)
    digest = hashlib.sha256()

    with path.open("wb") as output:
        output.write(BED_MAGIC)
        digest.update(BED_MAGIC)
        for first_variant in range(0, variants, variants_per_chunk):
            chunk_variants = min(variants_per_chunk, variants - first_variant)
            encoded = bytearray(
                random_bytes(rng, chunk_variants * bytes_per_variant).translate(
                    translation
                )
            )
            if remainder:
                for variant_in_chunk in range(chunk_variants):
                    last_byte = (variant_in_chunk + 1) * bytes_per_variant - 1
                    encoded[last_byte] &= padding_mask
            output.write(encoded)
            digest.update(encoded)
    return bytes_per_variant, digest.hexdigest()


def write_fam(path, samples):
    with path.open("w", encoding="utf-8") as output:
        for sample in range(1, samples + 1):
            sex = 1 + (sample % 2)
            output.write(f"{sample} {sample} 0 0 {sex} -9\n")


def proportional_counts(total, weights):
    if total < len(weights):
        raise ValueError("total must be at least the number of weights")
    counts = [1] * len(weights)
    remaining = total - len(weights)
    weight_sum = sum(weights)
    exact = [remaining * weight / weight_sum for weight in weights]
    floors = [int(value) for value in exact]
    for index, value in enumerate(floors):
        counts[index] += value
    unassigned = remaining - sum(floors)
    order = sorted(
        range(len(weights)),
        key=lambda index: (exact[index] - floors[index], weights[index]),
        reverse=True,
    )
    for index in order[:unassigned]:
        counts[index] += 1
    return counts


def write_bim(path, variants, chromosomes, chromosome_layout):
    chromosome_counts = [0] * chromosomes
    with path.open("w", encoding="utf-8") as output:
        if chromosome_layout == "equal":
            for variant in range(1, variants + 1):
                chromosome = 1 + ((variant - 1) * chromosomes // variants)
                chromosome_counts[chromosome - 1] += 1
                chromosome_variant = (variant - 1) % 1000000 + 1
                position = chromosome_variant * 10
                output.write(
                    f"{chromosome} rsSynthetic{variant} 0 {position} A G\n"
                )
        else:
            chromosome_counts = proportional_counts(
                variants, HUMAN_AUTOSOME_LENGTHS[:chromosomes]
            )
            variant = 1
            for chromosome, (count, chromosome_length) in enumerate(
                zip(
                    chromosome_counts,
                    HUMAN_AUTOSOME_LENGTHS[:chromosomes],
                ),
                start=1,
            ):
                for chromosome_variant in range(1, count + 1):
                    position = (
                        chromosome_variant * chromosome_length // (count + 1)
                    )
                    output.write(
                        f"{chromosome} rsSynthetic{variant} 0 {position} A G\n"
                    )
                    variant += 1
    return chromosome_counts


def covariate_values(sample):
    age = 35 + (sample % 45)
    sex = sample % 2
    pc1 = math.sin(sample * 0.017) + ((sample % 13) - 6) * 0.003
    pc2 = math.cos(sample * 0.013) + ((sample % 17) - 8) * 0.002
    return age, sex, pc1, pc2


def phenotype_value(sample, phenotype, seed):
    phase = (seed % 1009) * 0.0001 + phenotype * 0.37
    trend = ((sample * (phenotype * 2 + 1)) % 101 - 50) * 0.006
    return (
        math.sin(sample * (0.0047 + phenotype * 0.0003) + phase)
        + 0.55 * math.cos(sample * (0.0091 + phenotype * 0.0002) - phase)
        + trend
    )


def deterministic_uniform(sample, phenotype, seed, stream):
    """Return a stable pseudo-random value in [0, 1) without global state."""
    mask = (1 << 64) - 1
    value = (
        seed
        + sample * 0x9E3779B97F4A7C15
        + phenotype * 0xBF58476D1CE4E5B9
        + stream * 0x94D049BB133111EB
    ) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return ((value >> 11) & ((1 << 53) - 1)) / float(1 << 53)


def logistic_probability(base_probability, linear_predictor):
    base_log_odds = math.log(base_probability / (1.0 - base_probability))
    log_odds = base_log_odds + linear_predictor
    if log_odds >= 0:
        inverse = math.exp(-log_odds)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(log_odds)
    return exponent / (1.0 + exponent)


def binary_phenotype_value(sample, phenotype, seed, missingness_profile):
    # Four recurring incidence/missingness profiles keep multi-phenotype
    # benchmarks representative without making their behavior seed-dependent.
    incidence_probabilities = (0.01, 0.02, 0.04, 0.08)
    prevalent_probabilities = (0.03, 0.05, 0.07, 0.09)
    profile = (phenotype - 1) % len(incidence_probabilities)
    age, sex, pc1, pc2 = covariate_values(sample)
    shared_risk = (
        0.018 * (age - 57.0)
        + 0.22 * (sex - 0.5)
        + 0.16 * pc1
        - 0.10 * pc2
    )

    if missingness_profile == "incident":
        prevalent_probability = logistic_probability(
            prevalent_probabilities[profile], 1.15 * shared_risk
        )
        if deterministic_uniform(sample, phenotype, seed, 1) < prevalent_probability:
            return "NA", False, True

    incidence_probability = logistic_probability(
        incidence_probabilities[profile], shared_risk
    )
    is_case = (
        deterministic_uniform(sample, phenotype, seed, 2)
        < incidence_probability
    )
    return ("1" if is_case else "0"), is_case, False


def write_covariates(path, samples):
    with path.open("w", encoding="utf-8") as output:
        output.write("FID IID AGE SEX PC1 PC2\n")
        for sample in range(1, samples + 1):
            age, sex, pc1, pc2 = covariate_values(sample)
            output.write(
                f"{sample} {sample} {age} {sex} {pc1:.12g} {pc2:.12g}\n"
            )


def write_phenotypes(
    path, samples, phenotypes, seed, trait_type, missingness_profile
):
    case_counts = [0] * phenotypes
    missing_counts = [0] * phenotypes
    with path.open("w", encoding="utf-8") as output:
        phenotype_names = " ".join(
            f"Y{phenotype}" for phenotype in range(1, phenotypes + 1)
        )
        output.write(f"FID IID {phenotype_names}\n")
        for sample in range(1, samples + 1):
            if trait_type == "qt":
                values = " ".join(
                    f"{phenotype_value(sample, phenotype, seed):.12g}"
                    for phenotype in range(1, phenotypes + 1)
                )
            else:
                binary_values = []
                for phenotype in range(1, phenotypes + 1):
                    value, is_case, is_missing = binary_phenotype_value(
                        sample, phenotype, seed, missingness_profile
                    )
                    binary_values.append(value)
                    case_counts[phenotype - 1] += int(is_case)
                    missing_counts[phenotype - 1] += int(is_missing)
                values = " ".join(binary_values)
            output.write(f"{sample} {sample} {values}\n")
    return [
        {
            "name": f"Y{phenotype + 1}",
            "cases": case_counts[phenotype] if trait_type == "bt" else None,
            "missing": missing_counts[phenotype],
            "observed": samples - missing_counts[phenotype],
        }
        for phenotype in range(phenotypes)
    ]


def main():
    args = parse_args()
    started = time.monotonic()
    bytes_per_variant = (args.samples + 3) // 4
    bed_bytes = len(BED_MAGIC) + args.variants * bytes_per_variant
    maximum_bed_bytes = int(args.max_bed_gb * 1024**3)
    if bed_bytes > maximum_bed_bytes:
        raise SystemExit(
            "requested BED size "
            f"({bed_bytes / 1024**3:.3f} GiB) exceeds --max-bed-gb "
            f"({args.max_bed_gb:g} GiB)"
        )
    if args.chromosomes > args.variants:
        raise SystemExit("--chromosomes must not exceed --variants")
    if (
        args.chromosome_layout == "human"
        and args.chromosomes > len(HUMAN_AUTOSOME_LENGTHS)
    ):
        raise SystemExit(
            "--chromosome-layout human supports at most 22 chromosomes"
        )
    if args.missingness_profile == "incident" and args.trait_type != "bt":
        raise SystemExit(
            "--missingness-profile incident requires --trait-type bt"
        )

    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "bed": args.prefix.with_suffix(".bed"),
        "bim": args.prefix.with_suffix(".bim"),
        "fam": args.prefix.with_suffix(".fam"),
        "covar": args.prefix.with_suffix(".covar"),
        "pheno": args.prefix.with_suffix(".pheno"),
        "manifest": args.prefix.with_suffix(".json"),
    }
    temporary = {
        name: path.with_name(f".{path.name}.{os.getpid()}.tmp")
        for name, path in outputs.items()
    }

    try:
        actual_bytes_per_variant, bed_sha256 = write_bed(
            temporary["bed"],
            args.samples,
            args.variants,
            args.seed,
            args.variants_per_chunk,
        )
        chromosome_counts = write_bim(
            temporary["bim"],
            args.variants,
            args.chromosomes,
            args.chromosome_layout,
        )
        write_fam(temporary["fam"], args.samples)
        write_covariates(temporary["covar"], args.samples)
        phenotype_stats = write_phenotypes(
            temporary["pheno"],
            args.samples,
            args.phenotypes,
            args.seed,
            args.trait_type,
            args.missingness_profile,
        )

        elapsed_ms = (time.monotonic() - started) * 1000.0
        manifest = {
            "format": "PLINK BED variant-major",
            "samples": args.samples,
            "variants": args.variants,
            "phenotypes": args.phenotypes,
            "trait_type": args.trait_type,
            "missingness_profile": args.missingness_profile,
            "phenotype_stats": phenotype_stats,
            "chromosomes": args.chromosomes,
            "chromosome_layout": args.chromosome_layout,
            "chromosome_variant_counts": chromosome_counts,
            "seed": args.seed,
            "variants_per_chunk": args.variants_per_chunk,
            "bytes_per_variant": actual_bytes_per_variant,
            "bed_bytes": bed_bytes,
            "bed_sha256": bed_sha256,
            "generation_ms": elapsed_ms,
        }
        with temporary["manifest"].open("w", encoding="utf-8") as output:
            json.dump(manifest, output, indent=2, sort_keys=True)
            output.write("\n")

        for name, path in outputs.items():
            os.replace(temporary[name], path)
    finally:
        for path in temporary.values():
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    print(
        "STEP1_SYNTHETIC_DATA status=PASS "
        f"samples={args.samples} variants={args.variants} "
        f"phenotypes={args.phenotypes} chromosomes={args.chromosomes} "
        f"chromosome_layout={args.chromosome_layout} "
        f"trait_type={args.trait_type} "
        f"missingness_profile={args.missingness_profile} "
        f"seed={args.seed} variants_per_chunk={args.variants_per_chunk} "
        f"bed_bytes={bed_bytes} "
        f"bed_sha256={bed_sha256} generation_ms={elapsed_ms:.3f} "
        f"prefix={args.prefix}"
    )


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
