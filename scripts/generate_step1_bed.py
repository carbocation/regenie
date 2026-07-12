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
    parser.add_argument("--seed", type=positive_int, default=20260712)
    parser.add_argument("--variants-per-chunk", type=positive_int, default=256)
    parser.add_argument("--max-bed-gb", type=positive_float, default=4.0)
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


def write_bim(path, variants, chromosomes):
    with path.open("w", encoding="utf-8") as output:
        for variant in range(1, variants + 1):
            chromosome = 1 + ((variant - 1) * chromosomes // variants)
            chromosome_variant = (variant - 1) % 1000000 + 1
            position = chromosome_variant * 10
            output.write(
                f"{chromosome} rsSynthetic{variant} 0 {position} A G\n"
            )


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


def write_covariates(path, samples):
    with path.open("w", encoding="utf-8") as output:
        output.write("FID IID AGE SEX PC1 PC2\n")
        for sample in range(1, samples + 1):
            age, sex, pc1, pc2 = covariate_values(sample)
            output.write(
                f"{sample} {sample} {age} {sex} {pc1:.12g} {pc2:.12g}\n"
            )


def write_phenotypes(path, samples, phenotypes, seed):
    with path.open("w", encoding="utf-8") as output:
        phenotype_names = " ".join(
            f"Y{phenotype}" for phenotype in range(1, phenotypes + 1)
        )
        output.write(f"FID IID {phenotype_names}\n")
        for sample in range(1, samples + 1):
            values = " ".join(
                f"{phenotype_value(sample, phenotype, seed):.12g}"
                for phenotype in range(1, phenotypes + 1)
            )
            output.write(f"{sample} {sample} {values}\n")


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
        write_bim(temporary["bim"], args.variants, args.chromosomes)
        write_fam(temporary["fam"], args.samples)
        write_covariates(temporary["covar"], args.samples)
        write_phenotypes(
            temporary["pheno"], args.samples, args.phenotypes, args.seed
        )

        elapsed_ms = (time.monotonic() - started) * 1000.0
        manifest = {
            "format": "PLINK BED variant-major",
            "samples": args.samples,
            "variants": args.variants,
            "phenotypes": args.phenotypes,
            "chromosomes": args.chromosomes,
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
