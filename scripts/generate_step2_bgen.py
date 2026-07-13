#!/usr/bin/env python3
"""Generate a deterministic BGEN v1.2 dosage dataset for Step 2 benchmarks."""

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
import zlib
from pathlib import Path

try:
    import numpy as np
except ImportError as error:
    raise SystemExit(
        "generate_step2_bgen.py requires NumPy; install it for the Python "
        "interpreter running this script"
    ) from error


BGEN_HEADER_LENGTH = 20
BGEN_LAYOUT2_ZLIB_FLAGS = 1 | (2 << 2)
LOOKUP_TABLE_ENTRIES = 65536
VALID_8BIT_PROBABILITY_PAIRS = 256 * 257 // 2


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


def compression_level(value):
    parsed = int(value)
    if parsed < 0 or parsed > 9:
        raise argparse.ArgumentTypeError("must be between 0 and 9")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--samples", type=positive_int, default=100000)
    parser.add_argument("--variants", type=positive_int, default=1000)
    parser.add_argument("--chromosomes", type=positive_int, default=2)
    parser.add_argument("--seed", type=positive_int, default=20260712)
    parser.add_argument("--compression-level", type=compression_level, default=1)
    parser.add_argument("--max-bgen-gb", type=positive_float, default=4.0)
    return parser.parse_args()


def length_prefixed_u16(value):
    encoded = value.encode("utf-8")
    if len(encoded) > 65535:
        raise ValueError("BGEN identifying string exceeds 65535 bytes")
    return struct.pack("<H", len(encoded)) + encoded


def length_prefixed_u32(value):
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def write_sample_file(path, samples):
    with path.open("w", encoding="utf-8") as output:
        output.write("ID_1 ID_2 missing\n")
        output.write("0 0 0\n")
        for sample in range(1, samples + 1):
            output.write(f"{sample} {sample} 0\n")


def variant_probability_bytes(rng, samples, allele_frequency):
    """Return valid, deliberately diverse 8-bit unphased diploid probabilities."""
    genotype = rng.binomial(2, allele_frequency, size=samples)
    uncertainty = rng.beta(2.0, 8.0, size=samples) * 0.75
    split = rng.random(samples)

    first_minor = uncertainty * split
    second_minor = uncertainty - first_minor
    prob0 = first_minor.copy()
    prob1 = second_minor.copy()
    genotype0 = genotype == 0
    genotype1 = genotype == 1
    prob0[genotype0] = 1.0 - uncertainty[genotype0]
    prob1[genotype0] = first_minor[genotype0]
    prob1[genotype1] = 1.0 - uncertainty[genotype1]

    encoded0 = np.rint(prob0 * 255.0).astype(np.uint16)
    encoded1 = np.rint(prob1 * 255.0).astype(np.uint16)
    encoded1 = np.minimum(encoded1, 255 - encoded0)

    probabilities = np.empty((samples, 2), dtype=np.uint8)
    probabilities[:, 0] = encoded0
    probabilities[:, 1] = encoded1
    lookup_indices = encoded0 | (encoded1 << 8)
    return probabilities.tobytes(), lookup_indices


def write_bgen(path, samples, variants, chromosomes, seed, level, maximum_bytes):
    rng = np.random.default_rng(seed)
    ploidy = bytes([2]) * samples
    used_lookup_indices = np.zeros(LOOKUP_TABLE_ENTRIES, dtype=np.bool_)
    digest = hashlib.sha256()
    uncompressed_bytes = 0
    compressed_bytes = 0

    header = struct.pack(
        "<IIII4sI",
        BGEN_HEADER_LENGTH,
        BGEN_HEADER_LENGTH,
        variants,
        samples,
        b"bgen",
        BGEN_LAYOUT2_ZLIB_FLAGS,
    )

    with path.open("wb") as output:
        output.write(header)
        digest.update(header)

        for variant in range(1, variants + 1):
            chromosome = 1 + ((variant - 1) * chromosomes // variants)
            identifier = f"rsSyntheticDosage{variant}"
            identifying = b"".join(
                (
                    length_prefixed_u16(identifier),
                    length_prefixed_u16(identifier),
                    length_prefixed_u16(str(chromosome)),
                    struct.pack("<IH", variant * 10, 2),
                    length_prefixed_u32("A"),
                    length_prefixed_u32("G"),
                )
            )

            frequency_cycle = ((variant * 7919) % 10000) / 10000.0
            allele_frequency = 0.02 + 0.46 * frequency_cycle
            probabilities, lookup_indices = variant_probability_bytes(
                rng, samples, allele_frequency
            )
            used_lookup_indices[lookup_indices] = True

            genotype_data = b"".join(
                (
                    struct.pack("<IHBB", samples, 2, 2, 2),
                    ploidy,
                    bytes((0, 8)),
                    probabilities,
                )
            )
            compressed = zlib.compress(genotype_data, level)
            genotype_block = struct.pack(
                "<II", len(compressed) + 4, len(genotype_data)
            ) + compressed

            output.write(identifying)
            output.write(genotype_block)
            digest.update(identifying)
            digest.update(genotype_block)
            uncompressed_bytes += len(genotype_data)
            compressed_bytes += len(compressed)

            if output.tell() > maximum_bytes:
                raise RuntimeError(
                    "generated BGEN exceeded --max-bgen-gb; increase the cap "
                    "or request fewer samples/variants"
                )

    return {
        "bgen_bytes": path.stat().st_size,
        "bgen_sha256": digest.hexdigest(),
        "genotype_uncompressed_bytes": uncompressed_bytes,
        "genotype_compressed_bytes": compressed_bytes,
        "lookup_pairs": int(np.count_nonzero(used_lookup_indices)),
    }


def main():
    args = parse_args()
    if args.chromosomes > args.variants:
        raise SystemExit("--chromosomes must not exceed --variants")

    maximum_bytes = int(args.max_bgen_gb * 1024**3)
    conservative_bytes = 24 + args.variants * (128 + 3 * args.samples)
    if conservative_bytes > maximum_bytes:
        raise SystemExit(
            "conservative generated-size bound "
            f"({conservative_bytes / 1024**3:.3f} GiB) exceeds "
            f"--max-bgen-gb ({args.max_bgen_gb:g} GiB)"
        )

    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "bgen": args.prefix.with_suffix(".bgen"),
        "sample": args.prefix.with_suffix(".sample"),
        "manifest": args.prefix.with_suffix(".json"),
    }
    temporary = {
        name: path.with_name(f".{path.name}.{os.getpid()}.tmp")
        for name, path in outputs.items()
    }
    started = time.monotonic()

    try:
        bgen_summary = write_bgen(
            temporary["bgen"],
            args.samples,
            args.variants,
            args.chromosomes,
            args.seed,
            args.compression_level,
            maximum_bytes,
        )
        write_sample_file(temporary["sample"], args.samples)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        manifest = {
            "format": "BGEN v1.2 zlib-compressed 8-bit unphased diploid",
            "samples": args.samples,
            "variants": args.variants,
            "chromosomes": args.chromosomes,
            "seed": args.seed,
            "compression_level": args.compression_level,
            "numpy_version": np.__version__,
            "generation_ms": elapsed_ms,
            **bgen_summary,
        }
        manifest["lookup_table_coverage"] = (
            bgen_summary["lookup_pairs"] / LOOKUP_TABLE_ENTRIES
        )
        manifest["valid_probability_pair_coverage"] = (
            bgen_summary["lookup_pairs"] / VALID_8BIT_PROBABILITY_PAIRS
        )
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
        "STEP2_SYNTHETIC_BGEN status=PASS "
        f"samples={args.samples} variants={args.variants} "
        f"chromosomes={args.chromosomes} seed={args.seed} "
        f"compression=zlib compression_level={args.compression_level} "
        f"bgen_bytes={bgen_summary['bgen_bytes']} "
        f"lookup_pairs={bgen_summary['lookup_pairs']} "
        "lookup_table_coverage="
        f"{bgen_summary['lookup_pairs'] / LOOKUP_TABLE_ENTRIES:.6f} "
        "valid_pair_coverage="
        f"{bgen_summary['lookup_pairs'] / VALID_8BIT_PROBABILITY_PAIRS:.6f} "
        f"generation_ms={elapsed_ms:.3f} prefix={args.prefix}"
    )


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
