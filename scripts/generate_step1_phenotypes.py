#!/usr/bin/env python3
"""Generate deterministic Step 1 phenotype/covariate files without genotypes."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from generate_step1_bed import (
    positive_int,
    write_covariates,
    write_phenotypes,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--samples", type=positive_int, required=True)
    parser.add_argument("--phenotypes", type=positive_int, required=True)
    parser.add_argument("--seed", type=positive_int, default=20260712)
    parser.add_argument(
        "--trait-type", choices=("qt", "bt"), default="qt"
    )
    parser.add_argument(
        "--missingness-profile",
        choices=("none", "incident"),
        default="none",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.missingness_profile == "incident" and args.trait_type != "bt":
        raise SystemExit(
            "--missingness-profile incident requires --trait-type bt"
        )

    started = time.monotonic()
    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "covar": args.prefix.with_suffix(".covar"),
        "pheno": args.prefix.with_suffix(".pheno"),
        "manifest": args.prefix.with_suffix(".json"),
    }
    temporary = {
        name: path.with_name(f".{path.name}.{os.getpid()}.tmp")
        for name, path in outputs.items()
    }

    try:
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
            "samples": args.samples,
            "phenotypes": args.phenotypes,
            "trait_type": args.trait_type,
            "missingness_profile": args.missingness_profile,
            "phenotype_stats": phenotype_stats,
            "seed": args.seed,
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
        "STEP1_SYNTHETIC_PHENOTYPES status=PASS "
        f"samples={args.samples} phenotypes={args.phenotypes} "
        f"trait_type={args.trait_type} "
        f"missingness_profile={args.missingness_profile} "
        f"seed={args.seed} generation_ms={elapsed_ms:.3f} "
        f"prefix={args.prefix}"
    )


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
