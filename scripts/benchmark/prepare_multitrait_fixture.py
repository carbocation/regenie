#!/usr/bin/env python3
"""Create deterministic stratified multi-trait benchmark inputs."""

from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Sample:
    source_index: int
    fid: str
    iid: str
    group: str
    phenotype: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select equal-sized population strata and create correlated quantitative "
            "traits with complete and variable-missingness versions."
        )
    )
    parser.add_argument("source_pheno", type=Path)
    parser.add_argument("source_covar", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--source-trait", default="PHENO")
    parser.add_argument("--group-column", default="SUPERPOP")
    parser.add_argument("--samples-per-group", type=int, default=10_000)
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help=(
            "Use every source sample in input order instead of drawing equal-sized "
            "group strata."
        ),
    )
    parser.add_argument("--traits", type=int, default=32)
    parser.add_argument("--trait-correlation", type=float, default=0.4)
    parser.add_argument("--max-missing-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20_260_720)
    return parser.parse_args()


def normalize_id_header(value: str) -> str:
    return value.removeprefix("#")


def column_index(header: list[str], name: str, path: Path) -> int:
    try:
        return header.index(name)
    except ValueError as error:
        raise SystemExit(f"Column {name!r} is missing from {path}") from error


def standardized(values: np.ndarray, label: str) -> np.ndarray:
    centered = values - values.mean()
    standard_deviation = centered.std()
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        raise SystemExit(f"{label} has zero or invalid variance")
    return centered / standard_deviation


def select_samples(args: argparse.Namespace, rng: np.random.Generator) -> list[Sample]:
    reservoirs: dict[str, list[Sample]] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)

    with args.source_pheno.open() as pheno, args.source_covar.open() as covar:
        pheno_header = pheno.readline().split()
        covar_header = covar.readline().split()
        if [normalize_id_header(value) for value in pheno_header[:2]] != ["FID", "IID"]:
            raise SystemExit(f"Expected FID IID columns in {args.source_pheno}")
        if [normalize_id_header(value) for value in covar_header[:2]] != ["FID", "IID"]:
            raise SystemExit(f"Expected FID IID columns in {args.source_covar}")
        trait_index = column_index(pheno_header, args.source_trait, args.source_pheno)
        group_index = column_index(covar_header, args.group_column, args.source_covar)

        for source_index, (pheno_line, covar_line) in enumerate(
            itertools.zip_longest(pheno, covar), 1
        ):
            if pheno_line is None or covar_line is None:
                raise SystemExit(
                    "Phenotype and covariate tables have different lengths"
                )
            pheno_fields = pheno_line.split()
            covar_fields = covar_line.split()
            if pheno_fields[:2] != covar_fields[:2]:
                raise SystemExit(
                    f"Sample order differs at input row {source_index + 1}"
                )
            try:
                phenotype = float(pheno_fields[trait_index])
                group = covar_fields[group_index]
            except (IndexError, ValueError) as error:
                raise SystemExit(f"Invalid input row {source_index + 1}") from error
            if not np.isfinite(phenotype):
                raise SystemExit(
                    f"Non-finite source phenotype at row {source_index + 1}"
                )

            sample = Sample(
                source_index, pheno_fields[0], pheno_fields[1], group, phenotype
            )
            seen[group] += 1
            reservoir = reservoirs[group]
            if args.all_samples:
                reservoir.append(sample)
                continue
            if len(reservoir) < args.samples_per_group:
                reservoir.append(sample)
            else:
                replacement = int(rng.integers(seen[group]))
                if replacement < args.samples_per_group:
                    reservoir[replacement] = sample

    if not reservoirs:
        raise SystemExit("No samples were read")
    undersized = (
        {}
        if args.all_samples
        else {
            group: count
            for group, count in seen.items()
            if count < args.samples_per_group
        }
    )
    if undersized:
        details = ", ".join(
            f"{group}={count}" for group, count in sorted(undersized.items())
        )
        raise SystemExit(f"Groups smaller than --samples-per-group: {details}")

    return sorted(
        (sample for reservoir in reservoirs.values() for sample in reservoir),
        key=lambda sample: sample.source_index,
    )


def create_traits(
    source: np.ndarray,
    traits: int,
    correlation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    columns = np.empty((source.size, traits), dtype=np.float64)
    columns[:, 0] = standardized(source, "source phenotype")
    residual_scale = np.sqrt(1 - correlation * correlation)
    for trait in range(1, traits):
        noise = standardized(rng.normal(size=source.size), f"trait {trait + 1} noise")
        columns[:, trait] = standardized(
            correlation * columns[:, 0] + residual_scale * noise,
            f"trait {trait + 1}",
        )
    return columns


def missing_masks(
    samples: int,
    traits: int,
    max_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    rates = np.linspace(0.0, max_rate, traits)
    masks = np.zeros((samples, traits), dtype=np.bool_)
    for trait, rate in enumerate(rates):
        missing_count = round(rate * samples)
        if missing_count:
            masks[rng.choice(samples, size=missing_count, replace=False), trait] = True
    return rates, masks


def output_path(prefix: Path, suffix: str) -> Path:
    return Path(f"{prefix}{suffix}")


def write_fixture(
    prefix: Path,
    samples: list[Sample],
    traits: np.ndarray,
    rates: np.ndarray,
    masks: np.ndarray,
    group_column: str,
) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    trait_names = [f"PHENO{index + 1}" for index in range(traits.shape[1])]

    with output_path(prefix, ".keep").open("w") as stream:
        stream.write("#FID\tIID\n")
        for sample in samples:
            stream.write(f"{sample.fid}\t{sample.iid}\n")

    with output_path(prefix, ".covar").open("w") as stream:
        stream.write(f"FID\tIID\t{group_column}\n")
        for sample in samples:
            stream.write(f"{sample.fid}\t{sample.iid}\t{sample.group}\n")

    header = "FID\tIID\t" + "\t".join(trait_names) + "\n"
    with output_path(prefix, ".complete.pheno").open("w") as complete:
        complete.write(header)
        for row, sample in zip(traits, samples):
            values = "\t".join(f"{value:.12g}" for value in row)
            complete.write(f"{sample.fid}\t{sample.iid}\t{values}\n")

    with output_path(prefix, ".missing.pheno").open("w") as missing:
        missing.write(header)
        for sample_index, (row, sample) in enumerate(zip(traits, samples)):
            values = "\t".join(
                "NA" if masks[sample_index, trait] else f"{value:.12g}"
                for trait, value in enumerate(row)
            )
            missing.write(f"{sample.fid}\t{sample.iid}\t{values}\n")

    with output_path(prefix, ".missingness.tsv").open("w") as stream:
        stream.write("trait\tmissing_rate\tmissing_samples\tobserved_samples\n")
        for trait, rate in enumerate(rates):
            count = int(masks[:, trait].sum())
            stream.write(
                f"{trait_names[trait]}\t{rate:.9f}\t{count}\t{len(samples) - count}\n"
            )


def main() -> None:
    args = parse_args()
    if args.samples_per_group < 1:
        raise SystemExit("--samples-per-group must be positive")
    if args.traits < 1:
        raise SystemExit("--traits must be positive")
    if not -1 < args.trait_correlation < 1:
        raise SystemExit("--trait-correlation must be between -1 and 1")
    if not 0 <= args.max_missing_rate < 1:
        raise SystemExit("--max-missing-rate must be in [0, 1)")

    selection_seed, trait_seed, missing_seed = np.random.SeedSequence(args.seed).spawn(
        3
    )
    samples = select_samples(args, np.random.default_rng(selection_seed))
    source = np.asarray([sample.phenotype for sample in samples], dtype=np.float64)
    traits = create_traits(
        source, args.traits, args.trait_correlation, np.random.default_rng(trait_seed)
    )
    rates, masks = missing_masks(
        len(samples),
        args.traits,
        args.max_missing_rate,
        np.random.default_rng(missing_seed),
    )
    write_fixture(args.output_prefix, samples, traits, rates, masks, args.group_column)

    group_counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        group_counts[sample.group] += 1
    groups = ", ".join(
        f"{group}={count}" for group, count in sorted(group_counts.items())
    )
    print(f"Wrote {len(samples)} samples and {args.traits} traits ({groups})")
    print(
        f"Missingness ranges from {rates[0]:.3%} to {rates[-1]:.3%}; "
        f"seed={args.seed}"
    )


if __name__ == "__main__":
    main()
