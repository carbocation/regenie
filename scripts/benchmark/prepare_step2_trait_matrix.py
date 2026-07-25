#!/usr/bin/env python3
"""Derive matched quantitative, binary, and survival Step 2 panels."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np


BINARY_RATES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
EVENT_RATES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create complete and missing binary and survival panels from "
            "matched quantitative phenotype files."
        )
    )
    parser.add_argument("complete_pheno", type=Path)
    parser.add_argument("missing_pheno", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--trait-prefix", default="PHENO")
    parser.add_argument("--traits", type=int, default=32)
    parser.add_argument(
        "--time-coefficient",
        type=float,
        default=-0.5,
        help="Coefficient in time=exp(coefficient * quantitative_trait).",
    )
    return parser.parse_args()


def normalized_id_header(value: str) -> str:
    return value.removeprefix("#")


def read_panel(
    path: Path, trait_prefix: str, trait_count: int, allow_missing: bool
) -> tuple[list[str], list[tuple[str, str]], np.ndarray]:
    with path.open() as stream:
        header = stream.readline().split()
        if [normalized_id_header(value) for value in header[:2]] != ["FID", "IID"]:
            raise SystemExit(f"Expected FID IID columns in {path}")
        trait_names = [f"{trait_prefix}{index + 1}" for index in range(trait_count)]
        try:
            trait_indices = [header.index(name) for name in trait_names]
        except ValueError as error:
            raise SystemExit(
                f"Expected traits {', '.join(trait_names)} in {path}"
            ) from error

        sample_ids: list[tuple[str, str]] = []
        rows: list[list[float]] = []
        for line_number, line in enumerate(stream, 2):
            fields = line.split()
            if len(fields) <= max(trait_indices):
                raise SystemExit(f"Short row {line_number} in {path}")
            sample_ids.append((fields[0], fields[1]))
            values: list[float] = []
            for index in trait_indices:
                value = fields[index]
                if allow_missing and value.upper() == "NA":
                    values.append(float("nan"))
                    continue
                try:
                    parsed = float(value)
                except ValueError as error:
                    raise SystemExit(
                        f"Invalid value at row {line_number} in {path}"
                    ) from error
                if not np.isfinite(parsed):
                    raise SystemExit(
                        f"Non-finite value at row {line_number} in {path}"
                    )
                values.append(parsed)
            rows.append(values)

    if not rows:
        raise SystemExit(f"No samples were read from {path}")
    return header[:2], sample_ids, np.asarray(rows, dtype=np.float64)


def normal_threshold_indicators(
    values: np.ndarray, rates: tuple[float, ...]
) -> tuple[np.ndarray, list[int]]:
    samples, traits = values.shape
    indicators = np.zeros((samples, traits), dtype=np.uint8)
    counts: list[int] = []
    for trait in range(traits):
        rate = rates[trait % len(rates)]
        # Keep the threshold stable across standard-library implementations and
        # make the intended approximate prevalence obvious in the fixture.
        threshold = round(NormalDist().inv_cdf(1 - rate), 5)
        indicators[:, trait] = values[:, trait] > threshold
        count = int(indicators[:, trait].sum())
        if not 0 < count < samples:
            raise SystemExit(f"Trait {trait + 1} needs both outcome classes")
        counts.append(count)
    return indicators, counts


def write_binary(
    path: Path,
    id_header: list[str],
    sample_ids: list[tuple[str, str]],
    values: np.ndarray,
    missing: np.ndarray,
) -> None:
    with path.open("w") as stream:
        traits = "\t".join(f"BT{index + 1}" for index in range(values.shape[1]))
        stream.write(f"{id_header[0]}\t{id_header[1]}\t{traits}\n")
        for sample, (fid, iid) in enumerate(sample_ids):
            row = "\t".join(
                "NA" if missing[sample, trait] else str(values[sample, trait])
                for trait in range(values.shape[1])
            )
            stream.write(f"{fid}\t{iid}\t{row}\n")


def write_survival(
    path: Path,
    id_header: list[str],
    sample_ids: list[tuple[str, str]],
    times: np.ndarray,
    events: np.ndarray,
    missing: np.ndarray,
) -> None:
    with path.open("w") as stream:
        traits = "\t".join(
            f"TIME{index + 1}\tEVENT{index + 1}"
            for index in range(times.shape[1])
        )
        stream.write(f"{id_header[0]}\t{id_header[1]}\t{traits}\n")
        for sample, (fid, iid) in enumerate(sample_ids):
            fields: list[str] = []
            for trait in range(times.shape[1]):
                if missing[sample, trait]:
                    fields.extend(("NA", "NA"))
                else:
                    fields.extend(
                        (f"{times[sample, trait]:.12g}", str(events[sample, trait]))
                    )
            row = "\t".join(fields)
            stream.write(f"{fid}\t{iid}\t{row}\n")


def write_summary(
    path: Path,
    samples: int,
    binary_counts: list[int],
    event_counts: list[int],
    missing: np.ndarray,
) -> None:
    with path.open("w") as stream:
        stream.write(
            "trait\tbinary_prevalence\tevent_fraction\tmissing_count\t"
            "missing_fraction\n"
        )
        for trait, (binary_count, event_count) in enumerate(
            zip(binary_counts, event_counts)
        ):
            missing_count = int(missing[:, trait].sum())
            stream.write(
                f"{trait + 1}\t{binary_count / samples:.9f}\t"
                f"{event_count / samples:.9f}\t{missing_count}\t"
                f"{missing_count / samples:.9f}\n"
            )


def output_path(prefix: Path, suffix: str) -> Path:
    return Path(f"{prefix}{suffix}")


def main() -> None:
    args = parse_args()
    if args.traits <= 0:
        raise SystemExit("--traits must be positive")
    if not np.isfinite(args.time_coefficient):
        raise SystemExit("--time-coefficient must be finite")

    id_header, sample_ids, complete = read_panel(
        args.complete_pheno, args.trait_prefix, args.traits, False
    )
    missing_id_header, missing_sample_ids, missing_values = read_panel(
        args.missing_pheno, args.trait_prefix, args.traits, True
    )
    if id_header != missing_id_header or sample_ids != missing_sample_ids:
        raise SystemExit("Complete and missing panels do not have identical samples")
    missing = np.isnan(missing_values)
    observed = ~missing
    if not np.array_equal(missing_values[observed], complete[observed]):
        raise SystemExit("Observed values differ between complete and missing panels")

    binary, binary_counts = normal_threshold_indicators(complete, BINARY_RATES)
    events, event_counts = normal_threshold_indicators(complete, EVENT_RATES)
    times = np.fromiter(
        (
            math.exp(args.time_coefficient * value)
            for value in complete.flat
        ),
        dtype=np.float64,
        count=complete.size,
    ).reshape(complete.shape)
    if not np.isfinite(times).all():
        raise SystemExit("Derived survival times are not finite")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    no_missing = np.zeros_like(missing)
    write_binary(
        output_path(args.output_prefix, ".binary-complete.pheno"),
        id_header,
        sample_ids,
        binary,
        no_missing,
    )
    write_binary(
        output_path(args.output_prefix, ".binary-missing.pheno"),
        id_header,
        sample_ids,
        binary,
        missing,
    )
    write_survival(
        output_path(args.output_prefix, ".survival-complete.pheno"),
        id_header,
        sample_ids,
        times,
        events,
        no_missing,
    )
    write_survival(
        output_path(args.output_prefix, ".survival-missing.pheno"),
        id_header,
        sample_ids,
        times,
        events,
        missing,
    )
    write_summary(
        output_path(args.output_prefix, ".trait-matrix.tsv"),
        len(sample_ids),
        binary_counts,
        event_counts,
        missing,
    )
    print(
        f"Wrote {args.traits} binary and survival traits for "
        f"{len(sample_ids)} samples; mean missingness={missing.mean():.3%}"
    )


if __name__ == "__main__":
    main()
