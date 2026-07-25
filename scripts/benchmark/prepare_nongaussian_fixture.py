#!/usr/bin/env python3
"""Create deterministic binary and time-to-event benchmark phenotypes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_BINARY_PREVALENCES = "0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50"
DEFAULT_EVENT_FRACTIONS = "0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80"


def parse_rates(value: str) -> list[float]:
    try:
        rates = [float(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "rates must be comma-separated numbers"
        ) from error
    if not rates or any(not 0 < rate < 1 for rate in rates):
        raise argparse.ArgumentTypeError("every rate must be between 0 and 1")
    return rates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive binary traits with fixed prevalences and survival traits with "
            "fixed event fractions from a complete quantitative-trait table."
        )
    )
    parser.add_argument("source_pheno", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--source-trait-prefix", default="PHENO")
    parser.add_argument(
        "--binary-prevalences",
        type=parse_rates,
        default=parse_rates(DEFAULT_BINARY_PREVALENCES),
    )
    parser.add_argument(
        "--survival-event-fractions",
        type=parse_rates,
        default=parse_rates(DEFAULT_EVENT_FRACTIONS),
    )
    parser.add_argument("--log-hazard-ratio", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20_260_720)
    return parser.parse_args()


def normalize_id_header(value: str) -> str:
    return value.removeprefix("#")


def standardized(values: np.ndarray, label: str) -> np.ndarray:
    centered = values - values.mean()
    standard_deviation = centered.std()
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        raise SystemExit(f"{label} has zero or invalid variance")
    return centered / standard_deviation


def read_source(
    path: Path, trait_prefix: str, trait_count: int
) -> tuple[list[tuple[str, str]], np.ndarray]:
    trait_names = [f"{trait_prefix}{index + 1}" for index in range(trait_count)]
    sample_ids: list[tuple[str, str]] = []
    rows: list[list[float]] = []

    with path.open() as stream:
        header = stream.readline().split()
        if [normalize_id_header(value) for value in header[:2]] != ["FID", "IID"]:
            raise SystemExit(f"Expected FID IID columns in {path}")
        try:
            trait_indices = [header.index(name) for name in trait_names]
        except ValueError as error:
            raise SystemExit(
                f"Expected source traits {', '.join(trait_names)}"
            ) from error

        for line_number, line in enumerate(stream, 2):
            fields = line.split()
            try:
                values = [float(fields[index]) for index in trait_indices]
            except (IndexError, ValueError) as error:
                raise SystemExit(
                    f"Invalid input row {line_number} in {path}"
                ) from error
            if not np.isfinite(values).all():
                raise SystemExit(
                    f"Non-finite source trait at row {line_number} in {path}"
                )
            sample_ids.append((fields[0], fields[1]))
            rows.append(values)

    if not rows:
        raise SystemExit(f"No samples were read from {path}")
    return sample_ids, np.asarray(rows, dtype=np.float64)


def fixed_count_binary_traits(
    source: np.ndarray, prevalences: list[float]
) -> tuple[np.ndarray, list[int]]:
    binary = np.zeros(source.shape, dtype=np.uint8)
    case_counts: list[int] = []
    for trait, prevalence in enumerate(prevalences):
        case_count = round(prevalence * source.shape[0])
        if not 0 < case_count < source.shape[0]:
            raise SystemExit(
                f"Binary trait {trait + 1} needs at least one case and one control"
            )
        order = np.argsort(source[:, trait], kind="stable")
        binary[order[-case_count:], trait] = 1
        case_counts.append(case_count)
    return binary, case_counts


def censoring_rate_for_count(ratios: np.ndarray, event_count: int) -> float:
    descending = np.sort(ratios)[::-1]
    if event_count == 0:
        return float(np.nextafter(descending[0], np.inf))
    if event_count == ratios.size:
        return float(np.nextafter(descending[-1], 0.0))
    return float((descending[event_count - 1] + descending[event_count]) / 2)


def survival_traits(
    source: np.ndarray,
    event_fractions: list[float],
    log_hazard_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    samples, traits = source.shape
    times = np.empty((samples, traits), dtype=np.float64)
    events = np.empty((samples, traits), dtype=np.uint8)
    event_counts: list[int] = []

    for trait, event_fraction in enumerate(event_fractions):
        latent = standardized(source[:, trait], f"source trait {trait + 1}")
        event_draw = -np.log(rng.uniform(np.finfo(float).eps, 1, samples))
        censor_draw = -np.log(rng.uniform(np.finfo(float).eps, 1, samples))
        event_time = event_draw / np.exp(log_hazard_ratio * latent)
        target_count = round(event_fraction * samples)
        if not 0 < target_count < samples:
            raise SystemExit(
                f"Survival trait {trait + 1} needs at least one event and one censor"
            )
        censoring_rate = censoring_rate_for_count(
            censor_draw / event_time, target_count
        )
        censor_time = censor_draw / censoring_rate
        trait_events = event_time <= censor_time
        realized_count = int(trait_events.sum())
        if realized_count != target_count:
            raise RuntimeError(
                f"Could not realize target event count for survival trait {trait + 1}"
            )
        times[:, trait] = np.minimum(event_time, censor_time)
        events[:, trait] = trait_events
        event_counts.append(realized_count)

    return times, events, event_counts


def output_path(prefix: Path, suffix: str) -> Path:
    return Path(f"{prefix}{suffix}")


def write_binary(
    path: Path, sample_ids: list[tuple[str, str]], binary: np.ndarray
) -> None:
    with path.open("w") as stream:
        names = "\t".join(f"BT{index + 1}" for index in range(binary.shape[1]))
        stream.write(f"FID\tIID\t{names}\n")
        for (fid, iid), row in zip(sample_ids, binary):
            values = "\t".join(str(value) for value in row)
            stream.write(f"{fid}\t{iid}\t{values}\n")


def write_survival(
    path: Path,
    sample_ids: list[tuple[str, str]],
    times: np.ndarray,
    events: np.ndarray,
) -> None:
    with path.open("w") as stream:
        names = "\t".join(
            f"TIME{index + 1}\tEVENT{index + 1}" for index in range(times.shape[1])
        )
        stream.write(f"FID\tIID\t{names}\n")
        for sample, (fid, iid) in enumerate(sample_ids):
            values = "\t".join(
                f"{times[sample, trait]:.12g}\t{events[sample, trait]}"
                for trait in range(times.shape[1])
            )
            stream.write(f"{fid}\t{iid}\t{values}\n")


def write_summary(
    path: Path,
    samples: int,
    prevalences: list[float],
    case_counts: list[int],
    event_fractions: list[float],
    event_counts: list[int],
) -> None:
    with path.open("w") as stream:
        stream.write("model\ttrait\ttarget_fraction\tcount\trealized_fraction\n")
        for trait, (target, count) in enumerate(zip(prevalences, case_counts), 1):
            stream.write(
                f"binary\tBT{trait}\t{target:.9f}\t{count}\t{count / samples:.9f}\n"
            )
        for trait, (target, count) in enumerate(zip(event_fractions, event_counts), 1):
            stream.write(
                f"survival\tTIME{trait}/EVENT{trait}\t{target:.9f}\t{count}\t"
                f"{count / samples:.9f}\n"
            )


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.log_hazard_ratio):
        raise SystemExit("--log-hazard-ratio must be finite")
    trait_count = max(len(args.binary_prevalences), len(args.survival_event_fractions))
    sample_ids, all_source = read_source(
        args.source_pheno, args.source_trait_prefix, trait_count
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    binary_source = all_source[:, : len(args.binary_prevalences)]
    binary, case_counts = fixed_count_binary_traits(
        binary_source, args.binary_prevalences
    )
    survival_source = all_source[:, : len(args.survival_event_fractions)]
    times, events, event_counts = survival_traits(
        survival_source,
        args.survival_event_fractions,
        args.log_hazard_ratio,
        np.random.default_rng(args.seed),
    )

    write_binary(output_path(args.output_prefix, ".binary.pheno"), sample_ids, binary)
    write_survival(
        output_path(args.output_prefix, ".survival.pheno"), sample_ids, times, events
    )
    write_summary(
        output_path(args.output_prefix, ".trait-models.tsv"),
        len(sample_ids),
        args.binary_prevalences,
        case_counts,
        args.survival_event_fractions,
        event_counts,
    )
    print(
        f"Wrote {len(args.binary_prevalences)} binary and "
        f"{len(args.survival_event_fractions)} survival traits for "
        f"{len(sample_ids)} samples; seed={args.seed}"
    )


if __name__ == "__main__":
    main()
