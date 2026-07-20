#!/usr/bin/env python3
"""Summarize a run_profiled.sh result directory without external packages."""

from __future__ import annotations

import csv
import math
import re
import statistics
import sys
from pathlib import Path


NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    with path.open() as stream:
        next(stream, None)
        for line in stream:
            key, separator, value = line.rstrip("\n").partition("\t")
            if separator:
                values[key] = value
    return values


def number(value: str | None) -> float | None:
    if value is None:
        return None
    match = NUMBER_RE.search(value)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def profile_records(console_path: Path) -> list[tuple[str, dict[str, str]]]:
    records: list[tuple[str, dict[str, str]]] = []
    if not console_path.exists():
        return records
    with console_path.open(errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line.startswith(("STEP1_PROFILE", "STEP2_PROFILE")):
                continue
            fields = line.split()
            record = {}
            for field in fields[1:]:
                key, separator, value = field.partition("=")
                if separator:
                    record[key] = value
            records.append((fields[0], record))
    return records


def write_profile_kv(run_dir: Path, records: list[tuple[str, dict[str, str]]]) -> None:
    with (run_dir / "profile_kv.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["record_index", "record_type", "key", "value"])
        for index, (record_type, record) in enumerate(records, 1):
            for key, value in record.items():
                writer.writerow([index, record_type, key, value])


def phase_boundaries(path: Path) -> list[tuple[float, str]]:
    boundaries: list[tuple[float, str]] = []
    saw_level0 = False
    if not path.exists():
        return boundaries
    with path.open(errors="replace") as stream:
        for line in stream:
            epoch_text, separator, message = line.partition("\t")
            if not separator:
                continue
            try:
                epoch = float(epoch_text)
            except ValueError:
                continue
            if not saw_level0 and re.search(r"(?:^|\s)Chromosome\s+\S+", message):
                boundaries.append((epoch, "level0"))
                saw_level0 = True
            elif " Level 1 ridge" in message or message.lstrip().startswith(
                "Level 1 ridge"
            ):
                boundaries.append((epoch, "level1"))
            elif message.startswith(("STEP1_PROFILE_FINAL", "STEP2_PROFILE_FINAL")):
                boundaries.append((epoch, "finalize"))
    return boundaries


def load_gpu_samples(path: Path) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    if not path.exists():
        return samples
    with path.open(errors="replace") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            parsed: dict[str, float] = {}
            for key in (
                "epoch_s",
                "gpu_util_pct",
                "memory_util_pct",
                "memory_used_mib",
                "memory_total_mib",
                "power_w",
                "power_limit_w",
                "sm_clock_mhz",
                "memory_clock_mhz",
                "temperature_c",
            ):
                parsed_value = number(row.get(key))
                if parsed_value is None:
                    break
                parsed[key] = parsed_value
            else:
                samples.append(parsed)
    return samples


def vmstat_summary_rows(path: Path) -> list[tuple[str, str, str]]:
    samples: list[tuple[float, float, float, float, float, float, float]] = []
    if not path.exists():
        return []
    with path.open(errors="replace") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) < 18:
                continue
            try:
                run_queue = float(fields[0])
                blocked = float(fields[1])
                user = float(fields[12])
                system = float(fields[13])
                idle = float(fields[14])
                io_wait = float(fields[15])
                stolen = float(fields[16])
            except ValueError:
                continue
            samples.append((run_queue, blocked, user, system, idle, io_wait, stolen))
    # vmstat's first numeric row is a since-boot average, unlike subsequent
    # interval samples, so it is not part of the measured command.
    if len(samples) > 1:
        samples = samples[1:]
    if not samples:
        return []

    run_queue = [sample[0] for sample in samples]
    blocked = [sample[1] for sample in samples]
    user = [sample[2] for sample in samples]
    system = [sample[3] for sample in samples]
    idle = [sample[4] for sample in samples]
    io_wait = [sample[5] for sample in samples]
    stolen = [sample[6] for sample in samples]
    busy = [100.0 - sample[4] - sample[5] for sample in samples]
    return [
        ("host", "vmstat_samples", str(len(samples))),
        ("host", "cpu_busy_mean_pct", f"{statistics.fmean(busy):.3f}"),
        ("host", "cpu_busy_p90_pct", f"{percentile(busy, 0.90):.3f}"),
        ("host", "cpu_user_mean_pct", f"{statistics.fmean(user):.3f}"),
        ("host", "cpu_system_mean_pct", f"{statistics.fmean(system):.3f}"),
        ("host", "cpu_iowait_mean_pct", f"{statistics.fmean(io_wait):.3f}"),
        ("host", "cpu_stolen_mean_pct", f"{statistics.fmean(stolen):.3f}"),
        ("host", "run_queue_mean", f"{statistics.fmean(run_queue):.3f}"),
        ("host", "blocked_processes_mean", f"{statistics.fmean(blocked):.3f}"),
    ]


def phase_for(epoch: float, boundaries: list[tuple[float, str]]) -> str:
    phase = "setup"
    for boundary_epoch, boundary_phase in boundaries:
        if epoch < boundary_epoch:
            break
        phase = boundary_phase
    return phase


def gpu_summary_rows(
    samples: list[dict[str, float]], boundaries: list[tuple[float, str]]
) -> list[tuple[str, str, str]]:
    if not samples:
        return []
    groups: dict[str, list[dict[str, float]]] = {"all": samples}
    for sample in samples:
        groups.setdefault(phase_for(sample["epoch_s"], boundaries), []).append(sample)

    rows: list[tuple[str, str, str]] = []
    for phase, phase_samples in groups.items():
        util = [sample["gpu_util_pct"] for sample in phase_samples]
        power = [sample["power_w"] for sample in phase_samples]
        memory = [sample["memory_used_mib"] for sample in phase_samples]
        clocks = [sample["sm_clock_mhz"] for sample in phase_samples]
        power_limits = [sample["power_limit_w"] for sample in phase_samples]
        rows.extend(
            [
                (phase, "samples", str(len(phase_samples))),
                (phase, "gpu_util_mean_pct", f"{statistics.fmean(util):.3f}"),
                (phase, "gpu_util_median_pct", f"{statistics.median(util):.3f}"),
                (phase, "gpu_util_p10_pct", f"{percentile(util, 0.10):.3f}"),
                (phase, "gpu_util_p90_pct", f"{percentile(util, 0.90):.3f}"),
                (
                    phase,
                    "gpu_util_below_50_pct",
                    f"{100 * sum(x < 50 for x in util) / len(util):.3f}",
                ),
                (
                    phase,
                    "gpu_util_at_least_90_pct",
                    f"{100 * sum(x >= 90 for x in util) / len(util):.3f}",
                ),
                (phase, "power_mean_w", f"{statistics.fmean(power):.3f}"),
                (phase, "power_p90_w", f"{percentile(power, 0.90):.3f}"),
                (phase, "power_max_w", f"{max(power):.3f}"),
                (phase, "power_limit_mean_w", f"{statistics.fmean(power_limits):.3f}"),
                (
                    phase,
                    "power_mean_of_limit_pct",
                    f"{100 * statistics.fmean(power) / statistics.fmean(power_limits):.3f}",
                ),
                (phase, "memory_peak_mib", f"{max(memory):.3f}"),
                (phase, "sm_clock_mean_mhz", f"{statistics.fmean(clocks):.3f}"),
            ]
        )
    if len(samples) > 1:
        energy_wh = 0.0
        for left, right in zip(samples, samples[1:]):
            interval = max(0.0, right["epoch_s"] - left["epoch_s"])
            energy_wh += interval * (left["power_w"] + right["power_w"]) / 7200.0
        rows.append(("all", "gpu_energy_wh", f"{energy_wh:.3f}"))
    return rows


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_profile.py RUN_DIR")
    run_dir = Path(sys.argv[1])
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    metadata = read_key_values(run_dir / "metadata.tsv")
    resources = read_key_values(run_dir / "resource.tsv")
    records = profile_records(run_dir / "console.log")
    write_profile_kv(run_dir, records)
    boundaries = phase_boundaries(run_dir / "console.timestamped.log")
    host_rows = vmstat_summary_rows(run_dir / "vmstat.log")
    gpu_rows = gpu_summary_rows(load_gpu_samples(run_dir / "gpu.csv"), boundaries)

    rows: list[tuple[str, str, str]] = []
    rows.extend(("run", key, value) for key, value in metadata.items())
    rows.extend(("resource", key, value) for key, value in resources.items())
    rows.append(("profile", "structured_record_count", str(len(records))))
    rows.extend(host_rows)
    rows.extend(gpu_rows)

    with (run_dir / "summary.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["scope", "metric", "value"])
        writer.writerows(rows)

    print(f"BENCHMARK_SUMMARY={run_dir / 'summary.tsv'}")
    for scope, metric, value in rows:
        if scope in {"resource", "all", "level0", "level1"} and metric in {
            "wall_seconds",
            "max_rss_kb",
            "gpu_util_mean_pct",
            "gpu_util_median_pct",
            "power_mean_w",
            "power_mean_of_limit_pct",
            "memory_peak_mib",
        }:
            print(f"BENCHMARK_METRIC scope={scope} metric={metric} value={value}")


if __name__ == "__main__":
    main()
