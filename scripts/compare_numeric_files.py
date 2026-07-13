#!/usr/bin/env python3
"""Compare whitespace-delimited text files, tolerating floating-point drift."""

import argparse
import itertools
import math
import re
import warnings
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None


FLOAT_TOKEN = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?|nan)"
FLOAT_TOKEN_PATTERN = re.compile(rf"(?:{FLOAT_TOKEN})", re.IGNORECASE)
NUMERIC_TOKEN_PATTERN = re.compile(
    rf"(?<!\S)({FLOAT_TOKEN})(?!\S)", re.IGNORECASE
)


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite, non-negative number")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--rtol", type=nonnegative_float, default=1e-7)
    parser.add_argument("--atol", type=nonnegative_float, default=1e-9)
    parser.add_argument(
        "--output-significant-digits",
        type=positive_int,
        help=(
            "also tolerate one unit in the final serialized significant digit; "
            "use 6 for REGENIE's default text output"
        ),
    )
    parser.add_argument(
        "--engine",
        choices=("auto", "numpy", "python"),
        default="auto",
        help="comparison engine (default: NumPy when installed, otherwise Python)",
    )
    parser.add_argument(
        "--report-all",
        action="store_true",
        help=(
            "scan the complete files and report aggregate numerical differences "
            "before returning a failing status"
        ),
    )
    return parser.parse_args()


def as_float(token):
    if not FLOAT_TOKEN_PATTERN.fullmatch(token):
        return None
    try:
        return float(token)
    except ValueError:
        return None


def significant_digit_quantum(value, digits):
    if not math.isfinite(value) or value == 0:
        return 0.0
    exponent = math.floor(math.log10(abs(value)))
    return 10.0 ** (exponent - digits + 1)


def begins_with_numeric_token(line):
    start = 0
    while start < len(line) and line[start].isspace():
        start += 1
    end = start
    while end < len(line) and not line[end].isspace():
        end += 1
    return start < end and FLOAT_TOKEN_PATTERN.fullmatch(line[start:end]) is not None


def numpy_values(line):
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            values = np.fromstring(line, dtype=np.float64, sep=" ")
    except ValueError:
        return None
    if caught:
        return None
    return values


def numpy_significant_digit_quantum(values, digits):
    quantums = np.zeros(values.shape, dtype=np.float64)
    nonzero = np.isfinite(values) & (values != 0)
    quantums[nonzero] = np.power(
        10.0,
        np.floor(np.log10(np.abs(values[nonzero]))) - digits + 1,
    )
    return quantums


def compare_numpy_lines(expected_line, actual_line, line_number, args):
    expected_values = numpy_values(expected_line)
    actual_values = numpy_values(actual_line)
    if expected_values is None or actual_values is None:
        return None
    if expected_values.size != actual_values.size:
        raise SystemExit(f"token count differs on line {line_number}")

    expected_finite = np.isfinite(expected_values)
    actual_finite = np.isfinite(actual_values)
    equal_infinities = (
        np.isinf(expected_values)
        & np.isinf(actual_values)
        & (expected_values == actual_values)
    )
    bad_nonfinite = (~expected_finite | ~actual_finite) & ~equal_infinities
    if np.any(bad_nonfinite):
        column = int(np.flatnonzero(bad_nonfinite)[0]) + 1
        raise SystemExit(
            f"non-finite numeric difference at line {line_number}, "
            f"column {column}: {expected_values[column - 1]!r} != "
            f"{actual_values[column - 1]!r}"
        )

    finite = expected_finite & actual_finite
    errors = np.zeros(expected_values.shape, dtype=np.float64)
    errors[finite] = np.abs(expected_values[finite] - actual_values[finite])
    tolerances = np.full(expected_values.shape, args.atol, dtype=np.float64)
    tolerances[finite] = np.maximum(
        tolerances[finite],
        args.rtol
        * np.maximum(
            np.abs(expected_values[finite]), np.abs(actual_values[finite])
        ),
    )

    if args.output_significant_digits:
        serialization_tolerances = np.maximum(
            numpy_significant_digit_quantum(
                expected_values, args.output_significant_digits
            ),
            numpy_significant_digit_quantum(
                actual_values, args.output_significant_digits
            ),
        )
        tolerances = np.maximum(
            tolerances, serialization_tolerances * (1.0 + 1e-9)
        )

    failures = errors > tolerances
    if np.any(failures) and not args.report_all:
        candidate = int(np.flatnonzero(failures)[0])
        column = candidate + 1
        raise SystemExit(
            f"numeric difference at line {line_number}, column {column}: "
            f"{expected_values[candidate]} != {actual_values[candidate]} "
            f"(absolute error {errors[candidate]}, "
            f"tolerance {tolerances[candidate]})"
        )

    denominators = np.maximum(np.abs(expected_values), np.abs(actual_values))
    relative_errors = np.zeros(expected_values.shape, dtype=np.float64)
    np.divide(
        errors,
        denominators,
        out=relative_errors,
        where=finite & (denominators > 0),
    )
    tolerance_ratios = np.zeros(expected_values.shape, dtype=np.float64)
    np.divide(
        errors,
        tolerances,
        out=tolerance_ratios,
        where=tolerances > 0,
    )
    tolerance_ratios[(tolerances == 0) & (errors > 0)] = np.inf

    worst = None
    if np.any(failures):
        failure_indices = np.flatnonzero(failures)
        candidate = int(
            failure_indices[np.argmax(tolerance_ratios[failure_indices])]
        )
        worst = (
            line_number,
            candidate + 1,
            float(expected_values[candidate]),
            float(actual_values[candidate]),
            float(errors[candidate]),
            float(tolerances[candidate]),
            float(relative_errors[candidate]),
            float(tolerance_ratios[candidate]),
        )

    maximum_error = float(errors.max()) if errors.size else 0.0
    maximum = None
    if maximum_error > 0:
        candidate = int(np.argmax(errors))
        maximum = (
            line_number,
            candidate + 1,
            float(expected_values[candidate]),
            float(actual_values[candidate]),
            float(errors[candidate]),
            float(relative_errors[candidate]),
        )
    maximum_relative_error = (
        float(relative_errors.max()) if relative_errors.size else 0.0
    )
    return {
        "values": int(expected_values.size),
        "maximum_error": maximum_error,
        "maximum_relative_error": maximum_relative_error,
        "differing_values": int(np.count_nonzero(errors)),
        "tolerance_failures": int(np.count_nonzero(failures)),
        "sum_squared_error": float(np.dot(errors, errors)),
        "maximum": maximum,
        "worst": worst,
    }


def main():
    args = parse_args()
    if args.engine == "numpy" and np is None:
        raise SystemExit("NumPy comparison engine requested but NumPy is not installed")
    use_numpy = args.engine != "python" and np is not None
    maximum_error = 0.0
    maximum_relative_error = 0.0
    numeric_values = 0
    differing_values = 0
    tolerance_failures = 0
    sum_squared_error = 0.0
    maximum = None
    worst = None
    missing = object()
    with args.expected.open(encoding="utf-8") as expected_file, args.actual.open(
        encoding="utf-8"
    ) as actual_file:
        for line_number, lines in enumerate(
            itertools.zip_longest(expected_file, actual_file, fillvalue=missing),
            start=1,
        ):
            expected_line, actual_line = lines
            if expected_line is missing or actual_line is missing:
                raise SystemExit(f"line count differs at line {line_number}")

            if use_numpy and expected_line == actual_line:
                if begins_with_numeric_token(expected_line):
                    values = numpy_values(expected_line)
                    if values is not None:
                        if np.any(np.isnan(values)):
                            column = int(np.flatnonzero(np.isnan(values))[0]) + 1
                            raise SystemExit(
                                f"non-finite numeric difference at line "
                                f"{line_number}, column {column}: nan != nan"
                            )
                        numeric_values += int(values.size)
                        continue
                numeric_tokens = NUMERIC_TOKEN_PATTERN.findall(expected_line)
                for token in numeric_tokens:
                    if math.isnan(float(token)):
                        raise SystemExit(
                            f"non-finite numeric difference at line {line_number}: "
                            f"{token!r} != {token!r}"
                        )
                numeric_values += len(numeric_tokens)
                continue

            if (
                use_numpy
                and begins_with_numeric_token(expected_line)
                and begins_with_numeric_token(actual_line)
            ):
                comparison = compare_numpy_lines(
                    expected_line, actual_line, line_number, args
                )
                if comparison is not None:
                    numeric_values += comparison["values"]
                    line_maximum = comparison["maximum"]
                    if line_maximum is not None and (
                        maximum is None or line_maximum[4] > maximum[4]
                    ):
                        maximum = line_maximum
                        maximum_error = line_maximum[4]
                    maximum_relative_error = max(
                        maximum_relative_error,
                        comparison["maximum_relative_error"],
                    )
                    differing_values += comparison["differing_values"]
                    tolerance_failures += comparison["tolerance_failures"]
                    sum_squared_error += comparison["sum_squared_error"]
                    line_worst = comparison["worst"]
                    if line_worst is not None and (
                        worst is None or line_worst[7] > worst[7]
                    ):
                        worst = line_worst
                    continue

            expected_tokens = expected_line.split()
            actual_tokens = actual_line.split()
            if len(expected_tokens) != len(actual_tokens):
                raise SystemExit(f"token count differs on line {line_number}")
            for column, (expected_token, actual_token) in enumerate(
                zip(expected_tokens, actual_tokens), start=1
            ):
                expected_value = as_float(expected_token)
                actual_value = as_float(actual_token)
                if expected_value is None or actual_value is None:
                    if expected_token != actual_token:
                        raise SystemExit(
                            f"text differs at line {line_number}, column {column}: "
                            f"{expected_token!r} != {actual_token!r}"
                        )
                    continue
                numeric_values += 1
                if not math.isfinite(expected_value) or not math.isfinite(
                    actual_value
                ):
                    if (
                        expected_value == actual_value
                        and not math.isnan(expected_value)
                        and not math.isnan(actual_value)
                    ):
                        continue
                    raise SystemExit(
                        f"non-finite numeric difference at line {line_number}, "
                        f"column {column}: {expected_token!r} != {actual_token!r}"
                    )
                error = abs(expected_value - actual_value)
                if error > 0:
                    differing_values += 1
                sum_squared_error += error * error
                denominator = max(abs(expected_value), abs(actual_value))
                relative_error_value = error / denominator if denominator else 0.0
                maximum_relative_error = max(
                    maximum_relative_error, relative_error_value
                )
                if error > 0 and (maximum is None or error > maximum[4]):
                    maximum = (
                        line_number,
                        column,
                        expected_value,
                        actual_value,
                        error,
                        relative_error_value,
                    )
                    maximum_error = error
                tolerance = max(
                    args.atol,
                    args.rtol * max(abs(expected_value), abs(actual_value)),
                )
                if args.output_significant_digits:
                    tolerance = max(
                        tolerance,
                        significant_digit_quantum(
                            expected_value, args.output_significant_digits
                        )
                        * (1.0 + 1e-9),
                        significant_digit_quantum(
                            actual_value, args.output_significant_digits
                        )
                        * (1.0 + 1e-9),
                    )
                if error > tolerance:
                    tolerance_failures += 1
                    tolerance_ratio = error / tolerance if tolerance else math.inf
                    difference = (
                        line_number,
                        column,
                        expected_value,
                        actual_value,
                        error,
                        tolerance,
                        relative_error_value,
                        tolerance_ratio,
                    )
                    if worst is None or tolerance_ratio > worst[7]:
                        worst = difference
                    if not args.report_all:
                        raise SystemExit(
                            f"numeric difference at line {line_number}, "
                            f"column {column}: {expected_value} != {actual_value} "
                            f"(absolute error {error}, tolerance {tolerance})"
                        )

    serialization = (
        f" output_significant_digits={args.output_significant_digits}"
        if args.output_significant_digits
        else ""
    )
    engine = "numpy" if use_numpy else "python"
    if args.report_all:
        rms_error = math.sqrt(sum_squared_error / numeric_values) if numeric_values else 0
        status = "FAIL" if tolerance_failures else "PASS"
        summary = (
            f"NUMERIC_FILE_COMPARE status={status} values={numeric_values} "
            f"differing_values={differing_values} "
            f"tolerance_failures={tolerance_failures} "
            f"maximum_absolute_error={maximum_error} "
            f"maximum_relative_error={maximum_relative_error} "
            f"root_mean_squared_error={rms_error}{serialization} "
            f"engine={engine}"
        )
        if maximum is not None:
            summary += (
                f" maximum_absolute_line={maximum[0]} "
                f"maximum_absolute_column={maximum[1]} "
                f"maximum_absolute_expected={maximum[2]} "
                f"maximum_absolute_actual={maximum[3]} "
                f"maximum_absolute_relative_error={maximum[5]}"
            )
        if worst is not None:
            summary += (
                f" worst_line={worst[0]} worst_column={worst[1]} "
                f"worst_expected={worst[2]} worst_actual={worst[3]} "
                f"worst_absolute_error={worst[4]} "
                f"worst_tolerance={worst[5]} "
                f"worst_relative_error={worst[6]} "
                f"worst_tolerance_ratio={worst[7]}"
            )
        print(summary)
        if tolerance_failures:
            raise SystemExit(1)
    else:
        print(
            f"NUMERIC_FILE_COMPARE status=PASS values={numeric_values} "
            f"maximum_absolute_error={maximum_error}{serialization} "
            f"engine={engine}"
        )


if __name__ == "__main__":
    main()
