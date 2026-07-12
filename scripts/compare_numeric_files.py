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

    candidates = np.flatnonzero(errors > tolerances)
    for candidate in candidates:
        tolerance = tolerances[candidate]
        if args.output_significant_digits:
            tolerance = max(
                tolerance,
                significant_digit_quantum(
                    expected_values[candidate], args.output_significant_digits
                )
                * (1.0 + 1e-9),
                significant_digit_quantum(
                    actual_values[candidate], args.output_significant_digits
                )
                * (1.0 + 1e-9),
            )
        if errors[candidate] > tolerance:
            column = int(candidate) + 1
            raise SystemExit(
                f"numeric difference at line {line_number}, column {column}: "
                f"{expected_values[candidate]} != {actual_values[candidate]} "
                f"(absolute error {errors[candidate]}, tolerance {tolerance})"
            )

    maximum_error = float(errors.max()) if errors.size else 0.0
    return int(expected_values.size), maximum_error


def main():
    args = parse_args()
    if args.engine == "numpy" and np is None:
        raise SystemExit("NumPy comparison engine requested but NumPy is not installed")
    use_numpy = args.engine != "python" and np is not None
    maximum_error = 0.0
    numeric_values = 0
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
                    line_values, line_error = comparison
                    numeric_values += line_values
                    maximum_error = max(maximum_error, line_error)
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
                maximum_error = max(maximum_error, error)
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
                    raise SystemExit(
                        f"numeric difference at line {line_number}, column {column}: "
                        f"{expected_value} != {actual_value} "
                        f"(absolute error {error}, tolerance {tolerance})"
                    )

    serialization = (
        f" output_significant_digits={args.output_significant_digits}"
        if args.output_significant_digits
        else ""
    )
    print(
        f"NUMERIC_FILE_COMPARE status=PASS values={numeric_values} "
        f"maximum_absolute_error={maximum_error}{serialization} "
        f"engine={'numpy' if use_numpy else 'python'}"
    )


if __name__ == "__main__":
    main()
