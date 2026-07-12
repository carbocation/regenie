#!/usr/bin/env python3
"""Compare whitespace-delimited text files, tolerating floating-point drift."""

import argparse
import itertools
import math
from pathlib import Path


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
    return parser.parse_args()


def as_float(token):
    try:
        return float(token)
    except ValueError:
        return None


def significant_digit_quantum(value, digits):
    if not math.isfinite(value) or value == 0:
        return 0.0
    exponent = math.floor(math.log10(abs(value)))
    return 10.0 ** (exponent - digits + 1)


def main():
    args = parse_args()
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
        f"maximum_absolute_error={maximum_error}{serialization}"
    )


if __name__ == "__main__":
    main()
