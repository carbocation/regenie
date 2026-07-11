#!/usr/bin/env python3
"""Compare whitespace-delimited text files, tolerating floating-point drift."""

import argparse
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=1e-9)
    return parser.parse_args()


def as_float(token):
    try:
        return float(token)
    except ValueError:
        return None


def main():
    args = parse_args()
    expected_lines = args.expected.read_text().splitlines()
    actual_lines = args.actual.read_text().splitlines()
    if len(expected_lines) != len(actual_lines):
        raise SystemExit(
            f"line count differs: {args.expected}={len(expected_lines)}, "
            f"{args.actual}={len(actual_lines)}"
        )

    maximum_error = 0.0
    numeric_values = 0
    for line_number, (expected_line, actual_line) in enumerate(
        zip(expected_lines, actual_lines), start=1
    ):
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
            error = abs(expected_value - actual_value)
            maximum_error = max(maximum_error, error)
            if not math.isclose(
                expected_value, actual_value, rel_tol=args.rtol, abs_tol=args.atol
            ):
                raise SystemExit(
                    f"numeric difference at line {line_number}, column {column}: "
                    f"{expected_value} != {actual_value} (absolute error {error})"
                )

    print(
        f"NUMERIC_FILE_COMPARE status=PASS values={numeric_values} "
        f"maximum_absolute_error={maximum_error}"
    )


if __name__ == "__main__":
    main()
