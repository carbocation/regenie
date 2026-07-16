#!/usr/bin/env python3
"""Validate REGENIE release archives and emit a release-set manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import tarfile
from typing import Any


class VerificationError(RuntimeError):
    pass


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_metadata(archive: pathlib.Path) -> dict[str, str]:
    with tarfile.open(archive, "r:gz") as bundle:
        all_members = bundle.getmembers()
        members = [member for member in all_members if member.isfile()]
        member_names = [pathlib.PurePosixPath(member.name) for member in all_members]
        for member_name in member_names:
            if member_name.is_absolute() or ".." in member_name.parts:
                raise VerificationError(
                    f"{archive.name}: unsafe archive member {str(member_name)!r}"
                )
        if len({str(member_name) for member_name in member_names}) != len(member_names):
            raise VerificationError(f"{archive.name}: archive contains duplicate members")
        metadata_members = [
            member
            for member in members
            if pathlib.PurePosixPath(member.name).name == "BUILD-METADATA.txt"
        ]
        if len(metadata_members) != 1:
            raise VerificationError(
                f"{archive.name}: expected one BUILD-METADATA.txt, found "
                f"{len(metadata_members)}"
            )
        roots = {
            pathlib.PurePosixPath(member.name).parts[0]
            for member in all_members
            if pathlib.PurePosixPath(member.name).parts
        }
        if len(roots) != 1:
            raise VerificationError(
                f"{archive.name}: archive must contain exactly one top-level directory"
            )
        root = next(iter(roots))
        binary_name = f"{root}/bin/regenie"
        binary_members = [
            member for member in members if member.name.rstrip("/") == binary_name
        ]
        if len(binary_members) != 1 or not (binary_members[0].mode & 0o111):
            raise VerificationError(
                f"{archive.name}: expected one executable {binary_name}"
            )
        metadata_stream = bundle.extractfile(metadata_members[0])
        if metadata_stream is None:
            raise VerificationError(f"{archive.name}: could not read build metadata")
        metadata_text = metadata_stream.read(1024 * 1024).decode("utf-8")

    metadata: dict[str, str] = {}
    for line_number, raw_line in enumerate(metadata_text.splitlines(), start=1):
        if not raw_line or raw_line.startswith("#"):
            continue
        if "=" not in raw_line:
            raise VerificationError(
                f"{archive.name}: malformed metadata line {line_number}"
            )
        key, value = raw_line.split("=", 1)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise VerificationError(
                f"{archive.name}: invalid metadata key {key!r}"
            )
        if key in metadata:
            raise VerificationError(f"{archive.name}: duplicate metadata key {key}")
        metadata[key] = value

    for required_key in ("REGENIE_VERSION", "GIT_COMMIT", "BUILD_KIND"):
        if not metadata.get(required_key):
            raise VerificationError(
                f"{archive.name}: missing required metadata {required_key}"
            )
    return metadata


def verify_checksum(archive: pathlib.Path) -> str:
    checksum_path = pathlib.Path(f"{archive}.sha256")
    if not checksum_path.is_file():
        raise VerificationError(f"{archive.name}: missing {checksum_path.name}")
    fields = checksum_path.read_text(encoding="utf-8").split()
    if len(fields) < 2:
        raise VerificationError(f"{checksum_path.name}: malformed checksum file")
    expected, recorded_name = fields[0].lower(), fields[1].lstrip("*")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise VerificationError(f"{checksum_path.name}: invalid SHA-256 value")
    if pathlib.Path(recorded_name).name != archive.name:
        raise VerificationError(
            f"{checksum_path.name}: records {recorded_name!r}, expected {archive.name!r}"
        )
    actual = sha256_file(archive)
    if actual != expected:
        raise VerificationError(
            f"{archive.name}: SHA-256 mismatch, expected {expected}, received {actual}"
        )
    return actual


def resolve_archives(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    archives: list[pathlib.Path] = []
    for path in paths:
        if path.is_dir():
            archives.extend(sorted(path.glob("*.tar.gz")))
        elif path.is_file() and path.name.endswith(".tar.gz"):
            archives.append(path)
        else:
            raise VerificationError(f"release input does not exist or is not .tar.gz: {path}")
    unique = sorted({archive.resolve() for archive in archives})
    if not unique:
        raise VerificationError("no .tar.gz release assets were found")
    return unique


def make_asset_record(archive: pathlib.Path, checksum: str,
                      metadata: dict[str, str]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "filename": archive.name,
        "sha256": checksum,
        "size_bytes": archive.stat().st_size,
        "build_kind": metadata["BUILD_KIND"],
        "regenie_version": metadata["REGENIE_VERSION"],
        "git_commit": metadata["GIT_COMMIT"],
    }
    for key in (
        "CPU_ARCHITECTURE",
        "CPU_TARGET",
        "CUDA_ARCHITECTURES",
        "CUDA_PROFILE",
        "CUDA_TOOLKIT_VERSION",
        "GLIBC_REQUIRED",
        "MACOSX_DEPLOYMENT_TARGET",
        "DYNAMIC_NEEDED",
    ):
        if key in metadata:
            record[key.lower()] = metadata[key]
    return record


def write_json(path: pathlib.Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = pathlib.Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify checksums, metadata, and commit consistency for release assets."
    )
    parser.add_argument("paths", nargs="+", type=pathlib.Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--require-kind", action="append", default=[])
    parser.add_argument("--require-verified-dependencies", action="store_true")
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--sha256sums", type=pathlib.Path)
    args = parser.parse_args()

    try:
        archives = resolve_archives(args.paths)
        records: list[dict[str, Any]] = []
        commits: set[str] = set()
        versions: set[str] = set()
        kinds: set[str] = set()
        checksum_lines: list[str] = []

        for archive in archives:
            checksum = verify_checksum(archive)
            metadata = parse_metadata(archive)
            if args.require_verified_dependencies:
                provenance = {
                    key: value
                    for key, value in metadata.items()
                    if key.endswith("_PROVENANCE")
                }
                if "BGEN_PROVENANCE" not in provenance:
                    raise VerificationError(
                        f"{archive.name}: missing BGEN_PROVENANCE"
                    )
                for key, value in provenance.items():
                    if not value.startswith("verified-"):
                        raise VerificationError(
                            f"{archive.name}: {key} is not verified ({value})"
                        )
            commits.add(metadata["GIT_COMMIT"])
            versions.add(metadata["REGENIE_VERSION"])
            kinds.add(metadata["BUILD_KIND"])
            records.append(make_asset_record(archive, checksum, metadata))
            checksum_lines.append(f"{checksum}  {archive.name}")
            print(
                "RELEASE_ASSET_VERIFY "
                f"asset={archive.name} kind={metadata['BUILD_KIND']} "
                f"commit={metadata['GIT_COMMIT']} sha256={checksum}"
            )

        if len(commits) != 1:
            raise VerificationError(
                "release assets were built from different commits: " + ", ".join(sorted(commits))
            )
        if len(versions) != 1:
            raise VerificationError(
                "release assets report different REGENIE versions: "
                + ", ".join(sorted(versions))
            )
        commit = next(iter(commits))
        if args.expected_commit and commit != args.expected_commit:
            raise VerificationError(
                f"release commit is {commit}, expected {args.expected_commit}"
            )
        missing_kinds = sorted(set(args.require_kind) - kinds)
        if missing_kinds:
            raise VerificationError(
                "release set is missing required build kinds: " + ", ".join(missing_kinds)
            )

        document = {
            "format_version": 1,
            "git_commit": commit,
            "regenie_version": next(iter(versions)),
            "assets": sorted(records, key=lambda record: record["filename"]),
        }
        if args.manifest:
            write_json(args.manifest, document)
        if args.sha256sums:
            args.sha256sums.parent.mkdir(parents=True, exist_ok=True)
            args.sha256sums.write_text(
                "\n".join(sorted(checksum_lines)) + "\n", encoding="utf-8"
            )
        print(
            "RELEASE_ASSET_VERIFY "
            f"status=PASS assets={len(records)} commit={commit} "
            f"kinds={','.join(sorted(kinds))}"
        )
        return 0
    except (OSError, tarfile.TarError, UnicodeError, VerificationError) as error:
        print(f"RELEASE_ASSET_VERIFY status=FAIL error={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
