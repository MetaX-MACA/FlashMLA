#!/usr/bin/env python3
"""Audit MACA-related path variables for missing, duplicate, and stale entries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ENV_VARS = ["MACA_HOME", "LD_LIBRARY_PATH", "PYTHONPATH"]


def split_paths(value: str) -> list[str]:
    parts: list[str] = []
    for chunk in value.replace(";", os.pathsep).split(os.pathsep):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def normalize_path(raw: str) -> str:
    return os.path.normpath(os.path.abspath(os.path.expanduser(raw)))


def audit(env: dict[str, str]) -> dict[str, object]:
    findings: list[dict[str, str]] = []
    for name in ENV_VARS:
        seen: set[str] = set()
        for raw in split_paths(env.get(name, "")):
            normalized = normalize_path(raw)
            if normalized in seen:
                findings.append(
                    {
                        "env": name,
                        "path": raw,
                        "severity": "warning",
                        "message": "duplicate path entry",
                    }
                )
            seen.add(normalized)
            if not Path(normalized).exists():
                findings.append(
                    {
                        "env": name,
                        "path": raw,
                        "severity": "info",
                        "message": "path does not exist in this container",
                    }
                )
    return {"finding_count": len(findings), "findings": findings}


def self_test() -> None:
    missing = os.pathsep.join(["/definitely_missing", "/definitely_missing"])
    data = audit({"LD_LIBRARY_PATH": missing})
    if data["finding_count"] < 2:
        raise RuntimeError(f"self-test failed: {data}")
    print(
        json.dumps(
            {"ok": True, "finding_count": data["finding_count"]}, ensure_ascii=False
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    print(json.dumps(audit(dict(os.environ)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
