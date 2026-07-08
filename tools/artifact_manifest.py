#!/usr/bin/env python3
"""Build a reproducible artifact manifest with file sizes and hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

PATTERNS = ["build/**/*.so", "dist/**/*", "benchmark/**/*.log", "logs/**/*", "*.txt"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(root: Path) -> dict[str, object]:
    if not root.is_dir():
        raise NotADirectoryError(f"artifact root is not a directory: {root}")

    seen: set[str] = set()
    artifacts: list[dict[str, object]] = []
    for pattern in PATTERNS:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            artifacts.append(
                {"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    return {"root": str(root), "count": len(artifacts), "artifacts": artifacts}


def self_test() -> None:
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        sample = root / "artifact_manifest_sample.txt"
        sample.write_text("maca artifact\n", encoding="utf-8")
        data = collect(root)
    if not any(item["path"] == sample.name for item in data["artifacts"]):
        raise RuntimeError(f"self-test failed: {data}")
    print(json.dumps({"ok": True, "count": data["count"]}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    print(json.dumps(collect(Path(args.root)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
