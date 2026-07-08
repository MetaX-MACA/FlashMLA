#!/usr/bin/env python3
"""Compare baseline and current performance JSON and fail on regressions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRIC = "throughput"
TOLERANCE = 0.05


def load(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {str(item["name"]): float(item[METRIC]) for item in data}
    return {
        str(k): float(v[METRIC] if isinstance(v, dict) else v)
        for k, v in data.items()
    }


def compare(baseline: dict[str, float], current: dict[str, float]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    failed = False
    for name, old in sorted(baseline.items()):
        if name not in current:
            rows.append({"name": name, "status": "missing-current"})
            failed = True
            continue
        new = current[name]
        ratio = (new - old) / old if old else 0.0
        status = "regression" if ratio < -TOLERANCE else "ok"
        failed = failed or status != "ok"
        rows.append(
            {
                "name": name,
                "baseline": old,
                "current": new,
                "delta_ratio": ratio,
                "status": status,
            }
        )
    return {"ok": not failed, "metric": METRIC, "rows": rows}


def self_test() -> None:
    data_ok = compare({"case": 100.0}, {"case": 99.0})
    data_regression = compare({"case": 100.0}, {"case": 90.0})
    data_missing = compare({"case": 100.0}, {})
    if not data_ok["ok"] or data_regression["ok"] or data_missing["ok"]:
        raise RuntimeError(
            {
                "ok_case": data_ok,
                "regression_case": data_regression,
                "missing_case": data_missing,
            }
        )
    print(json.dumps({"ok": True, "rows": len(data_ok["rows"])}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", nargs="?")
    parser.add_argument("current", nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.baseline or not args.current:
        parser.error("baseline and current are required unless --self-test is used")
    result = compare(load(Path(args.baseline)), load(Path(args.current)))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
