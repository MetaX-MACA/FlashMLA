#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _path_status(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {"path": None, "exists": False, "is_dir": False}
    resolved = Path(path).expanduser()
    return {
        "path": str(resolved),
        "exists": resolved.exists(),
        "is_dir": resolved.is_dir(),
    }


def _command_output(command: List[str]) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:
        return {"command": command, "error": f"{type(exc).__name__}: {exc}"}


def collect_report(env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    env = os.environ if env is None else env
    maca_path = env.get("MACA_PATH")
    cuda_path = env.get("CUDA_HOME") or env.get("CUDA_PATH")
    if not cuda_path and maca_path:
        cuda_path = str(Path(maca_path) / "tools" / "cu-bridge")
    clang_path = env.get("MACA_CLANG_PATH")
    if not clang_path and maca_path:
        clang_path = str(Path(maca_path) / "mxgpu_llvm" / "bin")

    search_path = env.get("PATH", "")
    cucc = shutil.which("cucc", path=search_path)
    nvcc = shutil.which("nvcc", path=search_path)
    report: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "environment": {
            "MACA_PATH": _path_status(maca_path),
            "CUDA_HOME_OR_PATH": _path_status(cuda_path),
            "MACA_CLANG_PATH": _path_status(clang_path),
            "LD_LIBRARY_PATH_SET": bool(env.get("LD_LIBRARY_PATH")),
        },
        "executables": {
            "cucc": cucc,
            "nvcc": nvcc,
        },
        "libraries": {},
        "torch": {},
        "checks": [],
    }

    if maca_path:
        maca_root = Path(maca_path)
        report["libraries"] = {
            "mcruntime": _path_status(str(maca_root / "lib" / "libmcruntime.so")),
            "mcblas": _path_status(str(maca_root / "lib" / "libmcblas.so")),
        }

    if cucc:
        report["cucc_version"] = _command_output([cucc, "-V"])
    elif nvcc:
        report["nvcc_version"] = _command_output([nvcc, "-V"])

    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_version": getattr(torch.version, "cuda", None),
            "maca_version": getattr(torch.version, "maca", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            report["torch"]["device_name_0"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        report["torch"] = {"error": f"{type(exc).__name__}: {exc}"}

    required = [
        ("MACA_PATH", report["environment"]["MACA_PATH"]["is_dir"]),
        ("CUDA_HOME_OR_PATH", report["environment"]["CUDA_HOME_OR_PATH"]["is_dir"]),
        ("MACA_CLANG_PATH", report["environment"]["MACA_CLANG_PATH"]["is_dir"]),
        ("cucc_or_nvcc", bool(cucc or nvcc)),
    ]
    for name, ok in required:
        report["checks"].append({"name": name, "ok": ok})
    report["ok"] = all(item["ok"] for item in report["checks"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect FlashMLA MACA environment diagnostics.")
    parser.add_argument("--pretty", action="store_true", help="Print indented JSON.")
    args = parser.parse_args()

    report = collect_report()
    print(json.dumps(report, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
