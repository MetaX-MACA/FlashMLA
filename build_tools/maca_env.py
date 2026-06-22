import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping


@dataclass(frozen=True)
class MacaBuildEnv:
    maca_path: Path | None
    cuda_path: Path | None
    maca_clang_path: Path | None
    maca_lib_path: Path | None
    cucc_path: Path | None


def _path_from_env(env: Mapping[str, str], name: str) -> Path | None:
    value = env.get(name)
    cleaned = value.strip() if value else None
    return Path(cleaned).expanduser() if cleaned else None


def _candidate_file(path: Path | None, relative: str) -> Path | None:
    return path / relative if path is not None else None


def _find_executable(
    name: str,
    candidates: list[Path | None],
    which: Callable[[str], str | None] = shutil.which,
) -> Path | None:
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    resolved = which(name)
    return Path(resolved) if resolved else None


def resolve_maca_build_env(
    env: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> MacaBuildEnv:
    env = os.environ if env is None else env
    maca_path = _path_from_env(env, "MACA_PATH")
    cuda_path = (
        _path_from_env(env, "CUDA_HOME")
        or _path_from_env(env, "CUDA_PATH")
        or (maca_path / "tools" / "cu-bridge" if maca_path else None)
    )
    maca_clang_path = _path_from_env(env, "MACA_CLANG_PATH") or (
        maca_path / "mxgpu_llvm" / "bin" if maca_path else None
    )
    maca_lib_path = maca_path / "lib" if maca_path else None
    cucc_path = _find_executable(
        "cucc",
        [
            _candidate_file(cuda_path, "bin/cucc"),
            _candidate_file(maca_clang_path, "cucc"),
        ],
        which=which,
    )
    return MacaBuildEnv(
        maca_path=maca_path,
        cuda_path=cuda_path,
        maca_clang_path=maca_clang_path,
        maca_lib_path=maca_lib_path,
        cucc_path=cucc_path,
    )


def validate_maca_build_env(build_env: MacaBuildEnv) -> list[str]:
    errors: list[str] = []
    required_dirs = {
        "MACA_PATH": build_env.maca_path,
        "CUDA_HOME/CUDA_PATH": build_env.cuda_path,
        "MACA_CLANG_PATH": build_env.maca_clang_path,
        "MACA library directory": build_env.maca_lib_path,
    }
    for label, path in required_dirs.items():
        if path is None:
            errors.append(f"{label} is not configured")
        elif not path.is_dir():
            errors.append(f"{label} does not exist or is not a directory: {path}")

    if build_env.cucc_path is None:
        errors.append("cucc compiler was not found in CUDA_HOME/bin, MACA_CLANG_PATH, or PATH")
    elif not build_env.cucc_path.is_file():
        errors.append(f"cucc compiler path is not a file: {build_env.cucc_path}")

    return errors


def format_maca_build_env_errors(errors: list[str]) -> str:
    hint = (
        "Set MACA_PATH to the MACA toolkit root, CUDA_HOME or CUDA_PATH to "
        "$MACA_PATH/tools/cu-bridge, and MACA_CLANG_PATH to "
        "$MACA_PATH/mxgpu_llvm/bin before building FlashMLA from source."
    )
    return "Invalid MACA build environment:\n- " + "\n- ".join(errors) + "\n" + hint
