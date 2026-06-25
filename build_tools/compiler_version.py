import re
import subprocess
from pathlib import Path
from typing import Callable, List, Tuple, Union

from packaging.version import Version


def parse_cuda_release_version(output: str) -> Version:
    release_match = re.search(r"release\s+(\d+\.\d+)", output)
    if release_match:
        return Version(release_match.group(1))

    version_match = re.search(r"\bV(\d+\.\d+)(?:\.\d+)?\b", output)
    if version_match:
        return Version(version_match.group(1))

    raise ValueError(f"Cannot parse compiler release version from output: {output!r}")


def _compiler_candidates(cuda_dir: Union[str, Path]) -> List[Path]:
    bin_dir = Path(cuda_dir) / "bin"
    return [bin_dir / "cucc", bin_dir / "nvcc"]


def get_cuda_bare_metal_version(
    cuda_dir: Union[str, Path],
    check_output: Callable[..., str] = subprocess.check_output,
) -> Tuple[str, Version]:
    errors: List[str] = []
    for compiler in _compiler_candidates(cuda_dir):
        if not compiler.is_file():
            errors.append(f"{compiler} does not exist")
            continue
        try:
            raw_output = check_output(
                [str(compiler), "-V"],
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            return raw_output, parse_cuda_release_version(raw_output)
        except Exception as err:
            errors.append(f"{compiler}: {err}")

    details = "\n".join(f"- {error}" for error in errors)
    raise RuntimeError(
        f"Cannot detect CUDA-compatible compiler version under {cuda_dir}.\n{details}"
    )
