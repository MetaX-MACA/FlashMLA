import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from estimate_flash_mla_memory import estimate_bytes  # noqa: E402


def test_memory_estimator_counts_k_cache_blocks():
    args = Namespace(
        dtype="bf16",
        batch_size=2,
        s_q=1,
        mean_sk=17,
        h_q=4,
        h_kv=1,
        d=8,
        dv=4,
        block_size=16,
    )

    estimates = estimate_bytes(args)

    assert estimates["k_cache"] == 2 * 16 * 16 * 1 * 8 * 2
    assert estimates["out"] == 2 * 1 * 4 * 4 * 2
    assert estimates["total"] >= estimates["k_cache"]


if __name__ == "__main__":
    test_memory_estimator_counts_k_cache_blocks()
