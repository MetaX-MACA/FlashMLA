#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path
import sys

import torch
import triton


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_mla import flash_mla_with_kvcache, get_mla_metadata  # noqa: E402
from tests.test_flash_mla import scaled_dot_product_attention  # noqa: E402


def _dtype(value: str) -> torch.dtype:
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    raise argparse.ArgumentTypeError("dtype must be bf16 or fp16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small FlashMLA smoke test and report reference deltas."
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device to run on.")
    parser.add_argument("--dtype", type=_dtype, default=torch.bfloat16, help="bf16 or fp16.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--s-q", type=int, default=1)
    parser.add_argument("--mean-sk", type=int, default=4096)
    parser.add_argument("--h-q", type=int, default=16)
    parser.add_argument("--h-kv", type=int, default=1)
    parser.add_argument("--d", type=int, default=576)
    parser.add_argument("--dv", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--non-causal", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the final summary as JSON.")
    parser.add_argument("--max-out-cos-diff", type=float, default=None)
    parser.add_argument("--max-lse-cos-diff", type=float, default=None)
    return parser.parse_args()


def _metric_dict(x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    x = x.double()
    y = y.double()
    rmse = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    return {
        "rmse": rmse,
        "cos_diff": cos_diff,
        "amax_diff": amax_diff,
    }


def _run_flash_mla_case(args: argparse.Namespace) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
    cache_seqlens = torch.full((args.batch_size,), args.mean_sk, dtype=torch.int32)
    if args.varlen:
        for i in range(args.batch_size):
            cache_seqlens[i] = max(random.normalvariate(args.mean_sk, args.mean_sk / 2), args.s_q)
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256

    q = torch.randn(args.batch_size, args.s_q, args.h_q, args.d)
    block_table = torch.arange(
        args.batch_size * max_seqlen_pad // args.block_size, dtype=torch.int32
    ).view(args.batch_size, max_seqlen_pad // args.block_size)
    blocked_k = torch.randn(block_table.numel(), args.block_size, args.h_kv, args.d)
    for i in range(args.batch_size):
        blocked_k.view(args.batch_size, max_seqlen_pad, args.h_kv, args.d)[
            i, cache_seqlens[i].item() :
        ] = float("nan")
    blocked_v = blocked_k[..., : args.dv]

    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens, args.s_q * args.h_q // args.h_kv, args.h_kv
    )
    out_flash, lse_flash = flash_mla_with_kvcache(
        q,
        blocked_k,
        block_table,
        cache_seqlens,
        args.dv,
        tile_scheduler_metadata,
        num_splits,
        causal=not args.non_causal,
    )

    out_ref = torch.empty(args.batch_size, args.s_q, args.h_q, args.dv, dtype=torch.float32)
    lse_ref = torch.empty(args.batch_size, args.h_q, args.s_q, dtype=torch.float32)
    blocked_k_view = blocked_k.view(-1, args.h_kv, args.d)
    blocked_v_view = blocked_v.view(-1, args.h_kv, args.dv)
    for i in range(args.batch_size):
        begin = i * max_seqlen_pad
        end = begin + cache_seqlens[i]
        out_i, lse_i = scaled_dot_product_attention(
            q[i].transpose(0, 1),
            blocked_k_view[begin:end].transpose(0, 1),
            blocked_v_view[begin:end].transpose(0, 1),
            h_q=args.h_q,
            h_kv=args.h_kv,
            is_causal=not args.non_causal,
        )
        out_ref[i] = out_i.transpose(0, 1)
        lse_ref[i] = lse_i

    summary = {
        "shape": {
            "batch_size": args.batch_size,
            "s_q": args.s_q,
            "mean_sk": args.mean_sk,
            "h_q": args.h_q,
            "h_kv": args.h_kv,
            "d": args.d,
            "dv": args.dv,
            "block_size": args.block_size,
            "varlen": args.varlen,
            "causal": not args.non_causal,
        },
        "cache_seqlens": {
            "min": int(cache_seqlens.min().item()),
            "max": int(cache_seqlens.max().item()),
            "mean": float(cache_seqlens.float().mean().item()),
        },
        "out": _metric_dict(out_flash, out_ref),
        "lse": _metric_dict(lse_flash, lse_ref),
    }
    return summary, out_flash, lse_flash


def _enforce_thresholds(args: argparse.Namespace, summary: dict[str, object]) -> None:
    checks = []
    if args.max_out_cos_diff is not None:
        checks.append(("out.cos_diff", summary["out"]["cos_diff"], args.max_out_cos_diff))
    if args.max_lse_cos_diff is not None:
        checks.append(("lse.cos_diff", summary["lse"]["cos_diff"], args.max_lse_cos_diff))
    failed = [(name, value, limit) for name, value, limit in checks if value > limit]
    if failed:
        details = ", ".join(f"{name}={value:.6g} > {limit:.6g}" for name, value, limit in failed)
        raise SystemExit(f"flash_mla_smoke_threshold_failed: {details}")


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please check your MACA driver and PyTorch installation."
        )
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("FlashMLA smoke test requires a CUDA-compatible MACA device.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(
            f"Device index {device.index} is out of range. Total available devices: "
            f"{torch.cuda.device_count()}"
        )

    torch.set_default_dtype(args.dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)

    print(f"torch={torch.__version__}")
    print(f"device={torch.cuda.get_device_name(device)}")
    print(f"dtype={args.dtype}")
    summary, out_flash, lse_flash = _run_flash_mla_case(args)
    print(
        "out_metrics="
        f"cos_diff={summary['out']['cos_diff']:.6g}, "
        f"rmse={summary['out']['rmse']:.6g}, "
        f"amax_diff={summary['out']['amax_diff']:.6g}"
    )
    print(
        "lse_metrics="
        f"cos_diff={summary['lse']['cos_diff']:.6g}, "
        f"rmse={summary['lse']['rmse']:.6g}, "
        f"amax_diff={summary['lse']['amax_diff']:.6g}"
    )
    print(f"flash_out_shape={tuple(out_flash.shape)}")
    print(f"flash_lse_shape={tuple(lse_flash.shape)}")
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    _enforce_thresholds(args, summary)

    print("flash_mla_smoke_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
