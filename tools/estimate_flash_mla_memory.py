#!/usr/bin/env python3
import argparse
import json
import math


DTYPE_BYTES = {
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
}


def estimate_bytes(args: argparse.Namespace) -> dict[str, int]:
    dtype_bytes = DTYPE_BYTES[args.dtype]
    max_seqlen_pad = math.ceil(args.mean_sk / 256) * 256
    num_blocks = args.batch_size * math.ceil(max_seqlen_pad / args.block_size)

    q = args.batch_size * args.s_q * args.h_q * args.d * dtype_bytes
    k_cache = num_blocks * args.block_size * args.h_kv * args.d * dtype_bytes
    out = args.batch_size * args.s_q * args.h_q * args.dv * dtype_bytes
    lse = args.batch_size * args.h_q * args.s_q * 4
    block_table = args.batch_size * math.ceil(max_seqlen_pad / args.block_size) * 4
    cache_seqlens = args.batch_size * 4

    total = q + k_cache + out + lse + block_table + cache_seqlens
    return {
        "q": q,
        "k_cache": k_cache,
        "out": out,
        "lse": lse,
        "block_table": block_table,
        "cache_seqlens": cache_seqlens,
        "total": total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate FlashMLA test tensor memory.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--s-q", type=int, default=1)
    parser.add_argument("--mean-sk", type=int, default=4096)
    parser.add_argument("--h-q", type=int, default=16)
    parser.add_argument("--h-kv", type=int, default=1)
    parser.add_argument("--d", type=int, default=576)
    parser.add_argument("--dv", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", choices=sorted(DTYPE_BYTES), default="bf16")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    estimates = estimate_bytes(args)
    gib = estimates["total"] / 1024**3
    if args.json:
        payload = dict(estimates)
        payload["total_gib"] = gib
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for name, value in estimates.items():
            print(f"{name}: {value / 1024**2:.2f} MiB")
        print(f"total_gib: {gib:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
