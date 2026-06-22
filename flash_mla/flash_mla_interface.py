# Adapted from deepseek-ai/FlashMLA(https://github.com/deepseek-ai/FlashMLA)
from typing import Optional, Tuple

import torch

# import flash_mla_cuda
import flash_mla_cuda as flash_mla


def _check_dim(name: str, tensor: torch.Tensor, dim: int) -> None:
    if tensor.dim() != dim:
        raise ValueError(f"{name} must be a {dim}D tensor, got {tensor.dim()}D")


def _check_dtype(name: str, tensor: torch.Tensor, dtype: torch.dtype) -> None:
    if tensor.dtype is not dtype:
        raise TypeError(f"{name} must use dtype {dtype}, got {tensor.dtype}")


def _check_same_device(reference_name: str, reference: torch.Tensor, name: str, tensor: torch.Tensor) -> None:
    if tensor.device != reference.device:
        raise ValueError(
            f"{name} must be on the same device as {reference_name}: "
            f"got {tensor.device} and {reference.device}"
        )


def _validate_metadata_inputs(
    cache_seqlens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
) -> None:
    _check_dim("cache_seqlens", cache_seqlens, 1)
    _check_dtype("cache_seqlens", cache_seqlens, torch.int32)
    if num_heads_per_head_k <= 0:
        raise ValueError("num_heads_per_head_k must be positive")
    if num_heads_k <= 0:
        raise ValueError("num_heads_k must be positive")


def _validate_kvcache_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
) -> None:
    _check_dim("q", q, 4)
    _check_dim("k_cache", k_cache, 4)
    _check_dim("block_table", block_table, 2)
    _check_dim("cache_seqlens", cache_seqlens, 1)
    _check_dim("tile_scheduler_metadata", tile_scheduler_metadata, 2)
    _check_dim("num_splits", num_splits, 1)

    _check_dtype("block_table", block_table, torch.int32)
    _check_dtype("cache_seqlens", cache_seqlens, torch.int32)
    _check_dtype("tile_scheduler_metadata", tile_scheduler_metadata, torch.int32)
    _check_dtype("num_splits", num_splits, torch.int32)
    if not q.is_floating_point():
        raise TypeError(f"q must be a floating point tensor, got {q.dtype}")
    if not k_cache.is_floating_point():
        raise TypeError(f"k_cache must be a floating point tensor, got {k_cache.dtype}")
    if q.dtype != k_cache.dtype:
        raise TypeError(
            f"q and k_cache must have the same dtype, got {q.dtype} and {k_cache.dtype}"
        )

    batch_size, _, num_heads_q, head_dim = q.shape
    _, _, num_heads_k, cache_head_dim = k_cache.shape

    if batch_size != block_table.shape[0]:
        raise ValueError(
            "block_table batch dimension must match q: "
            f"got {block_table.shape[0]} and {batch_size}"
        )
    if batch_size != cache_seqlens.shape[0]:
        raise ValueError(
            "cache_seqlens length must match q batch dimension: "
            f"got {cache_seqlens.shape[0]} and {batch_size}"
        )
    if num_splits.shape[0] != batch_size + 1:
        raise ValueError(
            "num_splits length must be batch_size + 1: "
            f"got {num_splits.shape[0]} and {batch_size + 1}"
        )
    if num_heads_k <= 0:
        raise ValueError("k_cache must contain at least one KV head")
    if num_heads_q % num_heads_k != 0:
        raise ValueError(
            "q num_heads must be divisible by k_cache num_heads: "
            f"got {num_heads_q} and {num_heads_k}"
        )
    if head_dim != cache_head_dim:
        raise ValueError(
            "q head_dim must match k_cache head_dim: "
            f"got {head_dim} and {cache_head_dim}"
        )
    if head_dim_v <= 0:
        raise ValueError("head_dim_v must be positive")
    if head_dim_v > cache_head_dim:
        raise ValueError(
            "head_dim_v must not exceed k_cache head_dim: "
            f"got {head_dim_v} and {cache_head_dim}"
        )

    for name, tensor in (
        ("k_cache", k_cache),
        ("block_table", block_table),
        ("cache_seqlens", cache_seqlens),
        ("tile_scheduler_metadata", tile_scheduler_metadata),
        ("num_splits", num_splits),
    ):
        _check_same_device("q", q, name, tensor)


def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        cache_seqlens: (batch_size), dtype torch.int32.
        num_heads_per_head_k: Equals to seq_len_q * num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.

    Return:
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
        num_splits: (batch_size + 1), dtype torch.int32.
    """
    _validate_metadata_inputs(cache_seqlens, num_heads_per_head_k, num_heads_k)
    return flash_mla.get_mla_metadata(cache_seqlens, num_heads_per_head_k, num_heads_k)


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head_dim of v.
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), torch.int32, return by get_mla_metadata.
        num_splits: (batch_size + 1), torch.int32, return by get_mla_metadata.
        softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.

    Return:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    _validate_kvcache_inputs(
        q,
        k_cache,
        block_table,
        cache_seqlens,
        head_dim_v,
        tile_scheduler_metadata,
        num_splits,
    )
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, softmax_lse = flash_mla.fwd_kvcache_mla(
        q,
        k_cache,
        None,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
    )
    return out, softmax_lse
