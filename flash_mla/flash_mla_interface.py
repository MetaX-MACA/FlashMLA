# Adapted from deepseek-ai/FlashMLA(https://github.com/deepseek-ai/FlashMLA)
from typing import Optional, Tuple, Callable, Dict
import torch

import flash_mla_cuda as flash_mla
from functools import wraps
import warnings
import numpy as np

def _check_int32_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must use torch.int32, got {tensor.dtype}")


def _validate_flash_mla_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
) -> None:
    for name, tensor in (
        ("q", q),
        ("k_cache", k_cache),
        ("block_table", block_table),
        ("cache_seqlens", cache_seqlens),
        ("tile_scheduler_metadata", tile_scheduler_metadata),
        ("num_splits", num_splits),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
        if tensor.device != q.device:
            raise ValueError(
                f"All tensors must be on the same device, but {name} is on {tensor.device} "
                f"while q is on {q.device}"
            )
    if q.dim() != 4:
        raise ValueError(f"q must be 4D, got shape {tuple(q.shape)}")
    if k_cache.dim() != 4:
        raise ValueError(f"k_cache must be 4D, got shape {tuple(k_cache.shape)}")
    if block_table.dim() != 2:
        raise ValueError(f"block_table must be 2D, got shape {tuple(block_table.shape)}")
    if cache_seqlens.dim() != 1:
        raise ValueError(
            f"cache_seqlens must be 1D, got shape {tuple(cache_seqlens.shape)}"
        )
    if num_splits.dim() != 1:
        raise ValueError(f"num_splits must be 1D, got shape {tuple(num_splits.shape)}")
    if (
        q.shape[0] != block_table.shape[0]
        or q.shape[0] != cache_seqlens.shape[0]
        or num_splits.shape[0] != q.shape[0] + 1
    ):
        raise ValueError(
            f"batch size mismatch: q batch_size is {q.shape[0]}, but block_table has "
            f"{block_table.shape[0]}, cache_seqlens has {cache_seqlens.shape[0]}, and "
            f"num_splits must have size {q.shape[0] + 1}, got {num_splits.shape[0]}"
        )
    if q.shape[-1] != k_cache.shape[-1]:
        raise ValueError(
            f"q head_dim ({q.shape[-1]}) must match k_cache head_dim ({k_cache.shape[-1]})"
        )
    if head_dim_v <= 0 or head_dim_v > k_cache.shape[-1]:
        raise ValueError(
            f"head_dim_v must be in (0, {k_cache.shape[-1]}], got {head_dim_v}"
        )
    _check_int32_tensor("block_table", block_table)
    _check_int32_tensor("cache_seqlens", cache_seqlens)
    _check_int32_tensor("tile_scheduler_metadata", tile_scheduler_metadata)
    _check_int32_tensor("num_splits", num_splits)


def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    # num_q_tokens_per_head_k: int,
    # num_heads_k: int,
    # num_heads_q: Optional[int] = None,
    # is_fp8_kvcache: bool = False,
    # topk: Optional[int] = None
    *args,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        cache_seqlens: (batch_size), dtype torch.int32.
        num_q_tokens_per_head_k: Equals to num_q_tokens_per_q_seq * num_heads_q // num_heads_k.
        num_heads_k: The number of k heads.
        num_heads_q: The number of q heads. This argument is optional when sparse attention is not enabled
        is_fp8_kvcache: Whether the k_cache and v_cache are in fp8 format.
        topk: If not None, sparse attention will be enabled, and only tokens in the `indices` array passed to `flash_mla_with_kvcache_sm90` will be attended to.

    Returns:
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
        num_splits: (batch_size + 1), dtype torch.int32.
    """

    if "num_heads_per_head_k" in kwargs:
        """
        Handle Deprecation calls (using num_heads_per_head_k)
        Arguments:
            cache_seqlens: (batch_size), dtype torch.int32.
            num_heads_per_head_k: Equals to seq_len_q * num_heads_q // num_heads_k.
            num_heads_k: num_heads_k.
        """
        warnings.warn(
            "Parameter 'num_heads_per_head_k' is deprecated. Please use 'num_q_tokens_per_head_k' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        num_heads_per_head_k = kwargs.pop("num_heads_per_head_k")

        if "num_heads_k" in kwargs:
            num_heads_k = kwargs.pop("num_heads_k")
        elif len(args) >= 1:
            num_heads_k = args[0]
            args = args[1:]
        else:
            raise TypeError(
                "Legacy call missing required 'num_heads_k' (position 2 or keyword argument)"
            )

        if "num_heads_q" in kwargs or "is_fp8_kvcache" in kwargs or "topk" in kwargs:
            raise TypeError(
                "The legacy call does not support the parameters: (num_heads_q, is_fp8_kvcache, topk). \
                    If you want to use them, please replace the parameter name 'num_heads_per_head_k' with 'num_q_tokens_per_head_k'"
            )

        if len(args) > 0 or len(kwargs) > 0:
            extra_args = list(args) + list(kwargs.keys())
            raise TypeError(
                f"Legacy calls do not support extra position parameters: {extra_args}. Legacy parameters only support (cache_seqlens, num_heads_per_head_k, num_heads_k) \
                    \nIf you want to use (num_heads_q, is_fp8_kvcache, topk), please replace the parameter name 'num_heads_per_head_k' with 'num_q_tokens_per_head_k'"
            )

        return flash_mla.get_mla_metadata(
            cache_seqlens, num_heads_per_head_k, num_heads_k, None, False, None
        )
    else:
        """
        Handle v32 calls (using num_q_tokens_per_head_k)
        Arguments:
            cache_seqlens: (batch_size), dtype torch.int32.
            num_q_tokens_per_head_k: Equals to num_q_tokens_per_q_seq * num_heads_q // num_heads_k.
            num_heads_k: The number of k heads.
            num_heads_q: The number of q heads. This argument is optional when sparse attention is not enabled
            is_fp8_kvcache: Whether the k_cache and v_cache are in fp8 format.
            topk: If not None, sparse attention will be enabled, and only tokens in the `indices` array passed to `flash_mla_with_kvcache_sm90` will be attended to.
        """
        if len(args) > 5:
            raise TypeError(
                f"get_mla_metadata() takes 6 positional arguments but {len(args)+1} were given"
            )

        if "num_q_tokens_per_head_k" in kwargs:
            num_q_tokens_per_head_k = kwargs.pop("num_q_tokens_per_head_k")
        elif len(args) >= 1:
            num_q_tokens_per_head_k = args[0]
            args = args[1:]
        else:
            raise TypeError(
                "get_mla_metadata() missing required 'num_q_tokens_per_head_k' (position 1 or keyword argument)"
            )

        if len(args) >= 1 and "num_heads_k" not in kwargs:
            num_heads_k = args[0]
            args = args[1:]
        elif "num_heads_k" in kwargs:
            num_heads_k = kwargs.pop("num_heads_k")
        elif len(args) <= 2:
            raise TypeError(
                "get_mla_metadata() missing required 'num_heads_k' (position 2 or keyword argument)"
            )

        num_heads_q: Optional[int] = None
        is_fp8_kvcache: bool = False
        topk: Optional[int] = None
        if len(args) >= 1 and "num_heads_q" not in kwargs:
            num_heads_q = args[0]
            args = args[1:]
        elif "num_heads_q" in kwargs:
            num_heads_q = kwargs.pop("num_heads_q")

        if len(args) >= 1 and "is_fp8_kvcache" not in kwargs:
            is_fp8_kvcache = args[0]
            args = args[1:]
        elif "is_fp8_kvcache" in kwargs:
            is_fp8_kvcache = kwargs.pop("is_fp8_kvcache")

        if len(args) >= 1 and "topk" not in kwargs:
            topk = args[0]
            args = args[1:]
        elif "topk" in kwargs:
            topk = kwargs.pop("topk")

        if len(kwargs) > 0:
            raise TypeError(
                f"Unrecognized keyword arguments: {list(kwargs.keys())}. Supported parameters: cache_seqlens, num_q_tokens_per_head_k, num_heads_k, num_heads_q, is_fp8_kvcache, topk"
            )

        return flash_mla.get_mla_metadata(
            cache_seqlens,
            num_q_tokens_per_head_k,
            num_heads_k,
            num_heads_q,
            is_fp8_kvcache,
            topk,
        )


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
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    indices_all_valid_per_q: Optional[torch.tensor] = None,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqlen_k=None,
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
        is_fp8_kvcache: bool. Whether the k_cache and v_cache are in fp8 format. Do not support fp8 kvcahce right now.
        indices: (batch_size, seq_len_q, topk), torch.int32. If not None, sparse attention will be enabled, and only tokens in the `indices` array will be attended to. Invalid indices should be set to -1 .
        indices_all_valid_per_q: [batch_size, seq_len_q, 1], bool. A scalar boolean tensor indicating wheter all entries in indices for every query token are valid.
        - descale_q: (batch_size),
        torch.float32. Descaling factors for Q, used for fp8 quantization.
        - descale_k: (batch_size),
        torch.float32. Descaling factors for K, used for fp8 quantization.
    Return:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    _validate_flash_mla_inputs(
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
    if indices is not None:
        assert causal == False, "causal must be `false` if sparse attention is enabled."
    assert (descale_q is None) == (descale_k is None), (
        "descale_q and descale_k should be both None or both not None"
    )
    #bf16/fp16 dense/sparse MLA
    if indices_all_valid_per_q is None:
        batch_size = q.shape[0]
        seqlen_q = q.shape[1]
        indices_all_valid_per_q = torch.full((batch_size, seqlen_q, 1), False, dtype=torch.bool, device=q.device)
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
        is_fp8_kvcache,
        indices,
        indices_all_valid_per_q,
        cp_world_size,
        cp_rank,
        cp_tot_seqlen_k,
    )
    return out, softmax_lse

def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    indices_all_valid_per_q: Optional[torch.tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        indices_all_valid_per_q: [s_q, 1], bool. A scalar boolean tensor indicating wheter all entries for every query token in indices are valid.
        d_v: The dimension of value vectors. Can only be 512

    Returns:
        (output, max_logits, lse)
        About the definition of output, max_logits and lse, please refer to README.md
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, 2-based log-sum-exp
    """
    if indices_all_valid_per_q is None:
        indices_all_valid_per_q = torch.full((q.shape[0], 1), False, dtype=torch.bool, device=q.device)
    results = flash_mla.sparse_prefill_fwd(
        q, kv, indices, sm_scale, d_v, indices_all_valid_per_q
    )
    return results
