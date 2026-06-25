# Adapted from deepseek-ai/FlashMLA(https://github.com/deepseek-ai/FlashMLA)
__version__ = "1.0.1"

__all__ = ["__version__", "get_mla_metadata", "flash_mla_with_kvcache"]

_INTERFACE_FUNCS = None


def _load_interface():
    global _INTERFACE_FUNCS
    if _INTERFACE_FUNCS is None:
        try:
            from flash_mla.flash_mla_interface import flash_mla_with_kvcache, get_mla_metadata
        except ImportError as exc:
            raise ImportError(
                "flash_mla_cuda is not available. Build and install FlashMLA from source "
                "in a configured MACA environment before calling FlashMLA kernels."
            ) from exc
        _INTERFACE_FUNCS = (get_mla_metadata, flash_mla_with_kvcache)
    return _INTERFACE_FUNCS


def get_mla_metadata(cache_seqlens, num_heads_per_head_k, num_heads_k):
    metadata_func, _ = _load_interface()
    return metadata_func(cache_seqlens, num_heads_per_head_k, num_heads_k)


def flash_mla_with_kvcache(
    q,
    k_cache,
    block_table,
    cache_seqlens,
    head_dim_v,
    tile_scheduler_metadata,
    num_splits,
    softmax_scale=None,
    causal=False,
):
    _, flash_func = _load_interface()
    return flash_func(
        q,
        k_cache,
        block_table,
        cache_seqlens,
        head_dim_v,
        tile_scheduler_metadata,
        num_splits,
        softmax_scale=softmax_scale,
        causal=causal,
    )
