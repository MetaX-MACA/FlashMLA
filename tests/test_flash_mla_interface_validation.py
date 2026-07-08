import sys
import types

import pytest
import torch


class _FakeFlashMla(types.SimpleNamespace):
    def get_mla_metadata(self, *args, **kwargs):
        return None

    def fwd_kvcache_mla(self, *args, **kwargs):
        return torch.empty(1), torch.empty(1)


sys.modules.setdefault("flash_mla_cuda", _FakeFlashMla())

from flash_mla.flash_mla_interface import flash_mla_with_kvcache  # noqa: E402


def _valid_inputs():
    q = torch.empty(2, 1, 4, 8)
    k_cache = torch.empty(8, 16, 1, 8)
    block_table = torch.zeros(2, 4, dtype=torch.int32)
    cache_seqlens = torch.ones(2, dtype=torch.int32)
    metadata = torch.zeros(1, 8, dtype=torch.int32)
    num_splits = torch.zeros(3, dtype=torch.int32)
    return q, k_cache, block_table, cache_seqlens, metadata, num_splits


def test_flash_mla_rejects_mismatched_batch_size():
    q, k_cache, block_table, cache_seqlens, metadata, num_splits = _valid_inputs()
    cache_seqlens = torch.ones(3, dtype=torch.int32)

    with pytest.raises(ValueError, match="batch size"):
        flash_mla_with_kvcache(
            q, k_cache, block_table, cache_seqlens, 4, metadata, num_splits
        )


def test_flash_mla_rejects_non_int32_cache_lengths():
    q, k_cache, block_table, cache_seqlens, metadata, num_splits = _valid_inputs()
    cache_seqlens = cache_seqlens.to(torch.int64)

    with pytest.raises(TypeError, match="cache_seqlens"):
        flash_mla_with_kvcache(
            q, k_cache, block_table, cache_seqlens, 4, metadata, num_splits
        )


def test_flash_mla_rejects_bad_num_splits_length():
    q, k_cache, block_table, cache_seqlens, metadata, _num_splits = _valid_inputs()
    num_splits = torch.zeros(2, dtype=torch.int32)

    with pytest.raises(ValueError, match="num_splits must have size"):
        flash_mla_with_kvcache(
            q, k_cache, block_table, cache_seqlens, 4, metadata, num_splits
        )


def test_flash_mla_rejects_non_tensor_inputs():
    q, k_cache, block_table, cache_seqlens, metadata, num_splits = _valid_inputs()

    with pytest.raises(TypeError, match="block_table must be a torch.Tensor"):
        flash_mla_with_kvcache(
            q, k_cache, block_table.tolist(), cache_seqlens, 4, metadata, num_splits
        )


def test_flash_mla_rejects_cross_device_inputs():
    q, k_cache, block_table, cache_seqlens, metadata, num_splits = _valid_inputs()
    block_table = block_table.to("meta")

    with pytest.raises(ValueError, match="same device"):
        flash_mla_with_kvcache(
            q, k_cache, block_table, cache_seqlens, 4, metadata, num_splits
        )


def test_flash_mla_rejects_invalid_value_head_dim():
    q, k_cache, block_table, cache_seqlens, metadata, num_splits = _valid_inputs()

    with pytest.raises(ValueError, match="head_dim_v"):
        flash_mla_with_kvcache(
            q, k_cache, block_table, cache_seqlens, 16, metadata, num_splits
        )
