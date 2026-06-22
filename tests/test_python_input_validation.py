import importlib
import sys
import types
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeFlashMla(types.SimpleNamespace):
    def __init__(self):
        super().__init__(
            get_mla_metadata=self.get_mla_metadata,
            fwd_kvcache_mla=self.fwd_kvcache_mla,
        )
        self.metadata_calls = 0
        self.kvcache_calls = 0

    def get_mla_metadata(self, cache_seqlens, num_heads_per_head_k, num_heads_k):
        self.metadata_calls += 1
        metadata = torch.empty((1, 16), dtype=torch.int32, device=cache_seqlens.device)
        num_splits = torch.empty((cache_seqlens.shape[0] + 1,), dtype=torch.int32, device=cache_seqlens.device)
        return metadata, num_splits

    def fwd_kvcache_mla(
        self,
        q,
        k_cache,
        _v_cache,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
    ):
        self.kvcache_calls += 1
        out = torch.empty((*q.shape[:-1], head_dim_v), dtype=q.dtype, device=q.device)
        lse = torch.empty((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32, device=q.device)
        return out, lse


class PythonInputValidationTest(unittest.TestCase):
    def setUp(self):
        self.fake_extension = FakeFlashMla()
        self._saved_modules = {}
        for name in ("flash_mla_cuda", "flash_mla", "flash_mla.flash_mla_interface"):
            if name in sys.modules:
                self._saved_modules[name] = sys.modules[name]
        sys.modules["flash_mla_cuda"] = self.fake_extension
        sys.modules.pop("flash_mla", None)
        sys.modules.pop("flash_mla.flash_mla_interface", None)
        self.interface = importlib.import_module("flash_mla.flash_mla_interface")

    def tearDown(self):
        for name in ("flash_mla", "flash_mla.flash_mla_interface", "flash_mla_cuda"):
            sys.modules.pop(name, None)
        sys.modules.update(self._saved_modules)

    def _valid_kvcache_inputs(self):
        batch_size = 2
        q = torch.randn(batch_size, 1, 4, 8)
        k_cache = torch.randn(3, 16, 2, 8)
        block_table = torch.zeros((batch_size, 1), dtype=torch.int32)
        cache_seqlens = torch.full((batch_size,), 8, dtype=torch.int32)
        tile_scheduler_metadata = torch.zeros((1, 16), dtype=torch.int32)
        num_splits = torch.zeros((batch_size + 1,), dtype=torch.int32)
        return q, k_cache, block_table, cache_seqlens, 4, tile_scheduler_metadata, num_splits

    def test_metadata_rejects_wrong_cache_seqlens_dtype_before_extension(self):
        cache_seqlens = torch.ones((2,), dtype=torch.int64)

        with self.assertRaisesRegex(TypeError, "cache_seqlens"):
            self.interface.get_mla_metadata(cache_seqlens, 4, 2)

        self.assertEqual(self.fake_extension.metadata_calls, 0)

    def test_metadata_rejects_non_positive_head_counts_before_extension(self):
        cache_seqlens = torch.ones((2,), dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "num_heads_per_head_k"):
            self.interface.get_mla_metadata(cache_seqlens, 0, 2)

        with self.assertRaisesRegex(ValueError, "num_heads_k"):
            self.interface.get_mla_metadata(cache_seqlens, 4, 0)

        self.assertEqual(self.fake_extension.metadata_calls, 0)

    def test_kvcache_rejects_incompatible_heads_before_extension(self):
        q, k_cache, block_table, cache_seqlens, head_dim_v, metadata, num_splits = self._valid_kvcache_inputs()
        q = torch.randn(q.shape[0], q.shape[1], 3, q.shape[3])

        with self.assertRaisesRegex(ValueError, "divisible"):
            self.interface.flash_mla_with_kvcache(
                q, k_cache, block_table, cache_seqlens, head_dim_v, metadata, num_splits
            )

        self.assertEqual(self.fake_extension.kvcache_calls, 0)

    def test_kvcache_rejects_bad_num_splits_length_before_extension(self):
        q, k_cache, block_table, cache_seqlens, head_dim_v, metadata, _num_splits = self._valid_kvcache_inputs()
        num_splits = torch.zeros((q.shape[0],), dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "batch_size \\+ 1"):
            self.interface.flash_mla_with_kvcache(
                q, k_cache, block_table, cache_seqlens, head_dim_v, metadata, num_splits
            )

        self.assertEqual(self.fake_extension.kvcache_calls, 0)

    def test_kvcache_rejects_mismatched_q_and_k_cache_dtype_before_extension(self):
        q, k_cache, block_table, cache_seqlens, head_dim_v, metadata, num_splits = self._valid_kvcache_inputs()
        k_cache = k_cache.to(torch.float16)

        with self.assertRaisesRegex(TypeError, "same dtype"):
            self.interface.flash_mla_with_kvcache(
                q, k_cache, block_table, cache_seqlens, head_dim_v, metadata, num_splits
            )

        self.assertEqual(self.fake_extension.kvcache_calls, 0)

    def test_kvcache_accepts_valid_inputs_and_uses_default_scale(self):
        q, k_cache, block_table, cache_seqlens, head_dim_v, metadata, num_splits = self._valid_kvcache_inputs()

        out, lse = self.interface.flash_mla_with_kvcache(
            q, k_cache, block_table, cache_seqlens, head_dim_v, metadata, num_splits
        )

        self.assertEqual(out.shape, (2, 1, 4, head_dim_v))
        self.assertEqual(lse.shape, (2, 4, 1))
        self.assertEqual(self.fake_extension.kvcache_calls, 1)


if __name__ == "__main__":
    unittest.main()
