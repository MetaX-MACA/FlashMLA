import importlib
import inspect
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class LazyImportTest(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("flash_mla.flash_mla_interface", None)
        sys.modules.pop("flash_mla", None)

    def test_import_package_without_compiled_extension(self):
        module = importlib.import_module("flash_mla")

        self.assertEqual(module.__version__, "1.0.1")
        self.assertEqual(
            list(inspect.signature(module.get_mla_metadata).parameters),
            ["cache_seqlens", "num_heads_per_head_k", "num_heads_k"],
        )
        self.assertEqual(
            list(inspect.signature(module.flash_mla_with_kvcache).parameters),
            [
                "q",
                "k_cache",
                "block_table",
                "cache_seqlens",
                "head_dim_v",
                "tile_scheduler_metadata",
                "num_splits",
                "softmax_scale",
                "causal",
            ],
        )

    def test_interface_functions_are_cached_after_first_load(self):
        calls = []
        fake_interface = types.ModuleType("flash_mla.flash_mla_interface")

        def fake_get_mla_metadata(*args):
            calls.append(("metadata", args))
            return "metadata"

        def fake_flash_mla_with_kvcache(*args, **kwargs):
            calls.append(("flash", args, kwargs))
            return "flash"

        fake_interface.get_mla_metadata = fake_get_mla_metadata
        fake_interface.flash_mla_with_kvcache = fake_flash_mla_with_kvcache
        sys.modules["flash_mla.flash_mla_interface"] = fake_interface

        module = importlib.import_module("flash_mla")
        module._INTERFACE_FUNCS = None

        self.assertEqual(module.get_mla_metadata("seq", 2, 3), "metadata")
        sys.modules.pop("flash_mla.flash_mla_interface", None)
        self.assertEqual(module.get_mla_metadata("cached", 4, 5), "metadata")

        self.assertEqual(
            calls,
            [
                ("metadata", ("seq", 2, 3)),
                ("metadata", ("cached", 4, 5)),
            ],
        )


if __name__ == "__main__":
    unittest.main()
