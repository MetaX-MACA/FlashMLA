import importlib.util
import random
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch


def _identity_decorator(*args, **kwargs):
    def decorate(func):
        return func

    return decorate


def _load_benchmark_module():
    fake_torch = types.ModuleType("torch")
    fake_torch.manual_seed = Mock()
    fake_torch.inference_mode = _identity_decorator
    fake_torch.tensor = lambda *args, **kwargs: list(args[0]) if args else []
    fake_torch.int32 = object()
    fake_torch.bfloat16 = object()

    fake_triton = types.ModuleType("triton")
    fake_triton.autotune = _identity_decorator
    fake_triton.heuristics = _identity_decorator
    fake_triton.jit = _identity_decorator
    fake_triton.Config = lambda *args, **kwargs: (args, kwargs)
    fake_triton.testing = types.SimpleNamespace(do_bench=lambda func: 0)

    fake_tl = types.ModuleType("triton.language")
    fake_tl.__getattr__ = lambda name: object()
    fake_flash_mla = types.ModuleType("flash_mla")
    fake_flash_mla.flash_mla_with_kvcache = Mock()
    fake_flash_mla.get_mla_metadata = Mock()

    module_name = "bench_flash_mla_test_module"
    source_path = Path(__file__).resolve().parents[1] / "benchmark" / "bench_flash_mla.py"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    module = importlib.util.module_from_spec(spec)
    stubbed_modules = {
        "flashinfer": types.ModuleType("flashinfer"),
        "flash_mla": fake_flash_mla,
        "torch": fake_torch,
        "triton": fake_triton,
        "triton.language": fake_tl,
    }
    with patch.dict(sys.modules, stubbed_modules, clear=False):
        sys.modules.pop(module_name, None)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module, fake_torch


def test_set_benchmark_seed_updates_torch_and_python_random():
    module, fake_torch = _load_benchmark_module()
    module.set_benchmark_seed(123)

    fake_torch.manual_seed.assert_called_once_with(123)
    first = random.random()
    random.seed(123)
    assert first == random.random()


if __name__ == "__main__":
    test_set_benchmark_seed_updates_torch_and_python_random()
