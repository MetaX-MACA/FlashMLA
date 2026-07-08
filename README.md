# FlashMLA on MXMACA
We provide the implementation of FlashMLA from FlashAttention-2(version 2.6.3), based on MACA toolkit and C500 and C600 chips.

FlashAttention-2 currently supports:
1. Datatype fp16 and bf16.
2. Multi-Token Prediction greater or equal to 1.
3. Paged kvcache with block size equal to 2^n (n >= 0).

## How to run on MXMACA Device
## Installation

Requirements:
- MXMACA GPUs.
- MACA development toolkit.
- Mctlass source code.
- Pytorch2.1 and Triton2.1 from maca toolkit wheel package and above.

To install in conda env:
1. Make sure that maca pyTorch2.1 and Triton2.1 are installed.
2. Download mctlass source code from: https://sw-download.metax-tech.com/

### Set environment variables
```bash
export MACA_PATH=/your/maca/path
export CUDA_PATH=$MACA_PATH/tools/cu-bridge
export MACA_CLANG_PATH=$MACA_PATH/mxgpu_llvm/bin
export LD_LIBRARY_PATH=$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib:$LD_LIBRARY_PATH
```

### Install

```bash
python setup.py install
```

### Benchmark

```bash
python tests/test_flash_mla.py
```

### Smoke test

After building the extension, run a small smoke case before launching the full
benchmark suite:

```bash
python tools/run_flash_mla_smoke.py
```

The command prints the detected torch version and MACA device name, executes one
FlashMLA case, and reports the output and LSE deltas against the PyTorch
reference implementation. This is useful for validating that the extension can
compile, launch, and return numerically stable values on a target MACA stack.
Use `--dtype fp16`, the shape flags in `--help`, or optional thresholds such as
`--max-lse-cos-diff 1e-4` to tailor the check for your environment.

### Usage

```python
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)

for i in range(num_layers):
    ...
    o_i, lse_i = flash_mla_with_kvcache(
        q_i, kvcache_i, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits, causal=True,
    )
    ...
```

## Requirements

- MXMACA 2.27 and above
- PyTorch 2.0 and above

## Acknowledgement

FlashMLA is inspired by [FlashAttention 2&3](https://github.com/dao-AILab/flash-attention/) and [cutlass](https://github.com/nvidia/cutlass) projects.



## Citation

```bibtex
@misc{flashmla2025,
      title={FlashMLA: Efficient MLA decoding kernel},
      author={Jiashi Li},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/FlashMLA}},
}
```
