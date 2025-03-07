# FlashMLA on MXMACA
We provide the implementation of FlashMLA from FlashAttention-2(version 2.6.3), based on MACA toolkit and C500 chips.

FlashAttention-2 currently supports:
1. Datatype fp16 and bf16.
2. Multi-Token Prediction greater or equal to 1.
3. Paged kvcache with block size equal to 2^n (n >= 0)

## How to run on MXMACA Device
## Installation

Requirements:
- MXMACA GPUs.
- MACA development toolkit.
- mcTlass source code.
- mcPytorch2.1 and mcTriton2.1 from maca toolkit wheel package and above.

To install flash attn in conda env:
1. Make sure that maca pytorch2.1 and triton2.1 is installed.
2. Download mctlass source code from FlashAttention2/csrc/mctlass on website : https://sw-download.metax-tech.com/

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
