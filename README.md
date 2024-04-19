# compressed-tensors

This repository extends a [safetensors](https://github.com/huggingface/safetensors) format to efficiently store sparse and/or quantized tensors on disk. `compressed-tensors` format supports multiple compression types to minimize the disk space and facilitate the tensor manipulation.

## Motivation

### Reduce disk space by saving sparse tensors in a compressed format

The compressed format stores the data much more efficiently by taking advantage of two properties of tensors:

- Sparse tensors -> due to a large number of entries that are equal to zero.
- Quantized -> due to their low precision representation.


### Introduce an elegant interface to save/load compressed tensors

The library provides the user with the ability to compress/decompress tensors. The properties of tensors are defined by human-readable configs, allowing the users to understand the compression format at a quick glance.

## Installation

### Pip

```bash
pip install compressed-tensors
```

### From source

```bash
git clone https://github.com/neuralmagic/compressed-tensors
cd compressed-tensors
pip install -e .
```

## Getting started

### Saving

The function `save_compressed` returns an optional `compression_config` (if compression has been applied). It can be used to inspect the applied compression.

```python
from compressed_tensors import save_compressed
from torch import Tensor

tensors: Dict[str, Tensor] = ...
compression_config: Dict = save_compressed(tensors, "model.safetensors")
```

### Loading

```python
from compressed_tensors import load_compressed
from torch import Tensor

tensors: Dict[str, Tensor] = load_compressed("model.safetensors", device="cpu")
```

## Benefits
TODO

## SafeTensors File Format

For each parameter in the uncompressed state_dict, we store the following attributes needed for decompression in the compressed state_dict:

- Compressed tensor
- Bitmask
- Uncompressed shape
- Row offsets

```python
# Dense
{
    PARAM_NAME: uncompressed_tensor
}

# Compressed
{
    PARAM_NAME.compressed: compressed_tensor,  # 1d tensor
    PARAM_NAME.bitmask: value,  # 2d bitmask tensor (nrows x (ncols / 8))
    PARAM_NAME.shape: value,  # Uncompressed shape tensor
    PARAM_NAME.row_offsets: value  # 1d offsets tensor
}
```

The library provides pathways to automatically add the config information to the HF config file.

```json
// config.json
{
    "sparsity_config": {
        "format": "sparse_bitmask", // "dense_sparsity" for the original tensor format

        // Informational
        "sparsity_structure": "unstructured", // Or 2:4, 8:16, etc.
        "global_sparsity": "0.5"
    }
}
```