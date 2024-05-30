# compressed_tensors

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

### Saving/Loading Compressed Tensors (Bitmask Compression)

The function `save_compressed` uses the `compression_format` argument to apply compression to tensors.
The function `load_compressed` reverses the process: converts the compressed weights on disk to decompressed weights in device memory.

```python
from compressed_tensors import save_compressed, load_compressed, BitmaskConfig
from torch import Tensor
from typing import Dict

# the example BitmaskConfig method efficiently compresses 
# tensors with large number of zero entries 
compression_config = BitmaskConfig()

tensors: Dict[str, Tensor] = {"tensor_1": Tensor(
    [[0.0, 0.0, 0.0], 
     [1.0, 1.0, 1.0]]
)}
# compress tensors using BitmaskConfig compression format (save them efficiently on disk)
save_compressed(tensors, "model.safetensors", compression_format=compression_config.format)

# decompress tensors (load_compressed returns a generator for memory efficiency)
decompressed_tensors = {}
for tensor_name, tensor in load_compressed("model.safetensors", compression_config = compression_config):
    decompressed_tensors[tensor_name] = tensor
```

## Saving/Loading Compressed Models (Bitmask Compression)

We can apply bitmask compression to a whole model. For more detailed example see `example` directory.
```python
from compressed_tensors import save_compressed_model, load_compressed, BitmaskConfig
from transformers import AutoModelForCausalLM

model_name = "neuralmagic/llama2.c-stories110M-pruned50"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

original_state_dict = model.state_dict()

compression_config = BitmaskConfig()

# save compressed model weights
save_compressed_model(model, "compressed_model.safetensors", compression_format=compression_config.format)

# load compressed model weights (`dict` turns generator into a dictionary)
state_dict = dict(load_compressed("compressed_model.safetensors", compression_config))
```

For more in-depth tutorial on bitmask compression, refer to the [notebook](https://github.com/neuralmagic/compressed-tensors/blob/d707c5b84bc3fef164aebdcd97cb6eaa571982f8/examples/bitmask_compression.ipynb).


## Saving a Compressed Model with PTQ

We can use compressed-tensors to run basic post training quantization (PTQ) and save the quantized model compressed on disk

```python
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:0", torch_dtype="auto")

config = QuantizationConfig.parse_file("./examples/bit_packing/int4_config.json")
config.quantization_status = QuantizationStatus.CALIBRATION
apply_quantization_config(model, config)

dataset = load_dataset("ptb_text_only")["train"]
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding=False, truncation=True, max_length=1024)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
data_loader = DataLoader(tokenized_dataset, batch_size=1, collate_fn=DefaultDataCollator())

with torch.no_grad():
    for idx, sample in tqdm(enumerate(data_loader), desc="Running calibration"):
        sample = {key: value.to(device) for key,value in sample.items()}
        _ = model(**sample)

        if idx >= 512:
            break

model.apply(freeze_module_quantization)
model.apply(compress_quantized_weights)

output_dir = "./ex_llama1.1b_w4a16_packed_quantize"
compressor = ModelCompressor(quantization_config=config)
compressed_state_dict = compressor.compress(model)
model.save_pretrained(output_dir, state_dict=compressed_state_dict)
```

For more in-depth tutorial on quantization compression, refer to the [notebook](./examples/quantize_and_pack_int4.ipynb).
