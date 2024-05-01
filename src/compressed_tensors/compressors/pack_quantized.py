# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from typing import Dict, Generator, Tuple

import numpy as np
import torch
from compressed_tensors.compressors import Compressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.quantization.utils import get_torch_bit_depth
from compressed_tensors.utils import get_nested_weight_mappings, merge_names
from safetensors import safe_open
from torch import Tensor
from tqdm import tqdm


__all__ = ["PackedQuantizationCompressor", "pack_4bit_ints"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


@Compressor.register(name=CompressionFormat.int_quantized.value)
class PackedQuantizationCompressor(Compressor):
    """
    Compresses a quantized model by packing every 4 4-bit weights into a torch.int32
    """

    COMPRESSION_PARAM_NAMES = ["weight", "weight_scale", "weight_zero_point"]

    def compress(self, model_state: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        model_quant_args = kwargs["model_quant_args"]
        compressed_dict = {}
        _LOGGER.debug(
            f"Compressing model with {len(model_state)} parameterized layers..."
        )

        for name, value in tqdm(model_state.items(), desc="Compressing model"):
            if name.endswith(".weight"):
                prefix = name.removesuffix(".weight")
                scale = model_state.get(merge_names(prefix, "weight_scale"), None)
                zp = model_state.get(merge_names(prefix, "weight_zero_point"), None)
                if scale is not None and zp is not None:
                    # weight is quantized, compress it
                    quant_args = model_quant_args[prefix]
                    bit_depth = get_torch_bit_depth(value)
                    if bit_depth > quant_args.num_bits:
                        # convert weight to an int if needed
                        value = quantize(
                            x=value,
                            scale=scale,
                            zero_point=zp,
                            args=quant_args,
                            dtype=torch.int8,
                        )
                        value = pack_4bit_ints(value)

            compressed_dict[name] = value.to("cpu")

        return compressed_dict

    def decompress(
        self, path_to_model_or_tensors: str, device: str = "cpu"
    ) -> Generator[Tuple[str, Tensor], None, None]:
        weight_mappings = get_nested_weight_mappings(
            path_to_model_or_tensors, self.COMPRESSION_PARAM_NAMES
        )
        for weight_name in weight_mappings.keys():
            weight_data = {}
            for param_name, safe_path in weight_mappings[weight_name].items():
                full_name = merge_names(weight_name, param_name)
                with safe_open(safe_path, framework="pt", device=device) as f:
                    weight_data[param_name] = f.get_tensor(full_name)

            if len(weight_data) == len(self.COMPRESSION_PARAM_NAMES):
                decompressed = dequantize(
                    x_q=weight_data["weight"],
                    scale=weight_data["weight_scale"],
                    zero_point=weight_data["weight_zero_point"],
                )
                yield merge_names(weight_name, "weight"), decompressed


def pack_4bit_ints(value: torch.Tensor):
    if value.dtype is not torch.int8:
        raise ValueError("Tensor must be quantized to torch.int8 before packing")

    # convert to unsigned so we can pack
    pack_depth = 32
    temp = (value - 8).to(torch.uint8)
    bits = np.unpackbits(temp.numpy(), axis=-1, bitorder="little")
    ranges = np.array([range(x, x + 4) for x in range(0, bits.shape[1], 8)]).flatten()
    only_4_bits = bits[:, ranges]

    padding = (
        math.ceil(only_4_bits.shape[1] / pack_depth) * pack_depth - only_4_bits.shape[1]
    )
    padded_bits = np.pad(
        only_4_bits, pad_width=[(0, 0), (0, padding)], constant_values=0
    )
    compressed = np.packbits(padded_bits, axis=-1, bitorder="little")
    compressed = np.ascontiguousarray(compressed).view(np.int32)

    return torch.from_numpy(compressed)
