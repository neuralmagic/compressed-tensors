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


from typing import Dict, Optional, Tuple

import numpy
import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.compressors.quantized_compressors.base import (
    BaseQuantizationCompressor,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from torch import Tensor


FLOAT_TO_E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
]
conversion_dict = {}

# Dictionary between fp4 value and index
for i in range(len(FLOAT_TO_E2M1)):
    conversion_dict[FLOAT_TO_E2M1[i]] = i


def fp4_to_index(value):
    sign = torch.signbit(value)
    x = torch.abs(value)
    index = conversion_dict.get(x.item())

    if not sign:  # all positives
        return index
    else:  # all negatives
        return index + 8


@BaseCompressor.register(name=CompressionFormat.modelopt_quantized.value)
class ModelOptCompressor(BaseQuantizationCompressor):
    """
    Implements naive compression for quantized models. Weight of each
    quantized layer is converted from its original float type to the closest Pytorch
    type to the type specified by the layer's QuantizationArgs.
    """

    @property
    def compression_param_names(self) -> Tuple[str]:
        """
        Returns a tuple of compression parameter names introduced by
        the compressor during compression
        """
        return (
            "weight_packed",
            "weight_scale",
            "weight_zero_point",
            "weight_global_scale",
        )

    def compress_weight(
        self,
        weight: Tensor,
        scale: Tensor,
        global_scale: Tensor,
        quantization_args: QuantizationArgs,
        device: Optional[torch.device] = None,
        zero_point: Optional[torch.Tensor] = None,
        g_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        quantized_weight = quantize(
            x=weight,
            scale=scale,
            global_scale=global_scale,
            zero_point=zero_point,
            args=quantization_args,
        )
        compressed_dict = {}
        weight_packed = pack_fp4_to_uint8(quantized_weight)
        compressed_dict["weight_packed"] = weight_packed
        return compressed_dict

    def decompress_weight(
        self,
        compressed_data: Dict[str, Tensor],
        quantization_args: Optional[QuantizationArgs] = None,
    ) -> torch.Tensor:

        weight = compressed_data["weight_packed"]
        scale = compressed_data["weight_scale"]
        global_scale = compressed_data["weight_global_scale"]
        m, n = weight.shape
        unpacked = unpack_fp4_from_uint8(weight, m, n * 2)
        decompressed_weight = dequantize(
            x_q=unpacked, scale=scale, global_scale=global_scale, dtype=unpacked.dtype
        )

        return decompressed_weight


def pack_fp4_to_uint8(x: torch.Tensor):
    m, n = x.shape
    x_flatten = x.flatten()
    # convert to index value, unpack to bits
    x_index = numpy.array([fp4_to_index(i) for i in x_flatten], dtype=numpy.uint8)
    x_index_bits = torch.from_numpy(numpy.unpackbits(x_index)).to("cuda:0")

    packed_shape = (
        torch.zeros([x_index_bits.shape[0] // 2]).to(torch.uint8).to("cuda:0")
    )
    start = 0
    end = 16
    i = 0

    # janky bit manipulation
    while end <= len(x_index_bits):
        subset = x_index_bits[start:end]

        subset_a = subset[4:8]
        subset_b = subset[12:16]
        packed_shape[i + 4 : i + 8] = subset_a
        packed_shape[i : i + 4] = subset_b
        start = end
        end = start + 16
        i += 8

    # pack
    packed = numpy.packbits(packed_shape.cpu().numpy())
    packed = torch.Tensor(packed).to(torch.uint8).to("cuda:0")
    packed = packed.reshape(m, n // 2)
    return packed

kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)

# reference: : https://github.com/vllm-project/vllm/pull/16362
def unpack_fp4_from_uint8(a: torch.Tensor, m: int, n: int, dtype=torch.float32):
    assert a.dtype == torch.uint8

    # Vectorized nibble processing
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4  # Upper nibbles
    low = a_flat & 0x0F  # Lower nibbles

    # Combine nibbles for batch processing
    combined = torch.stack((low, high), dim=1).flatten()

    # Vectorized sign and magnitude extraction
    signs = (combined & 0x08).to(torch.bool)  # Sign bits
    abs_vals = (combined & 0x07).to(torch.long)  # Magnitude indices

    # Device-aware lookup and sign application
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)

    # Reshape to final form
    return values.reshape(m, n).to(dtype=dtype)
