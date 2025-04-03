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
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]
conversion_dict = {
    0.0: 0,
    0.5: 1,
    1.0: 2,
    1.5: 3,
    2.0: 4,
    3.0: 5,
    4.0: 6,
    6.0: 7,
    -0.0: 8,
    -0.5: 9,
    -1.0: 10,
    -1.5: 11,
    -2.0: 12,
    -3.0: 13,
    -4.0: 14,
    -6.0: 15,
}


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
            x_q=unpacked, scale=scale, global_scale=global_scale
        )

        return decompressed_weight


def pack_fp4_to_uint8(x: torch.Tensor):
    m, n = x.shape

    # convert to bits
    x_array = x.cpu().to(torch.float32).numpy()
    x_index = numpy.array(
        [[conversion_dict[i] for i in row] for row in x_array], dtype=numpy.uint8
    )
    x_index_bits = numpy.unpackbits(x_index)

    # unpack
    packed_shape = numpy.zeros([x_index_bits.shape[0] // 2], numpy.uint8)
    start = 0
    end = 16
    i = 0

    # janky bit manipulation
    while end < len(x_index_bits):
        packed_shape[i + 4 : i + 8] = x_index_bits[start:end][4:8]
        packed_shape[i : i + 4] = x_index_bits[start:end][12:16]
        start = end
        end = start + 16
        i += 8

    # pack
    packed = numpy.packbits(packed_shape)
    packed = torch.from_numpy(packed).to(torch.uint8)
    # reshape
    packed = packed.reshape(m, n // 2)
    return packed


# from vLLM
def unpack_fp4_from_uint8(x: torch.Tensor, m: int, n: int):
    v_2nd = x & 0xF
    v_1st = (x >> 4) & 0xF
    c = torch.stack((v_2nd, v_1st), dim=-1)
    out = torch.tensor([FLOAT_TO_E2M1[x] for x in c.flatten()])
    out = out.reshape(m, n).to(torch.float32)
    return out
