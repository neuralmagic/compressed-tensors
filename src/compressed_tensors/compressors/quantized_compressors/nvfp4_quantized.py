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
from compressed_tensors.compressors.quantized_compressors.utils import (
    pack_fp4_to_uint8,
    unpack_fp4_from_uint8,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from torch import Tensor


@BaseCompressor.register(name=CompressionFormat.nvfp4_pack_quantized.value)
class NVFP4PackedCompressor(BaseQuantizationCompressor):
    """
    Implements compression of FP4 values. Weights of each quantized layer
    are packed into uint8. Only supports symmetric weight compression for now.
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
        if device is not None:
            weight_packed = weight_packed.to(device)
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
        # TODO: use a user provided dequant dtype
        unpacked = unpack_fp4_from_uint8(weight, m, n * 2)
        decompressed_weight = dequantize(
            x_q=unpacked, scale=scale, global_scale=global_scale, dtype=unpacked.dtype
        )

        return decompressed_weight


@BaseCompressor.register(name=CompressionFormat.mxfp4_pack_quantized.value)
class MXFP4PackedCompressor(BaseQuantizationCompressor):
    """
    Implements compression of MXFP4 values. Weights of each quantized layer
    are packed into uint8. Only supports symmetric weight compression for now.
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
        )

    def compress_weight(
        self,
        weight: Tensor,
        scale: Tensor,
        quantization_args: QuantizationArgs,
        global_scale: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        zero_point: Optional[torch.Tensor] = None,
        g_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        quantized_weight = quantize(
            x=weight,
            scale=scale,
            zero_point=zero_point,
            args=quantization_args,
        )
        compressed_dict = {}
        weight_packed = pack_fp4_to_uint8(quantized_weight)
        if device is not None:
            weight_packed = weight_packed.to(device)
        compressed_dict["weight_packed"] = weight_packed
        return compressed_dict

    def decompress_weight(
        self,
        compressed_data: Dict[str, Tensor],
        quantization_args: Optional[QuantizationArgs] = None,
    ) -> torch.Tensor:

        weight = compressed_data["weight_packed"]
        scale = compressed_data["weight_scale"]
        m, n = weight.shape
        # TODO: use a user provided dequant dtype
        unpacked = unpack_fp4_from_uint8(weight, m, n * 2)
        decompressed_weight = dequantize(
            x_q=unpacked, scale=scale, dtype=unpacked.dtype
        )

        return decompressed_weight
