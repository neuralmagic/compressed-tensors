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
from typing import Dict, Optional

import torch
from compressed_tensors.compressors import Compressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.quantization.utils import can_quantize
from torch import Tensor


__all__ = [
    "QuantizationCompressor",
    "IntQuantizationCompressor",
    "FloatQuantizationCompressor",
]

_LOGGER: logging.Logger = logging.getLogger(__name__)


@Compressor.register(name=CompressionFormat.naive_quantized.value)
class QuantizationCompressor(Compressor):
    """
    Implements naive compression for quantized models. Weight of each
    quantized layer is converted from its original float type to the closest Pytorch
    type to the type specified by the layer's QuantizationArgs.
    """

    COMPRESSION_PARAM_NAMES = ["weight", "weight_scale", "weight_zero_point"]

    def compression_param_info(
        self,
        weight_shape: torch.Size,
        quantization_args: Optional[QuantizationArgs] = None,
    ):
        dtype = quantization_args.pytorch_dtype()
        return {"weight": (weight_shape, dtype)}

    def compress_weight(
        weight: Tensor,
        scale: Tensor,
        zero_point: Optional[Tensor] = None,
        quantization_args: Optional[QuantizationArgs] = None,
    ):
        if can_quantize(weight, quantization_args):
            compressed_weight = quantize(
                x=weight,
                scale=scale,
                zero_point=zero_point,
                args=quantization_args,
                dtype=quantization_args.pytorch_dtype(),
            )

            return compressed_weight

        return None

    def decompress_weight(
        self,
        compressed_data: Dict[str, Tensor],
        quantization_args: Optional[QuantizationArgs] = None,
    ):
        weight = compressed_data["weight"]
        scale = compressed_data["weight_scale"]
        zero_point = compressed_data["weight_zero_point"]
        decompressed_weight = dequantize(
            x_q=weight,
            scale=scale,
            zero_point=zero_point,
        )

        return decompressed_weight


@Compressor.register(name=CompressionFormat.int_quantized.value)
class IntQuantizationCompressor(QuantizationCompressor):
    """
    Alias for integer quantized models
    """

    pass


@Compressor.register(name=CompressionFormat.float_quantized.value)
class FloatQuantizationCompressor(QuantizationCompressor):
    """
    Alias for fp quantized models
    """

    pass
