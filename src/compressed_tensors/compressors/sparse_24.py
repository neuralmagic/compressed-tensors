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

import torch
from compressed_tensors.compressors import Compressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs
from torch import Tensor


__all__ = [
    "Sparse24Compressor",
]


@Compressor.register(name=CompressionFormat.sparse_24.value)
class Sparse24Compressor(Compressor):
    """ """

    COMPRESSION_PARAM_NAMES = []  # TODO: what is the expected format by the kernel?

    def compression_param_info(  # TODO: expand this function to work for sparsity, maybe we end up with two compressor types for sparsity and quantization?
        self,
        weight_shape: torch.Size,
        quantization_args: Optional[QuantizationArgs] = None,
    ) -> Dict[str, Tuple[torch.Size, torch.dtype]]:
        """
        Creates a dictionary of expected shapes and dtypes for each compression
            parameter used by the compressor

        :param weight_shape: uncompressed weight shape
        :param quantization_args: quantization parameters for the weight
        :return: dictionary mapping compressed parameter names to shape and dtype
        """
        raise NotImplementedError

    def compress_weight(  # TODO: expand this function to work for sparsity, maybe we end up with two compressor types for sparsity and quantization?
        self,
        weight: Tensor,
        scale: Tensor,
        zero_point: Optional[Tensor] = None,
        g_idx: Optional[torch.Tensor] = None,
        quantization_args: Optional[QuantizationArgs] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compresses a single uncompressed weight

        :param weight: uncompressed weight tensor
        :param scale: quantization scale for weight
        :param zero_point: quantization zero point for weight
        :param g_idx: optional mapping from column index to group index
        :param quantization_args: quantization parameters for weight
        :param device: optional device to move compressed output to
        :return: dictionary of compressed weight data
        """
        raise NotImplementedError

    def decompress_weight(  # TODO: expand this function to work for sparsity, maybe we end up with two compressor types for sparsity and quantization?
        self,
        compressed_data: Dict[str, Tensor],
        quantization_args: Optional[QuantizationArgs] = None,
    ) -> torch.Tensor:
        """
        Decompresses a single compressed weight

        :param compressed_data: dictionary of data needed for decompression
        :param quantization_args: quantization parameters for the weight
        :return: tensor of the decompressed weight
        """
        raise NotImplementedError
