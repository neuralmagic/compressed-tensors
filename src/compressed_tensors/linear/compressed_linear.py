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

from typing import Optional

import torch
from compressed_tensors.compressors.base import Compressor
from compressed_tensors.quantization import (
    QuantizationScheme,
    QuantizationStatus,
    initialize_module_for_quantization,
)
from torch import Tensor
from torch.nn import Parameter
from torch.nn.functional import linear
from torch.nn.modules import Module


class CompressedLinear(Module):
    """
    Wrapper module for running a compressed forward pass of a quantized Linear module.
    The wrapped layer will decompressed on each forward call.

    :param quantization_scheme:
    :param quantization_format:
    :param device:
    :param weight_shape:
    """

    def __init__(
        self,
        quantization_scheme: QuantizationScheme,
        quantization_format: str,
        device: torch.device,
        weight_shape: Optional[torch.Size] = None,
    ):
        super().__init__()

        # These will get replaced
        self.weight = None
        self.bias = None

        self.compressor = Compressor.load_from_registry(quantization_format)

        if quantization_scheme.weights is not None:
            dtype = quantization_scheme.weights.pytorch_dtype()
            # need a dummy weight of the correct shape for initialization
            # TODO: this actually won't work for packed compression, revisit
            self.weight = Parameter(
                torch.empty(weight_shape, device=device, dtype=dtype),
                requires_grad=False,
            )

        # this will initialize all the scales and zero points
        initialize_module_for_quantization(
            self, quantization_scheme, force_zero_point=False
        )

        # no need for this once quantization is initialized
        delattr(self, "weight")

        # get the shape and dtype of compressed parameters
        compression_params = self.compressor.compression_param_info(
            weight_shape, quantization_scheme.weights
        )

        # populate compressed weights and quantization parameters
        for name, (shape, dtype) in compression_params.items():
            param = Parameter(
                torch.empty(shape, device=device, dtype=dtype), requires_grad=False
            )
            self.register_parameter(name, param)

        # mark module as compressed
        self.quantization_status = QuantizationStatus.COMPRESSED

    def forward(self, input: Tensor) -> Tensor:
        """
        Decompresses the weight, then runs the wrapped forward pass
        """
        uncompressed_weight = self.compressor.decompress_module(self)
        return linear(input, uncompressed_weight, self.bias)
