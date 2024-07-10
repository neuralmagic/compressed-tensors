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
import torch.functional as F
from compressed_tensors.quantization import (
    QuantizationScheme,
    initialize_module_for_quantization,
)
from torch import Tensor
from torch.nn import Parameter
from torch.nn.modules import Module


class CompressedLinear(Module):
    # can pass in the shapes instead of the full parameters
    # then the state_dict can load automatically
    # could also pass in the kernel directly here
    # idea: kernel as part of the quantization_scheme, then the compressor is implied
    # idea: fn that takes in a scheme and spits out a valid kernel
    def __init__(
        self,
        quantization_scheme: QuantizationScheme,
        quantization_format: str,
        device,
        weight_shape: Optional[torch.Size] = None,
    ):
        super().__init__()

        # These will get replaced
        self.weight = None
        self.bias = None

        from compressed_tensors.compressors.base import Compressor

        self.compressor = Compressor.load_from_registry(quantization_format)

        if quantization_scheme.weights is not None:
            dtype = quantization_scheme.weights.pytorch_dtype()
            self.weight = Parameter(
                torch.empty(self.weight_shape, device=device, dtype=dtype),
                requires_grad=False,
            )

        # this will initialize all the scales and zero points
        initialize_module_for_quantization(self, quantization_scheme)
        delattr(self, "weight")

        compression_params = self.compressor.compression_param_shapes(weight_shape)
        for name, (shape, dtype) in compression_params.items():
            # todo, fill in dtype
            param = Parameter(
                torch.empty(shape, device=device, dtype=dtype), requires_grad=False
            )
            self.register_parameter(name, param)

    def forward(self, input: Tensor) -> Tensor:
        # this decompress call would become its own kernel,
        # in most use cases instead we would call a specific kernel
        uncompressed_weight = self.compressor.decompress_module(self)
        return F.linear(input, uncompressed_weight, self.bias)
