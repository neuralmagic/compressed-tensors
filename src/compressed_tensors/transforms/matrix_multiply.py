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
from compressed_tensors.transforms.base import (
    MatrixTransformBase,
    MatrixTransformFactory,
)
from compressed_tensors.transforms.helpers import (
    ParameterizedDefaultDict,
    get_matrix_size,
    get_offload_device,
)
from compressed_tensors.transforms.transform_args import TransformArgs
from compressed_tensors.transforms.transform_scheme import TransformsScheme
from compressed_tensors.transforms.utils import apply_matrix_transform
from torch import Tensor, device, dtype
from torch.nn import Linear, Module, Parameter


class RandomMatrixTransform(MatrixTransformBase):
    def __init__(self, weight: Tensor, inverse: Optional[Tensor], args: TransformArgs):
        super().__init__()
        self.weight = weight
        self.args = args

        self.register_buffer("inverse", inverse, persistent=False)  # extra memory

    def forward(self, value: Parameter) -> Parameter:
        if self.args.inverse:
            weight = self.weight
        elif self.inverse is not None:
            weight = self.inverse
        else:
            weight = torch.linalg.inv(self.weight)

        return apply_matrix_transform(weight, value, self.args.side)


@MatrixTransformFactory.register("matrix-mul")
class RandomMatrixFactory(MatrixTransformFactory):
    def __init__(
        self,
        name: str,
        scheme: TransformsScheme,
        seed: int,
        cache_inverses: bool = True,
    ):
        super().__init__(name, scheme, seed)
        self.cache_inverses = cache_inverses

        self.weights = ParameterizedDefaultDict(self._create_weight)
        self.inverses = ParameterizedDefaultDict(self._create_inverse)

    def create_transform(self, module: Module, args: TransformArgs):
        assert isinstance(module, Linear)
        size = get_matrix_size(module, args)
        dtype = module.weight.dtype
        device = get_offload_device(module)

        weight = self.weights[size, dtype, device]
        inverse = self.inverses[size, dtype, device]
        return RandomMatrixTransform(weight, inverse, args)

    def _create_weight(self, size: int, dtype: dtype, device: device) -> torch.Tensor:
        return torch.random((size, size), dtype=dtype, device=device)

    def _create_inverse(
        self, size: int, dtype: dtype, device: device
    ) -> Optional[torch.Tensor]:
        if self.cache_inverses:
            weight = self.weights[size, dtype, device]
            return torch.linalg.inv(weight)
        else:
            return None
