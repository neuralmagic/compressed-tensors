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
from compressed_tensors.transform import TransformArgs, TransformScheme
from compressed_tensors.transform.factory.base import TransformBase, TransformFactory
from compressed_tensors.transform.utils.helpers import ParameterizedDefaultDict
from compressed_tensors.transform.utils.utils import (
    apply_matrix_transform,
    get_matrix_size,
)
from compressed_tensors.utils import get_offloaded_device
from torch import Tensor, device, dtype
from torch.nn import Linear, Module, Parameter


@TransformFactory.register("matrix-mul")
class RandomMatrixFactory(TransformFactory):
    def __init__(
        self,
        name: str,
        scheme: TransformScheme,
        seed: int = 42,
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
        device = get_offloaded_device(module)

        weight = self.weights[size, dtype, device]
        inverse = self.inverses[size, dtype, device]
        return RandomMatrixTransform(weight, inverse, args)

    def _create_weight(self, size: int, dtype: dtype, device: device) -> torch.Tensor:
        return torch.rand((size, size), dtype=dtype, device=device)

    def _create_inverse(
        self, size: int, dtype: dtype, device: device
    ) -> Optional[torch.Tensor]:
        if self.cache_inverses:
            weight = self.weights[size, dtype, device]
            return high_precision_invert(weight)
        else:
            return None


class RandomMatrixTransform(TransformBase):
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
            weight = high_precision_invert(self.weight)

        return apply_matrix_transform(weight, value, self.args.side)


def high_precision_invert(weight: Tensor) -> Tensor:
    return torch.linalg.inv(weight.to(torch.float32)).to(weight.dtype)
