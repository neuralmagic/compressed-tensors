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
from compressed_tensors.transform.utils.hadamard_utils import (
    deterministic_hadamard_matrix,
)
from compressed_tensors.transform.utils.helpers import ParameterizedDefaultDict
from compressed_tensors.transform.utils.utils import (
    apply_matrix_transform,
    apply_permutation,
    get_matrix_size,
)
from compressed_tensors.utils import get_offloaded_device
from torch import device, dtype
from torch.nn import Linear, Module, Parameter


@TransformFactory.register("hadamard")
class HadamardFactory(TransformFactory):
    def __init__(self, name: str, scheme: TransformScheme, seed: int = 42):
        super().__init__(name, scheme, seed)
        self.weights = ParameterizedDefaultDict(self._create_weight)
        self.perms = ParameterizedDefaultDict(self._create_permutation)

    def create_transform(self, module: Module, args: TransformArgs):
        assert isinstance(module, Linear)
        size = get_matrix_size(module, args)
        dtype = module.weight.dtype
        device = get_offloaded_device(module)

        weight = self.weights[size, dtype, device]
        perm = self.perms[module, size] if self.scheme.randomize_modules else None
        return HadamardTransform(weight, perm, args)

    def _create_weight(self, size: int, dtype: dtype, device: device) -> Parameter:
        data = torch.tensor(deterministic_hadamard_matrix(size))  # seed=self.seed
        data = data.to(dtype=dtype, device=device)
        return Parameter(data, requires_grad=self.scheme.requires_grad)

    def _create_permutation(self, module: Module, size: int) -> Parameter:
        data = torch.randperm(size)
        return Parameter(data, requires_grad=False)


class HadamardTransform(TransformBase):
    def __init__(
        self, weight: Parameter, permutation: Optional[Parameter], args: TransformArgs
    ):
        super().__init__()
        self.weight = weight
        self.permutation = permutation
        self.args = args

    def forward(self, value: Parameter) -> Parameter:
        if not self.args.inverse:
            weight = self.weight
        else:
            weight = self.weight.T / self.weight.size(0)

        # if self.permutation is not None:
        #    weight = apply_permutation(weight, self.permutation)

        return apply_matrix_transform(weight, value, self.args)
