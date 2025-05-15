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

from typing import Optional, Union

import torch
from compressed_tensors.transforms.base import (
    MatrixTransformBase,
    MatrixTransformFactory,
)
from compressed_tensors.transforms.hadamard_utils import deterministic_hadamard_matrix
from compressed_tensors.transforms.helpers import (
    ParameterizedDefaultDict,
    get_matrix_size,
    get_offload_device,
)
from compressed_tensors.transforms.transform_args import TransformArgs
from compressed_tensors.transforms.transform_scheme import TransformsScheme
from compressed_tensors.transforms.utils import (
    apply_matrix_transform,
    apply_permutation,
)
from torch import Tensor, device, dtype
from torch.nn import Linear, Module, Parameter


class HadamardTransform(MatrixTransformBase):
    def __init__(
        self, weight: Tensor, permutation: Optional[Tensor], args: TransformArgs
    ):
        super().__init__()
        self.weight = weight
        self.permutation = permutation
        self.args = args

    def forward(self, value: Parameter) -> Parameter:
        weight = self.weight if not self.args.inverse else self.weight.T
        if self.permutation is not None:
            weight = apply_permutation(weight, self.permutation)

        return apply_matrix_transform(weight, value, self.args.side)

    def right_inverse(self, value: Parameter) -> Parameter:
        weight = self.weight.T if not self.args.inverse else self.weight
        if self.permutation is not None:
            weight = apply_permutation(weight, self.permutation)

        return apply_matrix_transform(weight, value, self.args.side)


@MatrixTransformFactory.register("hadamard")
class HadamardFactory(MatrixTransformFactory):
    def __init__(self, name: str, scheme: TransformsScheme, seed: int):
        super().__init__(name, scheme, seed)
        self.weights = ParameterizedDefaultDict(self._create_weight)
        self.perms = ParameterizedDefaultDict(self._create_permutation)

    def create_transform(self, module: Module, args: TransformArgs):
        assert isinstance(module, Linear)
        size = get_matrix_size(module, args)
        dtype = module.weight.dtype
        device = get_offload_device(module)

        weight = self.weights[size, dtype, device]
        perm = self.perms[module, size] if self.scheme.randomize_modules else None
        return HadamardTransform(weight, perm, args)

    def _create_weight(self, size: int, dtype: dtype, device: device) -> Parameter:
        data = torch.tensor(deterministic_hadamard_matrix(size))  # seed=self.seed
        data = data.to(dtype=dtype, device=device)
        return Parameter(data, requires_grad=self.scheme.requires_grad)

    def _create_permutation(self, module: Module, size: int) -> Parameter:
        data = torch.randperm(size)
        return Parameter(data, requires_grad=self.scheme.requires_grad)
