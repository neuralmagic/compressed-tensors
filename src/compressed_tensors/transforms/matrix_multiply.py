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
from compressed_tensors.transforms import Transforms
from compressed_tensors.transforms.utils import (
    SingletonMatrixRegistry,
    apply_matrix_transform,
)


# TODO: fix loading + add generic matrix registry?
@Transforms.register("matrix-mul")
class MatrixMultiply(Transforms):
    def __init__(
        self,
        name: str,
        transform_data: torch.Tensor,
        size: Optional[int] = None,
        empty: Optional[bool] = False,
        device: Optional[Union[str, torch.device]] = "cpu",
        dtype: Optional[torch.dtype] = torch.bfloat16,
    ):

        if empty and size is None:
            raise ValueError(
                "size is required when setting up empty transforms for deserialization "
            )

        # name required to either pull a cached matrix or cache a matrix itself
        # will assume each name corresponds to a unique matrix
        self.name = name
        self.matrix_registry = SingletonMatrixRegistry()

        if empty:
            transform = torch.empty((size, size)).to(dtype)
        else:
            transform = self.fetch().to(dtype).to(device)

        super().__init__(transform=transform)

        if not self.matrix_registry.contains(self.name):
            self.matrix_registry.set_matrix(self.name, self.transform)

    def fetch(self):
        transform = self.matrix_registry.get_matrix(self.name)
        if transform is None:
            transform = transform_data
        return transform

    def apply(
        self,
        input_tensor: torch.Tensor,
        transpose: bool = False,
        first: bool = True,
    ) -> torch.Tensor:
        return apply_matrix_transform(
            transform=self.transform,
            input_tensor=input_tensor,
            transpose=transpose,
            first=first,
        )

    def inverse_apply(
        self,
        input_tensor: torch.Tensor,
        transpose: bool = False,
        first: bool = True,
    ) -> torch.Tensor:
        """
        Apply the inverse operation of `apply`

        :param transform: matrix tensor
        :param input_tensor: tensor to which the transform matrix is applied
        :param transpose: whether or not the transform matrix is transposed before
            being applied.
        :param first: if the transform matrix will be the first or second matrix to be
            multiplied
        """

        # Note: not implemented for lower precision than float32
        return apply_matrix_transform(
            transform=torch.linalg.inv(self.transform),
            input_tensor=input_tensor,
            transpose=transpose,
            first=first,
        )
