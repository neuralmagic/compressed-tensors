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
from compressed_tensors.transforms.hadamard_utils import (
    SingletonHadamardRegistry,
    deterministic_hadamard_matrix,
)
from compressed_tensors.transforms.utils import apply_matrix_transform


@Transforms.register("hadamard")
class Hadamard(Transforms):
    def __init__(
        self,
        size: int,
        empty: Optional[bool] = False,
        device: Optional[Union[str, torch.device]] = "cuda",
        dtype: Optional[torch.dtype] = torch.bfloat16,
        learnable: Optional[bool] = False,
    ):

        """
        Produces a hadamard matrix with dims (size, size), with values
        -1 and 1, and the property HH.T == nI i.e the transformation
        matrix when multiplied by its transpose is a multiple of the identity.
        All rows and columns are orthonormal. The matrix returned
        is not normalized and will be deterministic.

        :param size: size of the matrix, if generating a new Hadamard matrix.
            The generated matrix will have dimensions (size, size)
        :param transform: if loading in a previously generated matrix, will
            use that through this transformation, as opposed to creating a new
            one
        :param dtype: type to cast the rotation matrix to

        """
        self.learnable = learnable
        self.hadamard_registry = SingletonHadamardRegistry()
        self.size = size
        self.dtype = dtype

        if empty:
            # If saved, would have a different lifecycle (would be loaded and not be
            # the same parameter, for now)
            # Would take more memory
            transform = torch.empty((size, size)).to(dtype).to(device)
        else:
            transform = self.fetch().to(device)

        super().__init__(transform=transform, learnable=learnable)

        # if not learnable, save parameter
        if not self.learnable and size not in self.hadamard_registry._data:
            self.hadamard_registry.set_hadamard(size, self.transform)

    def fetch(self):
        # TODO: this is deterministic; we should just serialize the size
        transform = self.hadamard_registry.get_hadamard(self.size)
        if transform is None:
            transform = torch.Tensor(deterministic_hadamard_matrix(size=self.size)).to(
                self.dtype
            )

            # if learnable, save actual data, not parameter
            if self.learnable:
                self.hadamard_registry.set_hadamard(self.size, transform)

        return transform

    def apply(
        self,
        input_tensor: torch.Tensor,
        transpose: bool = False,
        first: bool = True,
    ) -> torch.Tensor:
        return apply_matrix_transform(
            transform=self.transform.to(input_tensor.device),
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

        :param transform: hadamard tensor
        :param input_tensor: tensor to which the transform matrix is applied
        :param transpose: whether or not the transform matrix is transposed before
            being applied.
        :param first: if the transform matrix will be the first or second matrix to be
            multiplied
        """
        transpose = not transpose
        # need to normalize before sending back
        return (
            apply_matrix_transform(
                transform=self.transform.to(input_tensor.device),
                input_tensor=input_tensor,
                transpose=transpose,
                first=first,
            )
            / self.transform.shape[0]
        )
