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
from compressed_tensors.transforms import Transforms
from scipy.linalg import hadamard


@Transforms.register("hadamard")
class Hadamard(Transforms):
    def __init__(
        self,
        size: Optional[int] = None,
        transform: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = torch.bfloat16,
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

        if transform is not None:
            super().__init__(transform=transform)
        else:
            assert size is not None
            # TODO: this is deterministic; we should just serialize the size
            self.transform = torch.Tensor(hadamard(n=size)).to(dtype)

    def __call__(
        self, input_tensor: torch.Tensor, transpose: bool = False, first: bool = True
    ) -> torch.Tensor:
        """
        :param input_tensor: tensor to which the hadamard matrix is applied
        :param transpose: whether or not the hadamard matrix is transposed before
            being applied.
        :param first: if the hadmard matrix will be the first or second matrix to be
            multiplied

        returns a transformed input_tensor
        """
        if transpose:
            return (
                torch.matmul(self.transform.T, input_tensor)
                if first
                else torch.matmul(input_tensor, self.transform.T)
            )

        return (
            torch.matmul(self.transform, input_tensor)
            if first
            else torch.matmul(input_tensor, self.transform)
        )
