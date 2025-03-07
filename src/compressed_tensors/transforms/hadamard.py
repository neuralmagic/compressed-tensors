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
from compressed_tensors.transforms.hadamard_utils import deterministic_hadamard_matrix


@Transforms.register("hadamard")
class Hadamard(Transforms):
    def __new__(
        cls,
        size: int,
        empty: Optional[bool] = False,
        device: Optional[Union[str, torch.device]] = "cuda",
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
        if not empty:
            # TODO: this is deterministic; we should just serialize the size
            transform = torch.Tensor(deterministic_hadamard_matrix(size=size))
        else:
            transform = torch.empty((size, size))

        return super().__new__(cls, transform=transform, device=device, dtype=dtype)

    @staticmethod
    def inverse_apply(
        transform: torch.Tensor,
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
                transform=transform,
                input_tensor=input_tensor,
                transpose=transpose,
                first=first,
            )
            / transform.shape[0]
        )
