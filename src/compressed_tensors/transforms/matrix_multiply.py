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

import torch
from compressed_tensors.transforms import Transforms
from compressed_tensors.transforms.utils import apply_matrix_transform


# TODO: fix loading
@Transforms.register("matrix-mul")
class MatrixMultiply(Transforms):
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
        transform = torch.linalg.inv(transform)
        return apply_matrix_transform(
            transform=self.transform,
            input_tensor=input_tensor,
            transpose=transpose,
            first=first,
        )
