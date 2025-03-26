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

import math
from typing import Optional, Union

import torch
from compressed_tensors.transforms import Transforms
from compressed_tensors.transforms.hadamard_utils import random_hadamard_matrix
from compressed_tensors.transforms.utils import (
    SingletonMatrixRegistry,
    apply_matrix_transform,
)


# TODO: allow randomness for both potentially, separate by generation type
# this will make randomness a creation arg instead
@Transforms.register("random-hadamard")
class RandomHadamard(Transforms):
    def __init__(
        self,
        size: int,
        transform_name: str,
        permutation_name: str,
        empty: Optional[bool] = False,
        device: Optional[Union[str, torch.device]] = "cpu",
        dtype: Optional[torch.dtype] = torch.bfloat16,
        *args,
        **kwargs,
    ):
        """
        Produces a randomly generated matrix with dims (size, size), with values
        between -1 and 1, and the property HH.T == I i.e the transformation
        matrix when multiplied by its transpose is the identity.
        All rows and columns are orthonormal. The matrix returned
        is normalized and has the form (1/sqrt(size)) * M where all elements
        of M are -1 or +1.

        :param size: size of the matrix, if generating a new Hadamard matrix.
            The generated matrix will have dimensions (size, size)
        :param transform: if loading in a previously generated matrix, will
            use that through this transformation, as opposed to creating a new
            one
        :param dtype: type to cast the rotation matrix to

        TODO: We can likely make the serialization of this more efficient:
        The generation of this matrix starts with generating a random
        matrix with dims (size, size). We could potentially just store
        a randomly generated seed and the size, as opposed to storing the entire
        matrix, to reproduce an identical matrix during runtime. That way,
        we will not have to store the entire matrix. Will need to consider
        accuracy implications.
        """
        self.size = size
        self.normalized_size = math.sqrt(self.size)
        # TODO: potentially lives outside of the registry
        # And caching is controlled by llmcompressor
        self.matrix_registry = SingletonMatrixRegistry()

        if empty:
            transform = torch.empty((size, size)).to(dtype)
            permutation = torch.empty((size)).to(dtype).to(device)
        else:
            transform = self.fetch().to(dtype).to(device)
            permutation = (
                (torch.randint(low=0, high=2, size=(self.size,)) * 2 - 1)
                .to(dtype)
                .to(device)
            )

        super().__init__(
            transform=transform,
            permutation=permutation,
            transform_name=transform_name,
            permutation_name=permutation_name,
        )

        if not self.matrix_registry.contains(size):
            self.matrix_registry.set_matrix(self.size, self.transform)

    def fetch(self):
        transform = self.matrix_registry.get_matrix(self.size)
        if transform is None:
            transform = random_hadamard_matrix(size=self.size)
        return transform

    def apply(
        self,
        input_tensor: torch.Tensor,
        transpose: bool = False,
        first: bool = True,
    ) -> torch.Tensor:
        return apply_matrix_transform(
            transform=(self.transform * self.permutation) / self.normalized_size,
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
        return apply_matrix_transform(
            transform=(self.transform * self.permutation) / self.normalized_size,
            input_tensor=input_tensor,
            transpose=transpose,
            first=first,
        )
