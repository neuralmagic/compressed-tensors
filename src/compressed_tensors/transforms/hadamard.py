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
from compressed_tensors.transforms.hadamard_utils import random_hadamard_matrix


@Transforms.register("hadamard")
class Hadamard(Transforms):
    def __init__(
        self, size: Optional[int] = None, transform: Optional[torch.Tensor] = None
    ):
        """
        Produces a randomly generated matrix with dims (size, size), with values
        between -1 and 1, and the property HH.T == nI i.e the transformation
        matrix when multiplied by its transpose, is a multiple of the identity.

        :param size: size of the matrix, if generating a new Hadamard matrix.
            The generated matrix will have dimensions (size, size)
        :param transform: if loading in a previously generated matrix, will
            use that through this transformation, as opposed to creating a new
            one.

        TODO: We can likely make the serialization of this more efficient:
        The generation of this matrix starts with generating a random
        matrix with dims (size, size). We could potentially just store
        a randomly generated seed and the size, as opposed to storing the entire
        matrix, to reproduce an identical matrix during runtime. That way,
        we will not have to store the entire matrix. Will need to consider
        accuracy implications.
        """

        if transform:
            super().__init__(transform=transform)
        else:
            assert size is not None
            # Note: scipy's hadamard method seems faster however
            # has more restrictions on the dimensions (must be power of 2)/conflicts
            # with what is needed for model dimensions
            self.transform = random_hadamard_matrix(size=size)

    def __call__(
        self, input_tensor: torch.Tensor, transpose: bool = False
    ) -> torch.Tensor:
        """
        :param input_tensor: tensor to which the hadamard matrix is applied
        :param transpose: whether or not the hadamard matrix is transposed before
            being applied.

        returns a transformed input_tensor
        """
        if transpose:
            return torch.matmul(input_tensor, self.transform.T)
        return torch.matmul(input_tensor, self.transform)
