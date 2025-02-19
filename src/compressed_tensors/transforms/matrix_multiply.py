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

from compressed_tensors.transforms import Transforms


@Transforms.register("matrix_mul")
class MatrixMultiply(Transforms):
    def __call__(
        self, input_tensor: torch.Tensor, transpose: bool = False
    ) -> torch.Tensor:
        """
        :param input_tensor: tensor to which the transformation is applied
        :param transpose: whether or not the transformation is transposed before
            being applied.
        """

        if transpose:
            return torch.matmul(input_tensor, self.transform.T)
        return torch.matmul(input_tensor, self.transform)
