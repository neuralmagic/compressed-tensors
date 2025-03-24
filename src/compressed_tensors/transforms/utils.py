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


__all__ = ["apply_matrix_transform", "SingletonMatrixRegistry"]


class SingletonMatrixRegistry:
    _instance = None

    def __new__(cls):
        # Check if the instance already exists
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._data = {}  # Initialize the data storage
        return cls._instance

    def set_matrix(self, key, value):
        self._data[key] = value

    def get_matrix(self, key):
        return self._data.get(key, None)

    def contains(self, key):
        return key in self._data


def apply_matrix_transform(
    transform: torch.Tensor,
    input_tensor: torch.Tensor,
    transpose: bool = False,
    first: bool = True,
) -> torch.Tensor:
    """
    Apply a matrix-type transform

    :param transform: transform tensor
    :param input_tensor: tensor to which the transform matrix is applied
    :param transpose: whether or not the transform matrix is transposed before
        being applied.
    :param first: if the transform matrix will be the first or second matrix to be
        multiplied

    returns a transformed input_tensor
    """

    if transpose:
        return (
            torch.matmul(transform.T, input_tensor)
            if first
            else torch.matmul(input_tensor, transform.T)
        )

    return (
        torch.matmul(transform, input_tensor)
        if first
        else torch.matmul(input_tensor, transform)
    )
