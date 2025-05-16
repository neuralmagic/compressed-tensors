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

from typing import Literal

import torch


__all__ = ["apply_matrix_transform"]


def apply_matrix_transform(
    weight: torch.Tensor, value: torch.Tensor, side: Literal["left", "right"]
) -> torch.Tensor:
    if side == "left":
        return weight @ value

    else:
        return value @ weight


def apply_permutation(weight: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    weight = weight.clone()
    diag_indices = torch.arange(weight.size(0))
    weight[diag_indices, diag_indices] = weight.diagonal()[perm]
    return weight
