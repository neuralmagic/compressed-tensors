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
from compressed_tensors.transform import TransformArgs


__all__ = ["apply_matrix_transform"]


def get_matrix_size(module: torch.nn.Module, args: TransformArgs) -> int:
    assert isinstance(module, torch.nn.Linear)
    if args.location == "input" or (args.location == "weight" and args.side == "input"):
        return module.in_features
    else:
        return module.out_features


def apply_matrix_transform(
    weight: torch.Tensor, value: torch.Tensor, args: TransformArgs
) -> torch.Tensor:
    # let x          be input activation
    #     W          be weight,
    #     yh, xh, Wh be transformed output, input, weight
    #
    # note that
    #     y  = (x W.T)        // torch.nn.Linear
    #     yh = (xh) (Wh).T    // transformed
    #
    # show that the following values for yh, xh, and Wh are consistent
    #
    # let V, Vi      be transform matrices on input side
    #     U, Ui      be transform matrices on output side
    #
    # pik xh = (x V)
    #     Wh = (U.T W Vi.T)
    #     yh = (y U)
    #
    # (xh) (Wh).T = (x V) (U.T W Vi.T).T
    #             = (x V) (Vi W.T U)        // transpose matrix product identity
    #             = (x W.T) U
    #             = y U
    #             = yh

    if args.location == "input":
        return value @ weight

    elif args.location == "weight":
        if args.side == "input":
            return value @ weight.T

        elif args.side == "output":
            return weight.T @ value

    elif args.location == "output":
        return value @ weight


def apply_permutation(weight: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    weight = weight.clone()
    diag_indices = torch.arange(weight.size(0))
    weight[diag_indices, diag_indices] = weight.diagonal()[perm]
    return weight
