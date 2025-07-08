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
from compressed_tensors.transform import TransformLocation


__all__ = ["get_matrix_size", "apply_transform_weight"]


def get_matrix_size(
    module: torch.nn.Module,
    location: TransformLocation,
    head_dim: Optional[int] = None,
) -> int:
    """
    Determine the size of a matrix given its location on the module

    :param module: module that matrix will be applied to
    :param location: location on module
    :TODO head_dim:
    :return: size of matrix
    """
    assert isinstance(module, torch.nn.Linear)

    if location in (TransformLocation.INPUT, TransformLocation.WEIGHT_INPUT):
        size = module.in_features
    else:
        size = module.out_features

    if head_dim is not None:
        assert size % head_dim == 0
        return head_dim

    else:
        return size


def apply_transform_weight(
    weight: torch.Tensor,
    value: torch.Tensor,
    location: TransformLocation,
) -> torch.Tensor:
    return apply_transform_weight_linear(weight, value, location)


def apply_transform_weight_linear(
    weight: torch.Tensor,
    value: torch.Tensor,
    location: TransformLocation,
):
    """
    Using the transform location, determine how to apply the transform weight to the
    given value. For more info on input and output transforms, see `TransformLocation`

    The following explains how weights should be applied to values according to location

    let  x          be input activation
         W          be weight,
         yh, xh, Wh be transformed output, input, weight

    note that
         y  = (x W.T)        // torch.nn.Linear

    Choose values for yh, xh, and Wh which incorporate matrix transforms

    let  V, Vi      be transform matrices on input side
         U, Ui      be transform matrices on output side

    pick xh = (x V)
         Wh = (U.T W Vi.T)
         yh = (y U)

    The following shows that `yh = (xh) (Wh).T` for the chosen values of yh, xh, and Wh

    (xh) (Wh).T = (x V) (U.T W Vi.T).T
                = (x V) (Vi W.T U)        // transpose matrix product identity
                = (x W.T) U
                = y U
                = yh

    :param weight: transform weight to apply
    :param value: value to apply weight to
    :param location: determines how weight should be applied
    :return: value after transform weight has been applied
    """
    value_shape = value.shape
    weight_size = weight.shape[0]
    assert weight.shape[0] == weight.shape[1]

    if location == TransformLocation.INPUT:
        num_heads = value_shape[1] // weight_size
        value = value.reshape(value_shape[0], num_heads, weight_size)
        ret = value @ weight

    elif location == TransformLocation.WEIGHT_INPUT:
        num_heads = value_shape[1] // weight_size
        value = value.reshape(value_shape[0], num_heads, weight_size)
        ret = value @ weight.T

    elif location == TransformLocation.WEIGHT_OUTPUT:
        num_heads = value_shape[0] // weight_size
        value = value.reshape(num_heads, weight_size, value_shape[1])
        ret = weight.T @ value

    elif location == TransformLocation.OUTPUT:
        num_heads = value_shape[1] // weight_size
        value = value.reshape(value_shape[0], num_heads, weight_size)
        ret = value @ weight

    else:
        raise NotImplementedError(f"{location} has not been implemented yet")

    return ret.reshape(value_shape)
