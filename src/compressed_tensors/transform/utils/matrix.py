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

from typing import Callable, Optional, Tuple

import torch
from compressed_tensors.transform import TransformLocation


__all__ = ["get_transform_size", "apply_transform_weight"]


def get_transform_size(
    module: torch.nn.Module,
    location: TransformLocation,
    head_dim: Optional[int] = None,
) -> int:
    """
    Determine the size of a transform matrix given its location on the module

    :param module: module that matrix will be applied to
    :param location: location on module
    :param head_dim: size of head when transform is applied to mha
    :return: size of matrix
    """
    if isinstance(module, torch.nn.Linear):
        if location in (TransformLocation.INPUT, TransformLocation.WEIGHT_INPUT):
            size = module.in_features
        else:
            size = module.out_features
    else:
        raise NotImplementedError(f"Transforms on {type(module)} are not supported")

    if head_dim is not None:
        if size % head_dim != 0:
            raise ValueError(
                f"{head_dim} must divide {size} for {type(module)} at {location}"
            )

        size = head_dim

    return size


def apply_transform_weight(
    weight: torch.Tensor,
    value: torch.Tensor,
    location: TransformLocation,
    module_type: type[torch.nn.Module],
) -> torch.Tensor:
    fn, axis = get_linear_transform_fn(module_type, location)

    assert weight.shape[0] == weight.shape[1]
    head_dim = weight.shape[0]
    num_heads = value.shape[axis] // head_dim

    value = value.unflatten(axis, (num_heads, head_dim))
    value = fn(weight, value)
    value = value.flatten(axis - 1, axis)

    return value


def get_linear_transform_fn(
    module_type: type[torch.nn.Module],
    location: TransformLocation,
) -> Tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], int]:
    """
    Using the transform location, determine how to apply the transform weight to the
    given value wrt linear weights. For more info on input and output transforms,
    see `TransformLocation`

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
    fn = axis = None

    if module_type == torch.nn.Linear:
        if location == TransformLocation.INPUT:
            fn = lambda weight, value: value @ weight
            axis = -1

        elif location == TransformLocation.WEIGHT_INPUT:
            fn = lambda weight, value: value @ weight.T
            axis = -1

        elif location == TransformLocation.WEIGHT_OUTPUT:
            fn = lambda weight, value: weight.T @ value
            axis = -2

        elif location == TransformLocation.OUTPUT:
            fn = lambda weight, value: value @ weight
            axis = -1

    if fn is None:
        raise NotImplementedError(
            f"Applying transforms to {module_type} {location} is not supported"
        )

    return fn, axis
