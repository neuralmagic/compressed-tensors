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

from typing import Type
import torch
from compressed_tensors.transform import TransformLocation


__all__ = ["get_matrix_size", "apply_transform_weight"]


def get_matrix_size(module: torch.nn.Module, location: TransformLocation) -> int:
    """
    Determine the size of a matrix given its location on the module

    :param module: module that matrix will be applied to
    :param location: location on module
    :return: size of matrix
    """
    if isinstance(module, torch.nn.Linear):
        if location in (TransformLocation.INPUT, TransformLocation.WEIGHT_INPUT):
            return module.in_features
        else:
            return module.out_features
    elif isinstance(module, torch.nn.Embedding):
        if location in ("input", TransformLocation.WEIGHT_INPUT):
            return module.num_embeddings
        else:
            return module.embedding_dim

    raise ValueError(
        f"Unsupported module type {type(module)}, "
        "should be either Linear or Embedding."
    )


def apply_transform_weight(
    transform_weight: torch.Tensor,
    value: torch.Tensor,
    location: TransformLocation,
    module_type: Type,
) -> torch.Tensor:
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

    :param transform_weight: transform weight to apply
    :param value: value to apply transform_weight to
    :param location: determines how transform_weight should be applied
    :param model_type: result of type(module), passed in to determine application of
        weight transform. This is needed because torch uses convention:
        - torch.nn.Linear(in_features,out_features) has weight shape
            (out_features, in_features)
        - torch.nn.Embedding(num_embeddings, embedding_dim) has weight shape
            (num_embeddings, embedding_dim)
        The transform has to account for Linear's transposed weights
    :return: value after transform_weight has been applied
    """

    if location == TransformLocation.INPUT:
        return value @ transform_weight

    elif location == TransformLocation.WEIGHT_INPUT:
        if module_type is torch.nn.Linear:
            return value @ transform_weight.T
        else:
            raise NotImplementedError(
                f"{TransformLocation.WEIGHT_INPUT} transform not "
                f"implemented for module type {module_type}"
            )

    elif location == TransformLocation.WEIGHT_OUTPUT:
        if module_type is torch.nn.Linear:
            return transform_weight.T @ value
        elif module_type is torch.nn.Embedding:
            return value @ transform_weight
        else:
            raise NotImplementedError(
                f"{TransformLocation.WEIGHT_OUTPUT} transform not "
                f"implemented for module type {module_type}"
            )

    elif location == TransformLocation.OUTPUT:
        return value @ transform_weight

    else:
        raise NotImplementedError(f"{location} has not been implemented yet")
