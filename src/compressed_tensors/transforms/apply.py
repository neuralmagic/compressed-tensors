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
from compressed_tensors.transforms import Transforms
from compressed_tensors.transforms.transform_data import TransformData
from compressed_tensors.utils import update_parameter_data


__all__ = [
    "apply_transforms_to_activations_or_parameter",
    "apply_inverse_transforms_to_activations_or_parameter",
]


def apply_transforms_to_activations_or_parameter(
    module: torch.nn.Module,
    module_activation_or_parameter: torch.Tensor,
    transform_data: TransformData,
    update_in_place: Optional[bool] = True,
    base_name: Optional[str] = "weight",
) -> Optional[torch.Tensor]:
    """
    Apply all transforms relevant to a parameter using a module's
    transform data. The parameter data is updated in-place.

    :param module: torch.nn.Moudle
    :param module_activation_or_parameter: module Parameter or activations to transform
    :param transform_data: a module's TransformData
    """

    for transform_name, transform_values in transform_data.data.items():
        transform = transform_values.get("transform")
        call_args = transform_values.get("call_args")
        transformed_output_data = transform.apply(
            input_tensor=module_activation_or_parameter, **call_args
        )
        if not update_in_place:
            return transformed_output_data

        update_parameter_data(module, transformed_output_data, base_name)


def apply_inverse_transforms_to_activations_or_parameter(
    module: torch.nn.Module,
    module_activation_or_parameter: torch.nn.Parameter,
    transform_data: TransformData,
    update_in_place: Optional[bool] = True,
    base_name: Optional[str] = "weight",
) -> Optional[torch.Tensor]:
    """
    Apply all inverse transform operations relevant to a parameter using a module's
    TransformData. The parameter data is updated in-place.

    :param module: torch.nn.Moudle
    :param module_activation_or_parameter:  module Parameter or activations to transform
    :param transform_data: a module's TransformData
    """

    for transform_name, transform_values in reversed(transform_data.data.items()):
        transform = transform_values.get("transform")
        call_args = transform_values.get("call_args")
        transformed_output_data = transform.inverse_apply(
            input_tensor=module_activation_or_parameter, **call_args
        )
        if not update_in_place:
            return transformed_output_data

        update_parameter_data(module, transformed_output_data, base_name)
