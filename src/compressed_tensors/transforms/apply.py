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


__all__ = ["apply_transforms_to_parameter", "apply_inverse_transforms_to_parameter"]


def apply_transforms_to_parameter(
    module: torch.nn.Module,
    module_parameter: torch.nn.Parameter,
    transform_data: TransformData,
):
    """
    Apply all transforms relevant to a parameter using a module's
    transform data. The parameter data is updated in-place.

    :param module: torch.nn.Moudle
    :param module_parameter: the torch.nn.Parameter to transform
    :param transform_data: a module's TransformData

    Only implemented for weight parameters thus far.

    """

    for transform_name, transform_values in transform_data.data.items():
        transform = getattr(module, transform_name)
        apply = Transforms.fetch_apply(transform_values.get("type"))
        call_args = transform_values.get("call_args")
        transformed_param_data = apply(
            input_tensor=module_parameter, transform=transform, **call_args
        )
        module_parameter.data.copy_(transformed_param_data)


def apply_inverse_transforms_to_parameter(
    module: torch.nn.Module,
    module_parameter: torch.nn.Parameter,
    transform_data: TransformData,
):
    """
    Apply all inverse transform operations relevant to a parameter using a module's
    TransformData. The parameter data is updated in-place.

    :param module: torch.nn.Moudle
    :param module_parameter: the torch.nn.Parameter to transform
    :param transform_data: a module's TransformData

    Only implemented for weight parameters thus far.

    """

    for transform_name, transform_values in reversed(transform_data.data.items()):
        transform = getattr(module, transform_name)
        inverse_apply = Transforms.fetch_inverse_apply(transform_values.get("type"))
        call_args = transform_values.get("call_args")
        transformed_param_data = inverse_apply(
            input_tensor=module_parameter, transform=transform, **call_args
        )
        module_parameter.data.copy_(transformed_param_data)
