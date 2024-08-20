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

"""
Miscelaneous helpers for the quantization lifecycle
"""

import torch
from torch.nn import Module


__all__ = [
    "update_layer_weight_quant_params",
    "enable_quantization",
    "disable_quantization",
]

# these datatypes are missing implementations for the `index_put` operation
EXPERIMENTAL_DTYPES = [torch.float8_e4m3fn]


def update_layer_weight_quant_params(layer: Module):
    weight = getattr(layer, "weight", None)
    scale = getattr(layer, "weight_scale", None)
    zero_point = getattr(layer, "weight_zero_point", None)
    observer = getattr(layer, "weight_observer", None)

    if weight is None or observer is None or scale is None or zero_point is None:
        # scale, zp, or observer not calibratable or weight not available
        return

    updated_scale, updated_zero_point = observer(weight)

    # update scale and zero point
    device = next(layer.parameters()).device
    scale.data = updated_scale.to(device)
    zero_point.data = updated_zero_point.to(device)


def enable_quantization(module: Module):
    module.quantization_enabled = True


def disable_quantization(module: Module):
    module.quantization_enabled = False


def safe_permute(value: torch.Tensor, perm: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Perform out-of-place permutation without using torch.Tensor.index_put_,
    whose implementation is missing for datatypes such as `torch.float8_e4m3fn`

    :param value: tensor to permute
    :param perm: permutation map
    :param dim: dimension along which to apply permutation
    :return: permuted value
    """
    preceding_dims = [slice(None)] * dim
    if value.dtype not in EXPERIMENTAL_DTYPES:
        return value[tuple(preceding_dims + [perm])]

    else:
        value_ret = torch.zeros_like(value)

        for index, perm_index in enumerate(perm):
            value_ret[tuple(preceding_dims + [index])] = value[
                tuple(preceding_dims + [perm_index])
            ]

        return value_ret
