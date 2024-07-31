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


from typing import Optional

import torch
from torch.nn import Module


__all__ = [
    "update_layer_weight_quant_params",
    "enable_quantization",
    "disable_quantization",
]


def update_layer_weight_quant_params(
    layer: Module, g_idx: Optional[torch.Tensor] = None, perm: Optional[torch.Tensor] = None,
):
    weight = getattr(layer, "weight", None)
    scale = getattr(layer, "weight_scale", None)
    zero_point = getattr(layer, "weight_zero_point", None)
    observer = getattr(layer, "weight_observer", None)

    if weight is None or observer is None or scale is None or zero_point is None:
        # scale, zp, or observer not calibratable or weight not available
        return

    if perm is not None:
        weight= weight[:, perm]

    # breakpoint()
    # updated_scale, updated_zero_point = observer(weight)
    updated_scale, updated_zero_point = observer(weight, g_idx=g_idx)

    # update scale and zero point
    device = next(layer.parameters()).device
    scale.data = updated_scale.to(device)
    zero_point.data = updated_zero_point.to(device)
    """
    Parameter containing:
tensor([[0.0047, 0.0044, 0.0076,  ..., 0.0063, 0.0059, 0.0046],
        [0.0055, 0.0143, 0.0060,  ..., 0.0104, 0.0098, 0.0049],
        [0.0043, 0.0066, 0.0060,  ..., 0.0048, 0.0061, 0.0051],
        ...,
        [0.0077, 0.0062, 0.0059,  ..., 0.0069, 0.0115, 0.0045],
        [0.0070, 0.0058, 0.0058,  ..., 0.0063, 0.0118, 0.0045],
        [0.0071, 0.0058, 0.0058,  ..., 0.0064, 0.0120, 0.0045]],
       device='cuda:0', dtype=torch.bfloat16)"""


def enable_quantization(module: Module):
    module.quantization_enabled = True


def disable_quantization(module: Module):
    module.quantization_enabled = False
