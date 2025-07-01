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


import logging
import math
from enum import Enum
from typing import List, Optional

import torch
from compressed_tensors.quantization.lifecycle.forward import (
    wrap_module_forward_quantized,
)
from compressed_tensors.quantization.quant_args import (
    FP8_E4M3_DATA,
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.quant_config import QuantizationStatus
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from compressed_tensors.quantization.utils import is_fp4, is_kv_cache_quant_scheme
from compressed_tensors.utils import (
    disable_hf_hook,
    get_execution_device,
    register_offload_parameter,
)
from compressed_tensors.utils.helpers import patch_attr
from torch.nn import Module, Parameter
from transformers.configuration_utils import PretrainedConfig


__all__ = [
    "initialize_module_for_quantization",
    "is_attention_module",
]


def initialize_module_for_quantization(
    module: Module,
    scheme: Optional[QuantizationScheme] = None,
    force_zero_point: bool = True,
    scale_dtype: Optional[torch.dtype] = None,
):
    """
    attaches appropriate scales, zero points, and observers to a layer
    given its target quantization scheme

    apply to full model with `model.apply(initialize_module_for_quantization)`

    :param module: module to set for calibration
    :param scheme: scheme to use for quantization. if None is provided,
        will attempt to use scheme stored in the module under `quantization_scheme`,
        if not provided, the layer will be skipped
    :param force_zero_point: whether to force initialization of a zero point for
        symmetric quantization
    :param scale_dtype: dtype to used for the scales, if overriding the
        weight dtype as the scale dtype
    """
    # TODO: don't initialize parameters when running decompression
    scheme = scheme or getattr(module, "quantization_scheme", None)
    if scheme is None:
        return

    # initialize scheme and status
    module.quantization_scheme = scheme
    module.quantization_status = QuantizationStatus.INITIALIZED

    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        if scheme.input_activations is not None:
            _initialize_quantization_parameters(
                module,
                "input",
                scheme.input_activations,
                force_zero_point=force_zero_point,
                scale_dtype=scale_dtype,
            )

        if scheme.weights is not None:
            _initialize_quantization_parameters(
                module,
                "weight",
                scheme.weights,
                force_zero_point=force_zero_point,
                scale_dtype=scale_dtype,
            )

        if scheme.output_activations is not None:
            _initialize_quantization_parameters(
                module,
                "output",
                scheme.output_activations,
                force_zero_point=force_zero_point,
                scale_dtype=scale_dtype,
            )

        with disable_hf_hook(module):
            # wrap forward call of module to perform
            # quantized actions based on calltime status
            wrap_module_forward_quantized(module, scheme)

    elif is_attention_module(module):
        assert scheme.input_activations is not None
        for base_name in ("q", "k", "v"):
            _initialize_quantization_parameters(
                module,
                base_name,
                scheme.input_activations,
                force_zero_point=force_zero_point,
                scale_dtype=scale_dtype,
            )

    else:
        raise ValueError(f"Unsupported quantization target {type(module)}")


def is_attention_module(module: Module):
    # can redefine to inspect source code for references to ALL_ATTENTION_FUNCTIONS
    return "attention" in module.__class__.__name__.lower() and (
        hasattr(module, "k_proj")
        or hasattr(module, "v_proj")
        or hasattr(module, "qkv_proj")
    )


def _initialize_quantization_parameters(
    module: Module,
    base_name: str,
    quantization_args: QuantizationArgs,
    force_zero_point: bool = True,
    scale_dtype: Optional[torch.dtype] = None,
):
    if quantization_args.dynamic is True:
        return

    # initialize on execution device to avoid performing quantized ops on cpu
    device = get_execution_device(module)

    # 1. Create global_scales for tensor_group - generates
    # a per tensor scale
    if quantization_args.strategy == QuantizationStrategy.TENSOR_GROUP:
        init_global_scale = Parameter(
            torch.empty(1, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        register_offload_parameter(
            module, f"{base_name}_global_scale", init_global_scale
        )

    # 2. Infer expected scale/zero point shape
    if quantization_args.strategy == QuantizationStrategy.TOKEN:
        expected_shape = (1, 1)
    else:
        expected_shape = 1

    if base_name == "weight":
        weight_shape = getattr(module, "weight").shape
        if quantization_args.strategy == QuantizationStrategy.CHANNEL:
            # (output_channels, 1)
            expected_shape = (weight_shape[0], 1)
        elif quantization_args.strategy in (
            QuantizationStrategy.TENSOR_GROUP,
            QuantizationStrategy.GROUP,
        ):
            num_groups = math.ceil(weight_shape[1] / quantization_args.group_size)
            expected_shape = (weight_shape[0], max(num_groups, 1))

    # 3. Identify quantization scale and zp dtype
    scale_dtype = scale_dtype if scale_dtype is not None else module.weight.dtype

    if is_fp4(quantization_args=quantization_args):
        scale_dtype = zp_dtype = FP8_E4M3_DATA.dtype
    else:
        # TODO: consider erroring out in the future as if the dtype if not one of these,
        # there is likely bug
        if scale_dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            scale_dtype = torch.float16
        zp_dtype = quantization_args.pytorch_dtype()

    # 4. Initializes empty scale, zero point, and g_idx parameters for the module
    # do not init scales for quantzation_args.dynamic == DynamicType.local
    if not quantization_args.dynamic:
        init_scale = Parameter(
            torch.empty(expected_shape, dtype=scale_dtype, device=device),
            requires_grad=False,
        )
        register_offload_parameter(module, f"{base_name}_scale", init_scale)

    if force_zero_point or not quantization_args.symmetric:
        init_zero_point = Parameter(
            torch.zeros(expected_shape, device=device, dtype=zp_dtype),
            requires_grad=False,
        )
        register_offload_parameter(module, f"{base_name}_zero_point", init_zero_point)

    # only grouped activation ordering has g_idx
    if quantization_args.actorder == ActivationOrdering.GROUP:
        g_idx_shape = (weight_shape[1],)
        g_idx_dtype = torch.int
        init_g_idx = Parameter(
            torch.full(g_idx_shape, -1, device=device, dtype=g_idx_dtype),
            requires_grad=False,
        )
        register_offload_parameter(module, f"{base_name}_g_idx", init_g_idx)
