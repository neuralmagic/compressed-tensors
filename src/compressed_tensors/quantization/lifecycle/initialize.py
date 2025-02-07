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
from enum import Enum
from typing import Optional, Any

import torch
from compressed_tensors.quantization.lifecycle.forward import (
    wrap_module_forward_quantized,
)
from compressed_tensors.quantization.quant_args import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.quant_config import QuantizationStatus
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from compressed_tensors.quantization.utils import is_kv_cache_quant_scheme
from compressed_tensors.utils import (
    disable_hf_hook,
    has_offloaded_params,
    register_offload_parameter,
)
from torch.nn import Module, Parameter
from compressed_tensors.utils.offload import update_parameter_data


__all__ = [
    "initialize_module_for_quantization",
    "is_attention_module",
    "KVCacheScaleType",
    "TransformModule"
]


_LOGGER = logging.getLogger(__name__)


class KVCacheScaleType(Enum):
    KEY = "k_scale"
    VALUE = "v_scale"


"""
Pros:
- Same interface for quantized layer vs not quantized layers
- Easy to understand graph
- Allows easy way to manipulate forward pass 
- Easy to target non-weight parameters with transforms

Cons:
- Wrapping is annoying when it comes to overall module handling
- We will still need to add a specific transform check/using the TransformData 
when applying transforms to weights during weight-only quantization 
"""
class TransformModule(Module):
    def __init__(self, module: Module, transforms: dict):
        super(TransformModule, self).__init__()
        self.transforms = transforms
        self.module = module
    
    def forward(self, *args, **kwargs):
        # TODO: need a way to remove transformable layers at the end/folding in/applying the compressor
        current_module = self.module
        input_ = args[0]

        weight_transform = self.transforms.get("weight")
        input_transform = self.transforms.get("input_activations")
        output_transform = self.transforms.get("output_activations")
        
        # TODO: do we apply this just once and then keep a copy of the transforms weight?
        # Generically, are parameters updated in place or is a copy used? both options?
        # Note: weights currently do not require calibration data and are updated upfront
        
        if weight_transform:
            untransformed_weight = current_module.weight.data.clone()
            transformed_weight = weight_transform(current_module.weight)
            current_module.weight.data.copy_(transformed_weight)

        if input_transform:
            input_ = input_transform(input_)

        # Generic parameter transform updates
        for name, parameter in current_module.named_parameters():
            if name == "weight":
                continue
            elif name in self.transforms:
                param_transform = self.transform.get(name)
                updated_param = param_transform(parameter)
                update_parameter_data(current_module, updated_param, name)
    
        x = current_module(input_, *args[1:], **kwargs)

        if output_transform:
            x = output_transform(x)

        if weight_transform:
            current_module.weight.data.copy_(untransformed_weight)

        # - Need to apply the inverse transformation

        return x


from dataclasses import dataclass

# For both methods, we would still apply this as a way to target the weights
@dataclass
class TransformData:
    transforms: dict

"""
Pros
- Simplicity when it comes to the modules as the module remains the same, no wrapping
- Hooks are easy to remove/disable, unlike wrappers, which makes compressing easy

Cons:
- Less clarity in terms of the graph
- Harder to target non-weight parameter with transforms 
- Variation between quantized layers (where we have hooks) and layers we do not quantize (we have no hooks/control over their fwd pass)
"""

# TODO: would have to update to apply the weight transform as well
def transform_pre_hook(module: Module, args: Any):
    args = args[0] if isinstance(args, tuple) else args
    transforms = getattr(module, "transforms", None)
    input_ = args
    if transforms:
        input_transform = transforms.transforms.get("input_activations")
        if input_transform:
            input_ = input_transform(input_)
    
    return input_ 

def transform_post_hook(module: Module, _args: Any, output: torch.Tensor):
    transforms = getattr(module, "transforms", None)
    output_ = output
    if transforms:
        output_transform = transforms.transforms.get("output_activations")
        if output_transform:
            output_ = output_transform(output_)
    return output_ 

# Alternate - Hooks?

def initialize_module_for_quantization(
    module: Module,
    scheme: Optional[QuantizationScheme] = None,
    force_zero_point: bool = True,
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
    """
    scheme = scheme or getattr(module, "quantization_scheme", None)
    if scheme is None:
        # no scheme passed and layer not targeted for quantization - skip
        return

    if is_attention_module(module):
        # quantized actions based on calltime status
        _initialize_attn_scales(module)

    else:

        if scheme.input_activations is not None:
            _initialize_scale_zero_point(
                module,
                "input",
                scheme.input_activations,
                force_zero_point=force_zero_point,
            )
        if scheme.weights is not None:
            if hasattr(module, "weight"):
                weight_shape = None
                if isinstance(module, torch.nn.Linear):
                    weight_shape = module.weight.shape
                _initialize_scale_zero_point(
                    module,
                    "weight",
                    scheme.weights,
                    weight_shape=weight_shape,
                    force_zero_point=force_zero_point,
                )
            else:
                _LOGGER.warning(
                    f"module type {type(module)} targeted for weight quantization but "
                    "has no attribute weight, skipping weight quantization "
                    f"for {type(module)}"
                )

        if scheme.output_activations is not None:
            if not is_kv_cache_quant_scheme(scheme):
                _initialize_scale_zero_point(
                    module, "output", scheme.output_activations
                )

        module.quantization_scheme = scheme
        module.quantization_status = QuantizationStatus.INITIALIZED

        with disable_hf_hook(module):
            # wrap forward call of module to perform
            # quantized actions based on calltime status
            wrap_module_forward_quantized(module, scheme)


def is_attention_module(module: Module):
    return "attention" in module.__class__.__name__.lower() and (
        hasattr(module, "k_proj")
        or hasattr(module, "v_proj")
        or hasattr(module, "qkv_proj")
    )


def _initialize_scale_zero_point(
    module: Module,
    base_name: str,
    quantization_args: QuantizationArgs,
    weight_shape: Optional[torch.Size] = None,
    force_zero_point: bool = True,
):
    if quantization_args.dynamic:
        return

    # begin on the same device as other parameters or cpu if offloaded.
    # in the offloaded case, there's no point moving tensors to the execution device
    # if they're going to be immediately offloaded by `register_offload_parameter`
    params_device = next(module.parameters()).device
    device = "cpu" if has_offloaded_params(module) else params_device

    # infer expected scale/zero point shape
    if quantization_args.strategy == QuantizationStrategy.TOKEN:
        expected_shape = (1, 1)
    else:
        expected_shape = 1

    if base_name == "weight" and weight_shape is not None:
        if quantization_args.strategy == QuantizationStrategy.CHANNEL:
            # (output_channels, 1)
            expected_shape = (weight_shape[0], 1)
        elif quantization_args.strategy == QuantizationStrategy.GROUP:
            num_groups = weight_shape[1] // quantization_args.group_size
            expected_shape = (weight_shape[0], max(num_groups, 1))

    scale_dtype = module.weight.dtype
    if scale_dtype not in [torch.float16, torch.bfloat16, torch.float32]:
        scale_dtype = torch.float16

    # initializes empty scale, zero point, and g_idx parameters for the module
    init_scale = Parameter(
        torch.empty(expected_shape, dtype=scale_dtype, device=device),
        requires_grad=False,
    )
    register_offload_parameter(module, f"{base_name}_scale", init_scale)

    if force_zero_point or not quantization_args.symmetric:
        zp_dtype = quantization_args.pytorch_dtype()
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


def _initialize_attn_scales(module: Module) -> None:
    """Initlaize k_scale, v_scale for  self_attn"""

    expected_shape = 1  # per tensor

    param = next(module.parameters())
    scale_dtype = param.dtype
    device = param.device

    init_scale = Parameter(
        torch.empty(expected_shape, dtype=scale_dtype, device=device),
        requires_grad=False,
    )

    module.register_parameter(KVCacheScaleType.KEY.value, init_scale)

    init_scale = Parameter(
        torch.empty(expected_shape, dtype=scale_dtype, device=device),
        requires_grad=False,
    )
    module.register_parameter(KVCacheScaleType.VALUE.value, init_scale)
