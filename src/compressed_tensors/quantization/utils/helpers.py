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
from typing import Generator, List, Optional, Tuple

import torch
from compressed_tensors.quantization.quant_args import (
    FP4_E2M1_DATA,
    FP8_E4M3_DATA,
    FloatArgs,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from torch import FloatTensor, IntTensor, Tensor
from torch.nn import Module
from tqdm import tqdm


__all__ = [
    "infer_quantization_status",
    "is_module_quantized",
    "is_model_quantized",
    "module_type",
    "calculate_compression_ratio",
    "get_torch_bit_depth",
    "can_quantize",
    "parse_out_kv_cache_args",
    "KV_CACHE_TARGETS",
    "is_kv_cache_quant_scheme",
    "iter_named_leaf_modules",
    "iter_named_quantizable_modules",
    "compute_dynamic_scales_and_zp",
    "calculate_range",
    "calculate_qparams",
    "generate_gparam",
    "is_fp4",
]

# target the self_attn layer
# QuantizedKVParameterCache is responsible for obtaining the k_scale and v_scale
KV_CACHE_TARGETS = ["re:.*self_attn$"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


def is_fp4(quantization_args: QuantizationArgs):
    return (
        quantization_args.num_bits == 4
        and quantization_args.type == QuantizationType.FLOAT
    )


def calculate_qparams(
    min_vals: Tensor,
    max_vals: Tensor,
    quantization_args: QuantizationArgs,
    global_scale: Optional[Tensor] = None,
) -> Tuple[FloatTensor, IntTensor]:
    """
    :param min_vals: tensor of min value(s) to calculate scale(s) and zero point(s)
        from
    :param max_vals: tensor of max value(s) to calculate scale(s) and zero point(s)
        from
    :param quantization_args: settings to quantization
    :param global_scale: additional global scale to scale the locally generated scale
        currently only applied/supported for Fp4

    :return: tuple of the calculated scale(s) and zero point(s). For FP4, the calculated
        scale is of dtype FP8
    """
    # based on the implementations for consuming quantized values,
    # 0.0 must always be representable within the quantized range
    min_vals = torch.min(min_vals, torch.zeros_like(min_vals))
    max_vals = torch.max(max_vals, torch.zeros_like(max_vals))

    device = min_vals.device

    bit_min, bit_max = calculate_range(quantization_args, device)
    bit_range = bit_max - bit_min

    if is_fp4(quantization_args=quantization_args):
        zp_dtype = FP8_E4M3_DATA.dtype
    else:
        zp_dtype = quantization_args.pytorch_dtype()

    if quantization_args.symmetric:
        max_val_pos = torch.max(torch.abs(min_vals), torch.abs(max_vals))

        if is_fp4(quantization_args=quantization_args) and global_scale is not None:
            # Conditionally scale the generated local scale by a global_scale
            scales = global_scale * (max_val_pos / FP4_E2M1_DATA.max)
            scales = torch.clamp(scales, max=FP8_E4M3_DATA.max, min=FP8_E4M3_DATA.min)
            scales = scales.to(FP8_E4M3_DATA.dtype)

        else:
            scales = max_val_pos / (float(bit_range) / 2)

        # TODO: in the case of MoEs, the global_scale may also be 0/need to be clamped
        if scales.dtype == FP8_E4M3_DATA.dtype:
            # torch.clamp not supported for FP8
            # use the next largest fp8 value from 0
            scales = torch.where(
                scales == 0,
                torch.tensor(0.125, dtype=FP8_E4M3_DATA.dtype, device=device),
                scales,
            )
        else:
            scales = torch.clamp(scales, min=torch.finfo(torch.float32).eps)

        zero_points = torch.zeros(scales.shape, device=device, dtype=min_vals.dtype)
    else:
        if is_fp4(quantization_args=quantization_args):
            raise NotImplementedError(
                "Asymmetric Quantization is not supported for FP4"
            )

        scales = (max_vals - min_vals) / float(bit_range)
        scales = torch.clamp(scales, min=torch.finfo(torch.float32).eps)
        zero_points = bit_min - (min_vals / scales)
        zero_points = torch.clamp(zero_points, bit_min, bit_max)

    # match zero-points to quantized type
    # if casting to int, use round instead of truncate
    if quantization_args.type == QuantizationType.INT:
        zero_points = torch.round(zero_points)
    zero_points = zero_points.to(zp_dtype)

    if scales.ndim == 0:
        scales = scales.reshape(1)
        zero_points = zero_points.reshape(1)

    return scales, zero_points


def compute_dynamic_scales_and_zp(
    value: Tensor,
    args: QuantizationArgs,
    module: torch.nn.Module,
    global_scale: Optional[Tensor] = None,
):
    """
    Returns the computed scales and zero points for dynamic activation
    quantization.

    :param value: tensor to calculate quantization parameters for
    :param args: quantization args
    :param reduce_dims: optional tuple of dimensions to reduce along,
        returned scale and zero point will be shaped (1,) along the
        reduced dimensions
    :return: tuple of scale and zero point derived from the observed tensor
    """

    keep_dims = True
    if args.strategy == QuantizationStrategy.TOKEN:
        dim = {1, 2}
        reduce_dims = tuple(idx for idx in range(value.ndim) if idx not in dim)
    elif args.strategy == QuantizationStrategy.TENSOR:
        reduce_dims = None
    elif args.strategy == QuantizationStrategy.TENSOR_GROUP:
        if len(value.shape) > 2:
            value = value.squeeze(0)

        dim = {0, 1}
        reduce_dims = tuple(idx for idx in range(3) if idx not in dim)
        keep_dims = False
        value = torch.reshape(
            value,
            (
                value.shape[0],
                math.ceil(value.shape[1] / args.group_size),
                args.group_size,
            ),
        )
    else:
        raise ValueError(
            "Dynamic quantization is only supported for ",
            f"{QuantizationStrategy.TOKEN, QuantizationStrategy.TENSOR, QuantizationStrategy.TENSOR_GROUP}",
        )

    if not reduce_dims:
        min_val, max_val = torch.aminmax(value)
    else:
        min_val = torch.amin(value, dim=reduce_dims, keepdims=keep_dims)
        max_val = torch.amax(value, dim=reduce_dims, keepdims=keep_dims)

    return calculate_qparams(min_val, max_val, args, global_scale=global_scale)


def calculate_range(quantization_args: QuantizationArgs, device: str) -> Tuple:
    """
    Calculated the effective quantization range for the given Quantization Args

    :param quantization_args: quantization args to get range of
    :param device: device to store the range to
    :return: tuple endpoints for the given quantization range
    """
    if quantization_args.type == QuantizationType.INT:
        bit_range = 2**quantization_args.num_bits
        q_max = torch.tensor(bit_range / 2 - 1, device=device)
        q_min = torch.tensor(-bit_range / 2, device=device)
    elif quantization_args.type == QuantizationType.FLOAT:
        if quantization_args.num_bits == 8:
            q_max = torch.tensor(FP8_E4M3_DATA.max, device=device)
            q_min = torch.tensor(FP8_E4M3_DATA.min, device=device)
        elif quantization_args.num_bits == 4:
            q_max = torch.tensor(FP4_E2M1_DATA.max, device=device)
            q_min = torch.tensor(FP4_E2M1_DATA.min, device=device)
        else:
            raise NotImplementedError(
                "Range calculation only supported for 4 and 8 bits"
            )
    else:
        raise ValueError(f"Invalid quantization type {quantization_args.type}")

    return q_min, q_max


def infer_quantization_status(model: Module) -> Optional["QuantizationStatus"]:  # noqa
    """
    Checks the quantization status of a model. Assumes all modules in the model have
    the same status, so only the first quantized model is checked.

    :param model: model to check quantization status for
    :return: quantization status if the model is quantized, otherwise None
    """
    for module in model.modules():
        status = getattr(module, "quantization_status", None)
        if status is not None:
            return status
    return None


def is_module_quantized(module: Module) -> bool:
    """
    Check if a module is quantized, based on the existence of a non-empty quantization
    scheme

    :param module: pytorch module to check
    :return: True if module is quantized, False otherwise
    """
    if not hasattr(module, "quantization_scheme"):
        return False

    if module.quantization_scheme.weights is not None:
        return True

    if module.quantization_scheme.input_activations is not None:
        return True

    if module.quantization_scheme.output_activations is not None:
        return True

    return False


def is_model_quantized(model: Module) -> bool:
    """
    Check if any modules in a model are quantized, based on the existence of a non-empty
    quantization scheme in at least one module

    :param model: pytorch model
    :return: True if model is quantized, False otherwise
    """

    for _, submodule in iter_named_leaf_modules(model):
        if is_module_quantized(submodule):
            return True

    return False


def module_type(module: Module) -> str:
    """
    Gets a string representation of a module type

    :module: pytorch module to get type of
    :return: module type as a string
    """
    return type(module).__name__


def iter_named_leaf_modules(model: Module) -> Generator[Tuple[str, Module], None, None]:
    """
    Yields modules that do not have any submodules except observers. The observers
    themselves are not yielded
    :param model: model to get leaf modules of
    :returns: generator tuple of (name, leaf_submodule)
    """
    for name, submodule in model.named_modules():
        children = list(submodule.children())
        # TODO: verify if an observer would ever be attached in this case/remove check
        if len(children) == 0 and "observer" in name:
            yield name, submodule
        else:
            if len(children) > 0:
                named_children, children = zip(*list(submodule.named_children()))
            has_non_observer_children = False
            for i in range(len(children)):
                child_name = named_children[i]

                if "observer" not in child_name:
                    has_non_observer_children = True

            if not has_non_observer_children:
                yield name, submodule


def iter_named_quantizable_modules(
    model: Module,
    include_children: bool = True,
    include_attn: bool = False,
    include_mlp: bool = False,
) -> Generator[Tuple[str, Module], None, None]:
    """
    Yield name and submodule of
    - leaf modules, set by include_children
    - attention modyles, set by include_attn

    :param model: model to get leaf modules of
    :param include_children: flag to get the leaf modules
    :param inlcude_attn: flag to get the attention modules
    :returns: generator tuple of (name, submodule)
    """
    for name, submodule in model.named_modules():
        # TODO: verify if an observer would ever be attached in this case/remove check
        if include_children:
            children = list(submodule.children())
            if len(children) == 0 and "observer" not in name:
                yield name, submodule
            else:
                if len(children) > 0:
                    named_children, children = zip(*list(submodule.named_children()))
                has_non_observer_children = False
                for i in range(len(children)):
                    child_name = named_children[i]

                    if "observer" not in child_name:
                        has_non_observer_children = True

                if not has_non_observer_children:
                    yield name, submodule
        if include_attn:
            if name.endswith("self_attn"):
                yield name, submodule
        if include_mlp:
            if name.endswith("mlp"):
                yield name, submodule


def get_torch_bit_depth(value: torch.Tensor) -> int:
    """
    Determine the number of bits used to represent the dtype of a tensor

    :param value: tensor to check bit depth of
    :return: bit depth of each element in the value tensor
    """
    try:
        bit_depth = torch.finfo(value.dtype).bits
    except TypeError:
        bit_depth = torch.iinfo(value.dtype).bits

    return bit_depth


def can_quantize(value: torch.Tensor, quant_args: "QuantizationArgs") -> bool:  # noqa
    """
    Checks if value can be quantized by quant_args.

    :param value: tensor to check for quantization
    :param quant_args: QuantizationArgs to use for quantization
    :return: False if value is already quantized to quant_args or value is incompatible
    with quant_args, True if value can be quantized with quant_args
    """
    bit_depth = get_torch_bit_depth(value)
    requested_depth = quant_args.num_bits
    if bit_depth < quant_args.num_bits:
        _LOGGER.warn(
            f"Can't quantize tensor with bit depth {bit_depth} to {requested_depth}."
            "The QuantizationArgs provided are not compatible with the input tensor."
        )

    return bit_depth > quant_args.num_bits


def calculate_compression_ratio(model: Module) -> float:
    """
    Calculates the quantization compression ratio of a pytorch model, based on the
    number of bits needed to represent the total weights in compressed form. Does not
    take into account activation quantizatons.

    :param model: pytorch module to calculate compression ratio for
    :return: compression ratio of the whole model
    """
    total_compressed = 0.0
    total_uncompressed = 0.0
    for name, submodule in tqdm(
        iter_named_leaf_modules(model),
        desc="Calculating quantization compression ratio",
    ):
        for parameter in model.parameters():
            uncompressed_bits = get_torch_bit_depth(parameter)
            compressed_bits = uncompressed_bits
            if is_module_quantized(submodule) and submodule.quantization_scheme.weights:
                compressed_bits = submodule.quantization_scheme.weights.num_bits

            num_weights = parameter.numel()
            total_compressed += compressed_bits * num_weights
            total_uncompressed += uncompressed_bits * num_weights

    return total_uncompressed / total_compressed


def is_kv_cache_quant_scheme(scheme: QuantizationScheme) -> bool:
    """
    Check whether the QuantizationScheme targets the kv cache.
    It does if all the following criteria are met:
    - the scheme targets either exactly match the KV_CACHE_TARGETS
        or the match KV_CACHE_TARGETS regex pattern
    - the scheme quantizes output_activations (we want to quantize the
        outputs from the KV_CACHE_TARGETS, as their correspond to the
        keys and values that are to be saved in the cache)

    :param scheme: The QuantizationScheme to investigate
    :return: boolean flag
    """
    for target in scheme.targets:
        if target in KV_CACHE_TARGETS:
            return True

    return False


def parse_out_kv_cache_args(
    quant_scheme_to_layers: List[QuantizationScheme],
) -> Tuple[Optional[QuantizationArgs], List[QuantizationScheme]]:
    """
    If possible, parse out the kv cache specific QuantizationArgs
    from the list of the QuantizationSchemes. If no kv cache
    specific QuantizationArgs available, this function acts
    as an identity function

    :param quant_scheme_to_layers: list of QuantizationSchemes
    :return: kv_cache_args (optional) and the (remaining or original)
        list of the QuantizationSchemes
    """
    kv_cache_quant_scheme_to_layers = [
        scheme for scheme in quant_scheme_to_layers if is_kv_cache_quant_scheme(scheme)
    ]
    quant_scheme_to_layers = [
        scheme
        for scheme in quant_scheme_to_layers
        if not is_kv_cache_quant_scheme(scheme)
    ]

    if kv_cache_quant_scheme_to_layers:
        kv_cache_quant_scheme_to_layers = kv_cache_quant_scheme_to_layers[0]
        kv_cache_args = kv_cache_quant_scheme_to_layers.output_activations
    else:
        kv_cache_args = None

    return kv_cache_args, quant_scheme_to_layers


def generate_gparam(
    updated_min_val: torch.Tensor,
    updated_max_val: torch.Tensor,
    scale_data: Optional[FloatArgs] = FP8_E4M3_DATA,
    quant_data: Optional[FloatArgs] = FP4_E2M1_DATA,
    dtype: Optional[torch.dtype] = torch.float32,
):
    """
    Generate a global scale for an entire tensor (input_tensor).
    Goal of the scale is to ensure that the quantization (local) scale
    falls into the approproiate dtype range.

    E.g. for NVFP4, group (local) scales are in dtype FP8. The global_scale
    attempts to use the entire FP8 dtype range while mapping a per-group max
    to the FP4 max.
    """
    min_vals = torch.min(updated_min_val, torch.zeros_like(updated_min_val))
    max_vals = torch.max(updated_max_val, torch.zeros_like(updated_max_val))
    max_val_pos = torch.max(torch.abs(min_vals), torch.abs(max_vals))
    global_scale = scale_data.max * quant_data.max / max_val_pos
    return global_scale.to(dtype).reshape([1])
