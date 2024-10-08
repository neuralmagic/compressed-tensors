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
from typing import Generator, List, Optional, Tuple

import torch
from compressed_tensors.quantization.observers.base import Observer
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
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
]

# target the self_attn layer
# QuantizedKVParameterCache is responsible for obtaining the k_scale and v_scale
KV_CACHE_TARGETS = ["re:.*self_attn$"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


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
        if len(children) == 0 and not isinstance(submodule, Observer):
            yield name, submodule
        else:
            has_non_observer_children = False
            for child in children:
                if not isinstance(child, Observer):
                    has_non_observer_children = True

            if not has_non_observer_children:
                yield name, submodule


def iter_named_quantizable_modules(
    model: Module, include_children: bool = True, include_attn: bool = False
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
        if include_children:
            children = list(submodule.children())
            if len(children) == 0 and not isinstance(submodule, Observer):
                yield name, submodule
            else:
                has_non_observer_children = False
                for child in children:
                    if not isinstance(child, Observer):
                        has_non_observer_children = True

                if not has_non_observer_children:
                    yield name, submodule
        if include_attn:
            if name.endswith("self_attn"):
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
