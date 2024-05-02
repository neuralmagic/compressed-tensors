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

from typing import Optional, Tuple

import torch
from torch.nn import Module
from tqdm import tqdm


__all__ = [
    "infer_quantization_status",
    "is_module_quantized",
    "is_model_quantized",
    "iter_named_leaf_modules",
    "module_type",
    "calculate_compression_ratio",
    "get_torch_bit_depth",
]


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


def iter_named_leaf_modules(model: Module) -> Tuple[str, Module]:
    # yields modules that do not have any submodules
    # TODO: potentially expand to add list of allowed submodules such as observers
    for name, submodule in model.named_modules():
        if len(list(submodule.children())) == 0:
            yield name, submodule


def get_torch_bit_depth(value: torch.Tensor):
    try:
        bit_depth = torch.finfo(value.dtype).bits
    except TypeError:
        bit_depth = torch.iinfo(value.dtype).bits

    return bit_depth


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
            if is_module_quantized(submodule):
                compressed_bits = submodule.quantization_scheme.weights.num_bits
            num_weights = parameter.numel()
            total_compressed += compressed_bits * num_weights
            total_uncompressed += uncompressed_bits * num_weights

    return total_uncompressed / total_compressed
