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
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization.lifecycle.compressed import (
    compress_quantized_weights,
)
from compressed_tensors.quantization.lifecycle.initialize import (
    initialize_module_for_quantization,
    is_attention_module,
)
from compressed_tensors.quantization.quant_config import (
    QuantizationConfig,
    QuantizationStatus,
)
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from compressed_tensors.quantization.utils import (
    ATTN_TARGETS,
    infer_quantization_status,
)
from compressed_tensors.utils import (
    deprecated,
    get_safetensors_folder,
    is_narrow_match,
    match_named_modules,
    match_targets,
    replace_module,
    update_parameter_data,
)
from safetensors import safe_open
from torch.nn import Module
from transformers import PreTrainedModel


__all__ = [
    "load_pretrained_quantization_parameters",
    "apply_quantization_config",
    "apply_quantization_status",
    "find_name_or_class_matches",
]

from compressed_tensors.quantization.utils.helpers import is_module_quantized
from compressed_tensors.utils.safetensors_load import (
    get_quantization_parameter_to_path_mapping,
)


_LOGGER = logging.getLogger(__name__)


def load_pretrained_quantization_parameters(
    model: Module,
    model_name_or_path: Optional[str] = None,
    load_weight_qparams: Optional[bool] = False,
):
    """
    Loads the quantization parameters (scale and zero point) from model_name_or_path to
    a model that has already been initialized with a quantization config.

    NOTE: Will always load inputs/output parameters. Will conditioanlly load weight
    parameters, if load_weight_qparams is set to True.

    :param model: model to load pretrained quantization parameters to
    :param model_name_or_path: Hugging Face stub or local folder containing a quantized
        model, which is used to load quantization parameters
    :param load_weight_qparams: whether or not the weight quantization parameters
        should be loaded
    """
    model_path = get_safetensors_folder(model_name_or_path)
    mapping = get_quantization_parameter_to_path_mapping(model_path)

    for name, submodule in model.named_modules():
        if not is_module_quantized(submodule):
            continue
        if submodule.quantization_scheme.input_activations is not None:
            base_name = "input"
            _load_quant_args_from_mapping(
                base_name=base_name,
                module_name=name,
                module=submodule,
                mapping=mapping,
            )
        if submodule.quantization_scheme.output_activations is not None:
            base_name = "output"
            _load_quant_args_from_mapping(
                base_name=base_name,
                module_name=name,
                module=submodule,
                mapping=mapping,
            )

        if load_weight_qparams and submodule.quantization_scheme.weights:
            base_name = "weight"
            _load_quant_args_from_mapping(
                base_name=base_name,
                module_name=name,
                module=submodule,
                mapping=mapping,
            )


def apply_quantization_config(
    model: Module, config: Union[QuantizationConfig, None], run_compressed: bool = False
):
    """
    Initializes the model for quantization in-place based on the given config.
    Optionally coverts quantizable modules to compressed_linear modules

    :param model: model to apply quantization config to
    :param config: quantization config
    :param run_compressed: Whether the model will be run in compressed mode or
        decompressed fully on load
    """
    from compressed_tensors.linear.compressed_linear import CompressedLinear

    config = deepcopy(config)
    if config is None:  # see PR #180
        return dict()

    # preprocess to support kv cache scheme
    config = process_quantization_config(config)

    # mark appropriate layers for quantization by setting their quantization schemes
    for scheme in config.config_groups.values():
        for name, submodule in match_named_modules(
            model, scheme.targets, config.ignore or [], warn_on_fail=True
        ):
            # attach scheme (overwrite any existing)
            setattr(submodule, "quantization_scheme", scheme)

            # replace with run compressed if applicable
            # FUTURE: move this to model compressor
            if isinstance(submodule, torch.nn.Linear):
                if run_compressed:
                    format = config.format
                    if format != CompressionFormat.dense.value:
                        # TODO: expand to more module types
                        compressed_linear = CompressedLinear.from_linear(
                            submodule,
                            quantization_scheme=scheme,
                            quantization_format=format,
                        )
                        replace_module(model, name, compressed_linear)

            # attention quantization and/or kv cache quantization
            if is_attention_module(submodule) and not is_narrow_match(
                model, scheme.targets, name
            ):
                # do not quantize attention unless specifically targeted
                delattr(submodule, "quantization_scheme")

    # apply current quantization status across all targeted linear/embedding layers
    apply_quantization_status(model, config.quantization_status)


def process_quantization_config(config: QuantizationConfig) -> QuantizationConfig:
    """
    Preprocess the raw QuantizationConfig

    :param config: the raw QuantizationConfig
    :return: the processed QuantizationConfig
    """
    if config.kv_cache_scheme is not None:
        config = process_kv_cache_config(config)

    return config


def process_kv_cache_config(config: QuantizationConfig) -> QuantizationConfig:
    """
    Reformulate the `config.kv_cache` as a `config_group`
    and add it to the set of existing `config.groups`

    :param config: the QuantizationConfig
    :return: the QuantizationConfig with additional "kv_cache" group
    """
    _LOGGER.info(f"KV cache targets set to default value of: {ATTN_TARGETS}")

    scheme = QuantizationScheme(
        targets=ATTN_TARGETS,
        input_activations=config.kv_cache_scheme,
        kv_cache_only=True,
    )
    config.config_groups.update({"kv_cache": scheme})
    return config


def apply_quantization_status(model: Module, status: QuantizationStatus):
    """
    Applies in place the quantization lifecycle up to the given status

    :param model: model to apply quantization to
    :param status: status to update the module to
    """

    current_status = infer_quantization_status(model)

    if status >= QuantizationStatus.INITIALIZED > current_status:
        force_zero_point_init = status != QuantizationStatus.COMPRESSED

        model.apply(
            lambda module: initialize_module_for_quantization(
                module,
                force_zero_point=force_zero_point_init,
                model=model,
            )
        )

    if current_status < status >= QuantizationStatus.COMPRESSED > current_status:
        model.apply(compress_quantized_weights)


@deprecated(
    message="This function is deprecated and will be removed in a future release."
    "Please use `match_targets` from `compressed_tensors.utils.match` instead."
)
def find_name_or_class_matches(
    name: str, module: Module, targets: Iterable[str], check_contains: bool = False
) -> List[str]:
    """
    Returns all targets that match the given name or the class name.
    Returns empty list otherwise.
    The order of the output `matches` list matters.
    The entries are sorted in the following order:
        1. matches on exact strings
        2. matches on regex patterns
        3. matches on module names
    """
    if check_contains:
        raise NotImplementedError(
            "This function is deprecated, and the check_contains=True option has been"
            " removed."
        )

    return match_targets(name, module, targets)


def _load_quant_args_from_mapping(
    base_name: str, module_name: str, module: Module, mapping: Dict
):
    # TODO: skip update and just register here, don't do it in initialize
    """
    Loads scale and zero point from a state_dict into the specified module

    :param base_name: quantization target, one of: weights, input_activations or
    output_activations
    :param module_name: pytorch module name to look up in state_dict
    :module: pytorch module associated with module_name
    :mapping: mapping to search fetch paths on disk for a given parameter
    """
    scale_name = f"{base_name}_scale"
    zp_name = f"{base_name}_zero_point"
    g_idx_name = f"{base_name}_g_idx"

    state_dict_scale_path = mapping.get(f"{module_name}.{scale_name}", None)
    state_dict_zp_path = mapping.get(f"{module_name}.{zp_name}", None)
    state_dict_g_idx_path = mapping.get(f"{module_name}.{g_idx_name}", None)

    if state_dict_g_idx_path is not None:
        with safe_open(state_dict_g_idx_path, framework="pt", device="cpu") as f:
            state_dict_g_idx = f.get_tensor(f"{module_name}.{g_idx_name}")

        update_parameter_data(module, state_dict_g_idx, g_idx_name)

    if state_dict_scale_path is not None:
        # module is quantized
        with safe_open(state_dict_scale_path, framework="pt", device="cpu") as f:
            state_dict_scale = f.get_tensor(f"{module_name}.{scale_name}")

        update_parameter_data(module, state_dict_scale, scale_name)

        if state_dict_zp_path is None:
            # fill in zero point for symmetric quantization
            state_dict_zp = torch.zeros_like(state_dict_scale, device="cpu")
        else:
            with safe_open(state_dict_zp_path, framework="pt", device="cpu") as f:
                state_dict_zp = f.get_tensor(f"{module_name}.{zp_name}")

        update_parameter_data(module, state_dict_zp, zp_name)
