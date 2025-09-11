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
from collections import OrderedDict
from copy import deepcopy
from typing import List
from typing import OrderedDict as OrderedDictType
from typing import Union

import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization.lifecycle.initialize import (
    initialize_module_for_quantization,
)
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.quant_config import (
    QuantizationConfig,
    QuantizationStatus,
)
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from compressed_tensors.quantization.utils import (
    KV_CACHE_TARGETS,
    is_kv_cache_quant_scheme,
)
from compressed_tensors.utils.helpers import replace_module
from compressed_tensors.utils.match import match_named_modules, match_targets
from torch.nn import Module


__all__ = [
    "apply_quantization_config",
]


_LOGGER = logging.getLogger(__name__)


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

    # build mapping of targets to schemes for easier matching
    # use ordered dict to preserve target ordering in config
    target_to_scheme = OrderedDict()
    for scheme in config.config_groups.values():
        for target in scheme.targets:
            target_to_scheme[target] = scheme

    # mark appropriate layers for quantization by setting their quantization schemes
    for name, submodule in match_named_modules(
        model, target_to_scheme, config.ignore, warn_on_fail=True
    ):
        # mark modules to be quantized by adding
        # quant scheme to the matching layers
        matched_targets = match_targets(name, submodule, target_to_scheme)
        scheme = _scheme_from_targets(target_to_scheme, matched_targets, name)
        # target matched - add layer and scheme to target list
        submodule.quantization_scheme = scheme

        # replace with run compressed if applicable
        # FUTURE: move this to model compressor
        if (
            run_compressed
            and isinstance(submodule, torch.nn.Linear)
            and config.format != CompressionFormat.dense.value
        ):
            # TODO: expand to more module types
            compressed_linear = CompressedLinear.from_linear(
                submodule,
                quantization_scheme=scheme,
                quantization_format=config.format,
            )
            replace_module(model, name, compressed_linear)

        else:
            initialize_module_for_quantization(
                submodule,
                force_zero_point=config.quantization_status
                != QuantizationStatus.COMPRESSED,
            )

        submodule.quantization_status = config.quantization_status


def process_quantization_config(config: QuantizationConfig) -> QuantizationConfig:
    """
    Preprocess the raw QuantizationConfig

    :param config: the raw QuantizationConfig
    :return: the processed QuantizationConfig
    """
    if config.kv_cache_scheme is not None:
        config = process_kv_cache_config(config)

    return config


def process_kv_cache_config(
    config: QuantizationConfig, targets: Union[List[str], str] = KV_CACHE_TARGETS
) -> QuantizationConfig:
    """
    Reformulate the `config.kv_cache` as a `config_group`
    and add it to the set of existing `config.groups`

    :param config: the QuantizationConfig
    :return: the QuantizationConfig with additional "kv_cache" group
    """
    if targets == KV_CACHE_TARGETS:
        _LOGGER.info(f"KV cache targets set to default value of: {KV_CACHE_TARGETS}")

    kv_cache_dict = config.kv_cache_scheme.model_dump()
    kv_cache_scheme = QuantizationScheme(
        output_activations=QuantizationArgs(**kv_cache_dict),
        targets=targets,
    )
    kv_cache_group = dict(kv_cache=kv_cache_scheme)
    config.config_groups.update(kv_cache_group)
    return config


def _scheme_from_targets(
    target_to_scheme: OrderedDictType[str, QuantizationScheme],
    targets: List[str],
    name: str,
) -> QuantizationScheme:
    if len(targets) == 1:
        # if `targets` iterable contains a single element
        # use it as the key
        return target_to_scheme[targets[0]]

    # otherwise, we need to merge QuantizationSchemes corresponding
    # to multiple targets. This is most likely because `name` module
    # is being target both as an ordinary quantization target, as well
    # as kv cache quantization target
    schemes_to_merge = [target_to_scheme[target] for target in targets]
    return _merge_schemes(schemes_to_merge, name)


def _merge_schemes(
    schemes_to_merge: List[QuantizationScheme], name: str
) -> QuantizationScheme:
    kv_cache_quantization_scheme = [
        scheme for scheme in schemes_to_merge if is_kv_cache_quant_scheme(scheme)
    ]
    if not kv_cache_quantization_scheme:
        # if the schemes_to_merge do not contain any
        # kv cache QuantizationScheme
        # return the first scheme (the prioritized one,
        # since the order of schemes_to_merge matters)
        return schemes_to_merge[0]
    else:
        # fetch the kv cache QuantizationScheme and the highest
        # priority non-kv cache QuantizationScheme and merge them
        kv_cache_quantization_scheme = kv_cache_quantization_scheme[0]
        quantization_scheme = [
            scheme
            for scheme in schemes_to_merge
            if not is_kv_cache_quant_scheme(scheme)
        ][0]
        schemes_to_merge = [kv_cache_quantization_scheme, quantization_scheme]
        merged_scheme = {}
        for scheme in schemes_to_merge:
            scheme_dict = {
                k: v for k, v in scheme.model_dump().items() if v is not None
            }
            # when merging multiple schemes, the final target will be
            # the `name` argument - hence erase the original targets
            del scheme_dict["targets"]
            # make sure that schemes do not "clash" with each other
            overlapping_keys = set(merged_scheme.keys()) & set(scheme_dict.keys())
            if overlapping_keys:
                raise ValueError(
                    f"The module: {name} is being modified by two clashing "
                    f"quantization schemes, that jointly try to override "
                    f"properties: {overlapping_keys}. Fix the quantization config "
                    "so that it is not ambiguous."
                )
            merged_scheme.update(scheme_dict)

        merged_scheme.update(targets=[name])

        return QuantizationScheme(**merged_scheme)
