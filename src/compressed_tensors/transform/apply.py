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

from collections import defaultdict
from typing import List, Tuple

import torch
from compressed_tensors import TRANSFORM_CONFIG_NAME
from compressed_tensors.transform import TransformConfig, TransformFactory


__all__ = ["apply_transform_config"]


def apply_transform_config(model: torch.nn.Module, config: TransformConfig):
    """
    Apply a transform config to a model. Weight transforms are fused into weights, while
    activation transforms are attached as submodules and trigger via pytorch hooks

    :param model: model to apply config to
    :param config: transform config to apply
    """
    for name, scheme in config.config_groups.items():
        factory = TransformFactory.from_scheme(scheme, name=name)
        factory.apply_to_model(model)

    # attach config to model for compression/serialization
    setattr(model, TRANSFORM_CONFIG_NAME, config)

    # ensure that tied weight transforms can be serialized without aliases
    # In the future, this could be done by transformers or model compressor
    # which would make this more robust to changing dispatches after transforms
    _tie_offloaded_tensors(model)


def _tie_offloaded_tensors(model: torch.nn.Module):
    """
    Populate the `_dynamic_tied_weights_keys` attribute of transforms,
    which is used by transformers to detect and remove shared pointers
    during saving
    """
    from compressed_tensors.utils import has_offloaded_params

    # map from  to keys
    offloaded_ptrs: dict[int, List[Tuple[torch.nn.Module, str]]] = defaultdict(list)
    for module in model.modules():
        # NOTE: previously asserted that parent._hf_hook.place_submodules=False
        if has_offloaded_params(module):
            for key, _ in module.named_parameters(recurse=False):
                param = module._hf_hook.weights_map[key]
                offloaded_ptrs[id(param)].append((module, key))

    # populate `_dynamic_tied_weights_keys` if there is more than one key
    # and ensure that they share tensors. In the case of offloading, this
    for shared_keys in offloaded_ptrs.values():
        if len(shared_keys) > 1:
            first_tensor = getattr(shared_keys[0][0], shared_keys[0][1])
            assert first_tensor.device.type == "meta"
            for module, key in shared_keys:
                assert getattr(module, key).device.type == "meta"
                setattr(module, key, first_tensor)
