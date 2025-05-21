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

from typing import TYPE_CHECKING

import torch
from compressed_tensors.utils import has_offloaded_params


if TYPE_CHECKING:
    from compressed_tensors.transforms.transform_args import TransformArgs


class ParameterizedDefaultDict(dict):
    def __init__(self, default_factory):
        self.default_factory = default_factory

    def __missing__(self, key):
        if isinstance(key, tuple):
            value = self.default_factory(*key)
        else:
            value = self.default_factory(key)
        self[key] = value
        return value


def get_matrix_size(module: torch.nn.Module, args: "TransformArgs") -> int:
    assert isinstance(module, torch.nn.Linear)
    if args.location == "input" or args.location == "weight" and args.side == "right":
        return module.in_features
    else:
        return module.out_features


def get_offload_device(module: torch.nn.Module) -> torch.device:
    if has_offloaded_params(module):
        return module._hf_hook.weights_map.values().device
    else:
        next(module.parameters()).device
