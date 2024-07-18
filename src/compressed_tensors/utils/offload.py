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

__all__ = [
    "is_module_offloaded",
    "get_execution_device",
    "get_offloaded_device",
    "update_prefix_dict",
    "update_parameter_data",
]


def is_module_offloaded(module):
    return hasattr(module, "_hf_hook") and module._hf_hook.offload


def get_execution_device(module):
    if is_module_offloaded(module):
        return module._hf_hook.execution_device
    return next(module.parameters()).device


def get_offloaded_device(module):
    if is_module_offloaded(module):
        first_key = list(module._hf_hook.weights_map.keys())[0]
        prefix_dataset = module._hf_hook.weights_map.dataset
        return prefix_dataset[first_key].device
    return next(module.parameters()).device


def update_prefix_dict(module, key, data):
    if not is_module_offloaded(module):
        raise ValueError("Prefix dict is only applicable to offloaded modules")
    prefix_dict = module._hf_hook.weights_map
    prefix_dict.dataset[f"{prefix_dict.prefix}{key}"] = data


def update_parameter_data(module, new_param_data, param_name):
    device = next(module.parameters()).device

    offloaded = False
    if is_module_offloaded(module):
        offload_device = get_offloaded_device(module)
        offloaded = True

    parameter = getattr(module, param_name, None)
    dtype = parameter.dtype
    parameter.data = new_param_data.to(device).to(dtype)

    if offloaded:
        prefix_dict = module._hf_hook.weights_map.dataset
        prefix = module._hf_hook.weights_map.prefix
        prefix_dict[f"{prefix}{param_name}"] = new_param_data.to(offload_device).to(
            dtype
        )
