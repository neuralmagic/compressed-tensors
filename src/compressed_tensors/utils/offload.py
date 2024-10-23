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

import contextlib
from functools import wraps
from typing import Optional

import torch
from compressed_tensors.utils.helpers import getattr_chain


try:
    from accelerate.hooks import AlignDevicesHook
except ImportError:
    AlignDevicesHook = None


__all__ = [
    "is_module_offloaded",
    "get_execution_device",
    "get_offloaded_device",
    "update_prefix_dict",
    "update_parameter_data",
]


# upstream candidate
def has_offloaded_params(module: torch.nn.Module) -> bool:
    """
    Checks if a module has offloaded parameters by checking if the given module
    has a AlignDevicesHook attached with offloading enabled

    Args:
        module (`torch.nn.Module`): The module to check for an offload hook.

    Returns:
        bool: `True` if the module has an offload hook and offloading is enabled,
        `False` otherwise.
    """
    return (
        hasattr(module, "_hf_hook") and
        isinstance(module._hf_hook, AlignDevicesHook) and
        module._hf_hook.offload
    )


# depreciation candidate
@wraps(has_offloaded_params)
def is_module_offloaded(module: torch.nn.Module) -> bool:
    if AlignDevicesHook is None:
        return False

    return has_offloaded_params(module)


# depreciation candidate
def get_execution_device(module: torch.nn.Module) -> torch.device:
    """
    :param module: module to check
    :return: device module is loaded onto during forward pass
    """
    if is_module_offloaded(module):
        return module._hf_hook.execution_device
    device = next(module.parameters()).device

    # offload only gets set for leaf modules, fallback to checking for device type
    if device.type == "meta":
        return module._hf_hook.execution_device

    return device


# depreciation candidate
def get_offloaded_device(module: torch.nn.Module) -> torch.device:
    """
    :param module: module to check
    :return: device module is offloaded to onto after forward pass
    """
    if is_module_offloaded(module):
        first_key = list(module._hf_hook.weights_map.keys())[0]
        prefix_dataset = module._hf_hook.weights_map.dataset
        return prefix_dataset[first_key].device
    return next(module.parameters()).device


# depreciation candidate
def update_prefix_dict(module: torch.nn.Module, key: str, data: torch.Tensor):
    """
    Updates the offloaded state dict for a given module. Parameter named key is replaced
    by data. This is neccesary because parameter updates for offloaded modules do not
    persist automatically between loads. This function only affects the offloaded
    state dict and not the current state of the loaded module.

    :param module: module containing the parameter to update
    :param key: name of parameter to update
    :param data: tensor to update parameter with in the offloaded state dict
    """
    if not is_module_offloaded(module):
        raise ValueError("Prefix dict is only applicable to offloaded modules")
    prefix_dict = module._hf_hook.weights_map
    prefix_dict.dataset[f"{prefix_dict.prefix}{key}"] = data


# upstream candidate
def update_offload_parameter(
    module: torch.nn.Module,
    name: str,
    data: torch.Tensor,
    init_device: Optional[torch.device] = torch.device("cpu"),
):
    """
    :param module: module containing the parameter to update
    :param name: name of module parameter to update
    :param data: tensor to update parameter with
    :param init_device: offload device for newly registered parameters
    """
    param = getattr(module, name)
    param.data = data

    prefix_dict = getattr_chain(module, "module._hf_hook.weights_map.dataset", None)
    if prefix_dict is not None:
        prefix = module._hf_hook.weights_map.prefix
        key = f"{prefix}{name}"

        offload_device = prefix_dict[key].device if key in prefix_dict else init_device
        prefix_dict[key] = data.to(device=offload_device)


# depreciation candidate
def update_parameter_data(
    module: torch.nn.Module, new_param_data: torch.Tensor, param_name: str
):
    param = getattr(module, param_name)
    new_param_data = new_param_data.to(device=param.device, dtype=param.dtype)
    update_offload_parameter(module, param_name, new_param_data)


# upstream candidate
@contextlib.contextmanager
def align_module(module: torch.nn.Module, execution_device: Optional[torch.device] = None):
    """
    Move a module's parameters to the execution device

    :param module: module with parameters to align
    :param execution_device: if provided, overrides module execution device
        within the context
    """
    if has_offloaded_params(module):
        if execution_device is not None:
            original_device = module._hf_hook.execution_device
            module._hf_hook.execution_device = original_device

        module._hf_hook.pre_forward(module)
        yield
        module._hf_hook.post_forward(module, None)

        if execution_device is not None:
            module._hf_hook.execution_device = original_device

    elif execution_device is not None:
        devices = {}
        for name, param in module.named_parameters():
            devices[name] = param.device
            setattr(module, name, param.to(execution_device))

        yield

        for name, param_device in module.named_parameters:
            setattr(module, name, param.to(param_device))

    else:
        yield


@contextlib.contextmanager
def modify_offload_module(
    module: torch.nn.Module,
    execution_device: Optional[torch.device] = None,
    offload_device: Optional[torch.device] = None,
):
    with align_module(module, execution_device):
        yield

        # there is little performance gain from checking if a parameter's data
        # has been modified before copying since the new data must be copied
        # to the offload device anyways; just update all module parameters
        for name, param in module.named_parameters():
            update_offload_parameter(module, name, param.data, offload_device)


# upstream candidate?
def register_offload_parameter(
    module: torch.nn.Module,
    name: str,
    data: torch.Tensor,
    offload_device: Optional[torch.device],
):
    module.register_parameter(name, torch.nn.Parameter(data))
    update_offload_parameter(module, name, data, offload_device)
