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
"""
Utilities associated with offloading functionality provided by `accelerate`.

| ----------------------------------------------------------------------------------------------------- | # noqa: E501
| Operation | Without offloading support             | With offloading support                          | # noqa: E501
| --------- | -------------------------------------- | ------------------------------------------------ | # noqa: E501
| Add       | module.register_parameter(name, param) | register_offload_parameter(module, name, param)  | # noqa: E501
| Check     | N/A                                    | has_offloaded_params(module)                     | # noqa: E501
| Onload    | N/A                                    | with align_module_device(module)                 | # noqa: E501
| Update    | module.name.data.copy_(new_data)       | update_offload_parameter(module, name, new_data) | # noqa: E501
| Delete    | del module.name                        | delete_offload_parameter(module, name)           | # noqa: E501
| ----------------------------------------------------------------------------------------------------- | # noqa: E501
"""

import contextlib
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union

import torch


try:
    from accelerate.hooks import (
        AlignDevicesHook,
        add_hook_to_module,
        remove_hook_from_module,
    )
    from accelerate.utils import (
        OffloadedWeightsLoader,
        PrefixedDataset,
        set_module_tensor_to_device,
    )

    _has_accelerate = True
except ImportError:
    _has_accelerate = False
    AlignDevicesHook = None
    add_hook_to_module = None
    remove_hook_from_module = None
    OffloadedWeightsLoader = None
    PrefixedDataset = None
    set_module_tensor_to_device = None


__all__ = [
    "is_module_offloaded",
    "get_execution_device",
    "get_offloaded_device",
    "update_prefix_dict",
    "update_parameter_data",
    "register_offload_parameter",
    "update_offload_parameter",
    "delete_offload_parameter",
    "has_offloaded_params",
    "disable_hf_hook",
    "align_module_device",
]


def check_accelerate(fallback: Any):
    def decorator(func: Callable[[Any], Any]):
        if not _has_accelerate:

            @wraps(func)
            def fallback_fn(*args, **kwargs):
                return fallback

            return fallback_fn

        return func

    return decorator


""" Candidates for Depreciation """


@check_accelerate(fallback=False)
def is_module_offloaded(module: torch.nn.Module) -> bool:
    return has_offloaded_params(module)


def get_execution_device(module: torch.nn.Module) -> torch.device:
    """
    :param module: module to check
    :return: device module is loaded onto during forward pass
    """
    if has_offloaded_params(module):
        return module._hf_hook.execution_device
    device = next(module.parameters()).device

    # offload only gets set for leaf modules, fallback to checking for device type
    if device.type == "meta":
        return module._hf_hook.execution_device

    return device


def get_offloaded_device(module: torch.nn.Module) -> torch.device:
    """
    :param module: module to check
    :return: device module is offloaded to onto after forward pass
    """
    if has_offloaded_params(module):
        first_key = list(module._hf_hook.weights_map.keys())[0]
        prefix_dataset = module._hf_hook.weights_map.dataset
        return prefix_dataset[first_key].device
    return next(module.parameters()).device


@check_accelerate(fallback=None)
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
    if not has_offloaded_params(module):
        raise ValueError("Prefix dict is only applicable to offloaded modules")
    prefix_dict = module._hf_hook.weights_map
    prefix_dict.dataset[f"{prefix_dict.prefix}{key}"] = data


def update_parameter_data(
    module: torch.nn.Module, new_param_data: torch.Tensor, param_name: str
):
    """
    Update the data of an existing parameter and its offload dict. Supports both
    parameters of offloaded modules and non-offloaded modules

    :param module: module containing the parameter to update
    :param new_param_data: tensor to update parameter with
    :param param_name: name of module parameter to update
    """
    update_offload_parameter(module, param_name, new_param_data)


""" Candidates for Upstreaming """


def register_offload_parameter(
    module: torch.nn.Module,
    name: str,
    parameter: torch.nn.Parameter,
):
    """
    Register a parameter to the given module which may be offloaded

    :param module: maybe offloaded module
    :param name: name of newly registered parameter
    :param parameter: parameter being registered
    """
    if has_offloaded_params(module):
        module.register_parameter(name, parameter)
        update_offload_parameter(module, name, parameter.data)
        set_module_tensor_to_device(module, name, "meta")
    else:
        device = next(module.parameters()).device
        parameter = parameter.to(device)
        module.register_parameter(name, parameter)


def update_offload_parameter(
    module: torch.nn.Module,
    name: str,
    data: Optional[torch.Tensor],
):
    """
    Update the data of an existing parameter and its offload dict. Supports both
    parameters of offloaded modules and non-offloaded modules

    :param module: module containing the parameter to update
    :param name: name of module parameter to update
    :param data: tensor to update parameter with
    """
    param = getattr(module, name)
    data = data.to(param.dtype)

    # copy data into onloaded parameter if applicable
    if param.device != "meta":
        param.data.copy_(data)

    # update offload dict
    if has_offloaded_params(module):
        weights_map = module._hf_hook.weights_map
        offload_to_weights_map(weights_map, name, data)


def delete_offload_parameter(module: torch.nn.Module, name: str):
    """
    Delete a parameter from a module which may be offloaded

    :param module: maybe offloaded module
    :param name: name of parameter being deleted
    """
    delattr(module, name)

    if has_offloaded_params(module):
        weights_map = module._hf_hook.weights_map

        # for upstreaming, better to add write capabilities to weight map classes first
        if isinstance(weights_map, PrefixedDataset):
            dataset = weights_map.dataset
            prefix = weights_map.prefix
            if dataset is not None:
                del dataset[f"{prefix}{name}"]

        elif isinstance(weights_map, dict):
            del weights_map[name]

        elif isinstance(weights_map, OffloadedWeightsLoader):
            raise NotImplementedError()

        elif weights_map is not None:
            raise NotImplementedError(
                f"Cannot delete parameter from weights_map of type {type(weights_map)}"
            )


@check_accelerate(fallback=contextlib.nullcontext())
@contextlib.contextmanager
def disable_hf_hook(module: torch.nn.Module, recurse: bool = False):
    hooks = {}

    def collect_hooks(module):
        nonlocal hooks
        if hasattr(module, "_hf_hook"):
            hooks[module] = module._hf_hook
            remove_hook_from_module(module)

        for submodule in module.children():
            collect_hooks(submodule)

    collect_hooks(module)

    yield

    for submodule, hook in hooks.items():
        add_hook_to_module(submodule, hook)


def offload_to_weights_map(
    weights_map: Union[PrefixedDataset, Dict, OffloadedWeightsLoader],
    key: str,
    value: torch.Tensor,
    default_device: torch.device = torch.device("cpu"),
):
    """
    Helper function which implements offloaded item assignment for PrefixedDataset,
    OffloadedWeightsLoader, and Dict types.

    :param weights_map: weight map to be updated with offload information
    :param key: key used to identify weight location
    :param value: weight being offloaded
    :param default_device: in the event that the weights_map does already contain
        offloaded weights or use disk offloading, the weight will be offloaded to the
        `default_device`
    """
    if isinstance(weights_map, PrefixedDataset):
        dataset = weights_map.dataset
        key = f"{weights_map.prefix}{key}"
        offload_to_weights_map(dataset, key, value)

    elif isinstance(weights_map, OffloadedWeightsLoader):
        if key not in weights_map.all_keys:
            weights_map.all_keys.append(key)

        if len(weights_map.index) <= 0:
            offload_to_weights_map(weights_map.state_dict, key, value)

        else:
            raise NotImplementedError(
                "Updating weights_map with disk offloading is not implemented yet"
            )
            # TODO: below may not be correct and has not been tested
            # FUTURE: upstream as OffloadedWeightsLoader.__set_item__
            # use_index = "safetensors_file" in next(iter(weights_map.values()))
            # if use_index:
            #     if key not in weights_map:
            #         weights_map.index[key] = {
            #             "safetensors_file": ???,
            #             "weight_name": key,
            #             "dtype": str(value.dtype)
            #         }

            #     weight_info = weights_map.index[key]
            #     file_path = weight_info["safetensors_file"]
            #     with safetensors.create_file(file_path) as file:
            #         file.write(value)

            # else:
            #     assert self.save_folder is not None
            #     weight_file = os.path.join(self.save_folder, f"{key}.dat")
            #     need_index_update = not os.path.exists(weight_file)
            #     offload_weight(
            #         value,
            #         key,
            #         weights_map.save_folder,
            #         weights_map.index
            #     )

            #     if need_index_update:
            #         save_offload_index(weights_map.index, weights_map.save_folder)

    elif isinstance(weights_map, dict):
        if key in weights_map:
            offload_device = weights_map[key].device
        else:
            tens = next(iter(weights_map.values()), None)
            offload_device = tens.device if tens is not None else default_device

        weights_map[key] = value.to(device=offload_device)

    else:
        raise NotImplementedError(
            "Updating offload data not implemented for weights_map of type "
            f"{type(weights_map)}"
        )


""" Upstreamed Functions """


# introduced in accelerate v1.1.0
@check_accelerate(fallback=False)
def has_offloaded_params(module: torch.nn.Module) -> bool:
    """
    Checks if a module has offloaded parameters by checking if the given module has a
    AlignDevicesHook attached with offloading enabled

    Args:
        module (`torch.nn.Module`): The module to check for an offload hook.

    Returns:
        bool: `True` if the module has an offload hook and offloading is enabled,
        `False` otherwise.
    """
    return (
        hasattr(module, "_hf_hook")
        and isinstance(module._hf_hook, AlignDevicesHook)
        and module._hf_hook.offload
    )


# introduced in accelerate v1.1.0
@check_accelerate(fallback=contextlib.nullcontext())
@contextlib.contextmanager
def align_module_device(
    module: torch.nn.Module, execution_device: Optional[torch.device] = None
):
    """
    Context manager that moves a module's parameters to the specified execution device.

    Args:
        module (`torch.nn.Module`):
            Module with parameters to align.
        execution_device (`torch.device`, *optional*):
            If provided, overrides the module's execution device within the context.
            Otherwise, use hook execution device or pass
    """
    if has_offloaded_params(module):
        if execution_device is not None:
            original_device = module._hf_hook.execution_device
            module._hf_hook.execution_device = execution_device

        try:
            module._hf_hook.pre_forward(module)
            yield
        finally:
            module._hf_hook.post_forward(module, None)
            if execution_device is not None:
                module._hf_hook.execution_device = original_device

    elif execution_device is not None:
        devices = {
            name: param.device for name, param in module.named_parameters(recurse=False)
        }
        try:
            for name in devices:
                set_module_tensor_to_device(module, name, execution_device)
            yield
        finally:
            for name, device in devices.items():
                set_module_tensor_to_device(module, name, device)

    else:
        yield
