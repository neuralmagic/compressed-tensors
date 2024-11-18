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
from typing import Any, Callable, Optional

import torch
from compressed_tensors.utils.helpers import getattr_chain


try:
    from accelerate.hooks import AlignDevicesHook
    from accelerate.utils import (
        OffloadedWeightsLoader,
        PrefixedDataset,
        set_module_tensor_to_device,
    )

    _has_accelerate = True
except ImportError:
    _has_accelerate = False
    AlignDevicesHook = None
    OffloadedWeightsLoader = None
    PrefixedDataset = None


__all__ = [
    "is_module_offloaded",
    "get_execution_device",
    "get_offloaded_device",
    "update_prefix_dict",
    "update_parameter_data",
    "register_offload_parameter",
    "update_offload_data",
    "delete_offload_parameter",
    "has_offloaded_params",
    "align_module",
]


def check_accelerate(fallback: Any):
    def decorator(func: Callable[[Any], Any]):
        if not _has_accelerate:
            return lambda *args, **kwargs: fallback

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
    if not has_offloaded_params(module):
        raise ValueError("Cannot infer offload device from non-offloaded module")

    first_key = next(module._hf_hook.weights_map.keys(), None)
    if first_key is None:
        raise ValueError("Cannot infer offload device from empty weights map")

    prefix_dataset = module._hf_hook.weights_map.dataset
    return prefix_dataset[first_key].device


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
    update_offload_data(module, param_name, new_param_data)


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
        update_offload_data(module, name, parameter.data)
        set_module_tensor_to_device(module, name, "meta")
    else:
        device = next(module.parameters()).device
        parameter = parameter.to(device)
        module.register_parameter(name, parameter)


def update_offload_data(
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

    # copy data into onloaded parameter if applicable
    if param.device != "meta":
        param.data.copy_(data)

    # update offload dict
    if has_offloaded_params(module):
        weights_map = module._hf_hook.weights_map

        # for upstreaming, better to add write capabilities to weight map classes first
        if isinstance(weights_map, PrefixedDataset):
            dataset = getattr_chain(module, "module._hf_hook.weights_map.dataset", None)
            if dataset is not None:
                prefix = module._hf_hook.weights_map.prefix
                key = f"{prefix}{name}"

                breakpoint()

                offload_device = (
                    dataset[key].device
                    if key in dataset
                    else next(dataset.values()).device
                )
                dataset[key] = param.data.to(device=offload_device)

        if isinstance(weights_map, OffloadedWeightsLoader):
            raise NotImplementedError()

        else:
            raise NotImplementedError()


def delete_offload_parameter(module: torch.nn.Module, name: str):
    """
    Delete a module from a module which may be offloaded

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

        elif isinstance(weights_map, OffloadedWeightsLoader):
            raise NotImplementedError()

        elif weights_map is not None:
            raise NotImplementedError(
                f"Cannot delete parameter from weights_map of type {type(weights_map)}"
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
@contextlib.contextmanager
def align_module(
    module: torch.nn.Module, execution_device: Optional[torch.device] = None
):
    """
    Moves a module's parameters to the specified execution device.

    Args:
        module (torch.nn.Module): Module with parameters to align.
        execution_device (Optional[torch.device]): If provided, overrides the
            module's execution device within the context.

    Yields:
        None: Yields control while the module's parameters are aligned to the execution
        device.
    """
    if has_offloaded_params(module):
        if execution_device is not None:
            original_device = module._hf_hook.execution_device
            module._hf_hook.execution_device = execution_device

        module._hf_hook.pre_forward(module)
        yield
        module._hf_hook.post_forward(module, None)

        if execution_device is not None:
            module._hf_hook.execution_device = original_device

    elif execution_device is not None:
        devices = {}
        for name, param in module.named_parameters():
            devices[name] = param.device
            set_module_tensor_to_device(
                module,
                name,
                execution_device,
            )

        yield

        for name, param in module.named_parameters():
            set_module_tensor_to_device(
                module,
                name,
                devices[name],
            )

    else:
        yield
