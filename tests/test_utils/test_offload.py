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

import torch
from compressed_tensors.utils import (
    align_module_device,
    delete_offload_parameter,
    disable_hf_hook,
    has_offloaded_params,
    register_offload_parameter,
    update_offload_data,
)
from tests.testing_utils import requires_accelerate


class ExampleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(0).float())
        self.b = torch.nn.Parameter(torch.tensor(0).float())

    def forward(self, x):
        return x * self.a + self.b


@requires_accelerate()
def test_has_offloaded_params():
    from accelerate.big_modeling import cpu_offload_with_hook
    from accelerate.hooks import attach_align_device_hook, remove_hook_from_module

    module = ExampleModule()
    assert not has_offloaded_params(module)

    attach_align_device_hook(module, offload=False)
    assert not has_offloaded_params(module)

    remove_hook_from_module(module)
    module, _ = cpu_offload_with_hook(module)
    assert not has_offloaded_params(module)

    remove_hook_from_module(module)
    attach_align_device_hook(module, offload=True)
    assert has_offloaded_params(module)


@requires_accelerate()
def test_register_offload_parameter():
    from accelerate.hooks import attach_align_device_hook

    module = ExampleModule()
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    # register a param prior to offloading
    register_offload_parameter(module, "c", parameter)
    assert hasattr(module, "c") and module.c == parameter

    # offloading, check that added param was offloaded
    attach_align_device_hook(module, offload=True)
    assert "c" in module._hf_hook.weights_map

    # register a param after offloading, check that added param was offloaded
    register_offload_parameter(module, "d", parameter)
    assert hasattr(module, "d") and module.d.device == torch.device("meta")
    assert "d" in module._hf_hook.weights_map

    # added parameters can be onloaded and offloaded
    with align_module_device(module, execution_device="cpu"):
        assert module.c.device == torch.device("cpu")
        assert module.d.device == torch.device("cpu")
    assert module.c.device == torch.device("meta")
    assert module.d.device == torch.device("meta")


@requires_accelerate()
def test_update_offload_data():
    from accelerate.hooks import attach_align_device_hook

    module = ExampleModule()
    param_a = torch.nn.Parameter(torch.tensor(1.0))
    param_b = torch.nn.Parameter(torch.tensor(2.0))

    # can update modules which are not offloaded
    update_offload_data(module, "a", param_a)
    assert module.a == param_a

    # can update modules which are offloaded
    attach_align_device_hook(module, offload=True)
    update_offload_data(module, "b", param_b)
    assert module.b.device == torch.device("meta")
    assert module._hf_hook.weights_map["b"] == param_b.data

    # data persists across onloading
    with align_module_device(module, execution_device="cpu"):
        assert module.a == param_a
        assert module.b == param_b
        assert module._hf_hook.weights_map["a"] == param_a.data
        assert module._hf_hook.weights_map["b"] == param_b.data

    # data persists across offloading
    assert module.a.device == torch.device("meta")
    assert module.b.device == torch.device("meta")
    assert module._hf_hook.weights_map["a"] == param_a.data
    assert module._hf_hook.weights_map["b"] == param_b.data


@requires_accelerate()
def test_delete_offload_parameter():
    from accelerate.hooks import attach_align_device_hook

    module = ExampleModule()
    param_c = torch.nn.Parameter(torch.tensor(1.0))
    param_d = torch.nn.Parameter(torch.tensor(2.0))
    register_offload_parameter(module, "c", param_c)
    register_offload_parameter(module, "d", param_d)

    # parameters are deleted
    delete_offload_parameter(module, "a")
    delete_offload_parameter(module, "c")
    assert not hasattr(module, "a")
    assert hasattr(module, "b")
    assert not hasattr(module, "c")
    assert hasattr(module, "d")

    # parameters and their offload are deleted
    attach_align_device_hook(module, offload=True)
    delete_offload_parameter(module, "b")
    delete_offload_parameter(module, "d")
    assert not hasattr(module, "a")
    assert not hasattr(module, "b")
    assert not hasattr(module, "c")
    assert not hasattr(module, "d")
    assert "a" not in module._hf_hook.weights_map
    assert "b" not in module._hf_hook.weights_map
    assert "c" not in module._hf_hook.weights_map
    assert "d" not in module._hf_hook.weights_map


@requires_accelerate()
def test_disable_hf_hook():
    from accelerate.hooks import attach_align_device_hook

    module = ExampleModule()

    def custom_forward():
        pass

    attach_align_device_hook(module, offload=True)
    with disable_hf_hook(module):
        assert not hasattr(module, "_hf_hook")
        module.forward = custom_forward

    assert hasattr(module, "_hf_hook")
    assert module._old_forward == custom_forward