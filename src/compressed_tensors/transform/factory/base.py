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

from abc import ABC, abstractmethod

import torch
import torch.nn.utils.parametrize as P
from compressed_tensors.registry.registry import RegistryMixin, T
from compressed_tensors.transform import TransformArgs, TransformScheme
from compressed_tensors.utils import (
    get_execution_device,
    has_offloaded_params,
    patch_attr,
    register_offload_module,
    update_offload_parameter,
)
from torch.nn import Linear, Module, Parameter


__all__ = ["TransformFactory", "TransformBase"]


class TransformFactory(RegistryMixin, ABC):
    def __init__(
        self,
        name: str,
        scheme: TransformScheme,
        seed: int = 42,
        keep_onloaded: bool = False,
    ):
        self.name = name
        self.scheme = scheme
        self.seed = seed

        self.keep_onloaded = keep_onloaded
        self.params_map = None

        if self.keep_onloaded and self.scheme.randomize_modules:
            # warning
            pass

    @classmethod
    def from_scheme(cls: type[T], scheme: TransformScheme, **kwargs) -> T:
        constructor = cls.get_value_from_registry(name=scheme.type)
        return constructor(scheme=scheme, **kwargs)

    @abstractmethod
    def create_transform(self, module: Module, args: TransformArgs) -> "TransformBase":
        raise NotImplementedError()

    def apply_to_model(self, model: Module):
        for path, module in model.named_modules():
            for arg in self.scheme.apply:
                # if match_targets(path, arg.targets):
                if isinstance(module, Linear):
                    self._apply_to_module(module, arg)

    def _apply_to_module(self, module: Module, args: TransformArgs):
        name = self._get_transform_name(args)

        # create transform as submodule
        transform = self.create_transform(module, args)
        assert all(pm.device == torch.device("cpu") for pm in transform.parameters())

        # because transform weights are often shared between weights,
        if self.keep_onloaded and has_offloaded_params(transform):
            self._manually_onload(transform)

        # register input transformation hook
        if args.location == "input":

            def input_hook(_, args):
                input = args[0]
                return transform(input)

            module.register_forward_pre_hook(input_hook)
            register_offload_module(module, name, transform)

        # eagerly apply transformation to weight
        elif args.location == "weight":
            assert isinstance(module, torch.nn.Linear)
            assert module.bias is None
            with torch.no_grad():
                transformed_weight = transform(module.weight)
                update_offload_parameter(module, "weight", transformed_weight)

            if self.scheme.requires_grad:
                # for training, the weight changes with every forward pass
                # so we can leverage parametrization to propagate gradient
                if has_offloaded_params(module):
                    raise ValueError("Offloaded training is not supported")
                P.register_parametrization(module, "weight", transform)

        # register output transformation hook
        elif args.location == "output":

            def output_hook(_, _input, output):
                return transform(output)

            module.register_forward_hook(output_hook)
            register_offload_module(module, name, transform)

        # other locations such as q_attn and k_attn  has not been implemented
        else:
            raise NotImplementedError()

    def _manually_onload(self, module: Module):
        assert not self.scheme.requires_grad
        exec_device = get_execution_device(module)

        # parameters are onloaded to self.params_map which is shared across hooks
        if self.params_map is None:
            self.params_map = module._hf_hook.tied_params_map
        elif module._hf_hook.tied_params_map is not self.params_map:
            raise ValueError()

        # onload all parameters associated withe the transform
        for param in module.parameters():
            if param.data_ptr() not in self.params_map:
                self.params_map[param.data_ptr()] = {}
            self.params_map[param.data_ptr()][exec_device] = param.to(exec_device)

    def _get_transform_name(self, args: TransformArgs) -> str:
        components = [self.name, args.location]
        if args.side is not None:
            components.append(args.side)

        return "_".join(components)


class TransformBase(Module, ABC):
    args: TransformArgs

    @abstractmethod
    def forward(self, value: Parameter) -> Parameter:
        raise NotImplementedError()

    def right_inverse(self, value: Parameter) -> Parameter:
        with patch_attr(self.args, "inverse", not self.args.inverse):
            return self.forward(value)

    def __repr__(self):
        return f"{self.__class__.__name__}(inverse={self.args.inverse})"


# def compress_module_transforms(module: Module):
#     for submodule in module.children():
#         if isinstance(submodule, MatrixTransformBase):
#             if submodule.args.location == "input":


#             for submodule.named_parameters()
#                 register_offload_parameter()
#                 submodule
