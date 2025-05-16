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
from compressed_tensors.registry.registry import RegistryMixin
from compressed_tensors.transforms.transform_args import TransformArgs
from compressed_tensors.transforms.transform_scheme import TransformsScheme
from compressed_tensors.utils.offload import update_offload_parameter
from torch.nn import Linear, Module, Parameter


__all__ = ["Transforms"]


class MatrixTransformBase(Module, ABC):
    @abstractmethod
    def forward(self, value: Parameter) -> Parameter:
        raise NotImplementedError()

    @abstractmethod
    def right_inverse(self, value: Parameter) -> Parameter:
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__}(inverse={self.args.inverse})"


class MatrixTransformFactory(RegistryMixin, ABC):
    def __init__(self, name: str, scheme: TransformsScheme, seed: int):
        self.name = name
        self.scheme = scheme
        self.seed = seed
        self.transforms = []

    def apply_to_model(self, model: Module):
        for path, module in model.named_modules():
            for arg in self.scheme.apply:
                # if match_targets(path, arg.targets):
                if isinstance(module, Linear):
                    self._apply_to_module(module, arg)

    def _apply_to_module(self, module: Module, args: TransformArgs):
        name = self._get_transform_name(args)
        assert isinstance(module, torch.nn.Linear)
        assert not hasattr(module, name)
        transform = self.create_transform(module, args)
        module.register_module(name, transform)

        if args.location == "input":
            # TODO: need to specify side
            module.register_forward_pre_hook(lambda _, args: transform(args[0]))

        elif args.location == "weight":
            with torch.no_grad():
                print(module.weight)
                update_offload_parameter(module, "weight", transform(module.weight))
                # register_offload_parameterization(module, "weight", transform)
                P.register_parametrization(module, "weight", transform)
                print(module.parametrizations["weight"].original)
                # TODO: I don't like how creating a parametrizations list creates an
                # extra step for serialization. It'd be nicer if we had our own
                # parametrization implementation that still overloaded the get/setattr,
                # but simply checked an attached list of parametrizations, rather than
                # creating a separate ModuleList. Simliar to module._forward_hooks

                # we can also just disable the state_dict of module.parametrizations

        elif args.location == "output":
            module.register_forward_hook(lambda _, __, output: transform(output))

        else:
            raise NotImplementedError()

        self.transforms.append(transform)

    def _get_transform_name(self, args: TransformArgs) -> str:
        components = [self.name, args.location]
        if args.side is not None:
            components.append(args.side)

        return "_".join(components)

    @abstractmethod
    def create_transform(
        self, module: Module, args: TransformArgs
    ) -> MatrixTransformBase:
        raise NotImplementedError()
