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
from compressed_tensors.utils import (
    has_offloaded_params,
    patch_attr,
    register_offload_module,
    update_offload_parameter,
)
from torch.nn import Linear, Module, Parameter


__all__ = ["TransformFactory", "TransformBase"]


class TransformFactory(RegistryMixin, ABC):
    def __init__(self, name: str, scheme: TransformsScheme, seed: int = 42):
        self.name = name
        self.scheme = scheme
        self.seed = seed
        self.transforms = []

    @classmethod
    def load_from_scheme(cls, name: str, scheme: TransformsScheme, **kwargs):
        constructor = cls.get_value_from_registry(name=scheme.type)
        return constructor(name=name, scheme=scheme, **kwargs)

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
        assert isinstance(module, torch.nn.Linear)
        assert not hasattr(module, name)

        # register transform as submodule
        transform = self.create_transform(module, args)
        assert all(pm.device == torch.device("cpu") for pm in transform.parameters())
        register_offload_module(module, name, transform)

        # register input transformation hook
        if args.location == "input":

            def input_hook(_, args):
                input = args[0]
                transform(input)

            module.register_forward_pre_hook(input_hook)

        # eagerly apply transformation to weight
        elif args.location == "weight":
            with torch.no_grad():
                transformed_weight = transform(module.weight)
                update_offload_parameter(module, "weight", transformed_weight)

            if self.scheme.requires_grad:
                # for training, the weight changes with every forward pass
                # so we can leverage parametrization to propagate gradient
                assert not has_offloaded_params(module), "offloaded training"
                P.register_parametrization(module, "weight", transform)

        # register output transformation hook
        elif args.location == "output":

            def output_hook(_, _input, output):
                return transform(output)

            module.register_forward_hook(output_hook)

        # other locations such as q_attn and k_attn  has not been implemented
        else:
            raise NotImplementedError()

        self.transforms.append(transform)  # for updating

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
