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

from typing import Any, Optional, Union

import torch
from compressed_tensors.registry.registry import RegistryMixin


__all__ = ["Transforms"]


# TODO: We don't need to save all the __call__ args for serialization or even have
# them defined by a recipe. Some of them, such as if the transformation should be the
# first or second matirx in torch.matmul depending on dimensions, can be inferred
# by the layer time likely.


class Transforms(RegistryMixin):
    def __new__(
        cls,
        transform: torch.Tensor,
        device: Optional[Union[str, torch.device]] = "cpu",
        *args,
        **kwargs,
    ):
        """
        Base class for setting up transforms. The registry creates transforms
        as parameters which can be attached to modules.

        import torch

        size = 1024
        dtype = torch.bfloat16
        module = torch.nn.Linear(size, size)

        # Load transform 1
        hadamard_transform = Transforms.load_from_registry(
            "random_hadamard", size=size, dtype=dtype
        )
        hadamard_apply = Transform.fetch_apply("random_hadamard")
        module.weight_transform = hadamard_transform

        transformed_output = hadamard_apply(input_tensor=module.weight, transform=moduel.weight_transform)

        :param transform: transform (e.g. torch.Tensor, scalar) to be applied
        """
        return torch.nn.Parameter(transform.to(device), requires_grad=False)

    @classmethod
    def fetch_apply(cls, name: str):
        constructor = cls.get_value_from_registry(name=name)
        return constructor.apply

    # TODO: this is a static method but we could potentially just have a set of utils
    # which fetch the appropriate apply method since how the transform is applied
    # will likely be the same for matrices and therefore, we can have one common apply
    # for RandomHadamard, Hadamard, MatrixMultiply etc.
    @staticmethod
    def apply(
        transform: torch.Tensor, input_tensor: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        :param transform: transform tensor to be applied to the input_tensor
        :param input_tensor: tensor to which the transformation is applied

        returns a transformed input tensor
        """
        raise NotImplementedError()
