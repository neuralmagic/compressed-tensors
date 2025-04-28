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
from compressed_tensors.transforms.utils import apply_matrix_transform
from compressed_tensors.utils import register_offload_parameter, update_parameter_data


__all__ = ["Transforms"]


# TODO: We don't need to save all the __call__ args for serialization or even have
# them defined by a recipe. Some of them, such as if the transformation should be the
# first or second matirx in torch.matmul depending on dimensions, can be inferred
# by the layer time likely.


class Transforms(RegistryMixin):
    def __init__(
        self,
        transform: torch.Tensor,
        learnable: Optional[bool] = True,
        device: Optional[Union[str, torch.device]] = "cuda",
        dtype: Optional[torch.dtype] = torch.bfloat16,
    ):
        self.learnable = learnable
        """
        Base class for setting up transforms. The registry creates transforms
        as parameters which can be attached to modules.

        import torch

        size = 1024
        dtype = torch.bfloat16
        module = torch.nn.Linear(size, size)
        name = "weight_transform"

        hadamard_transform = Transforms.load_from_registry(
            "random_hadamard", size=size, dtype=dtype
        )

        hadamard_transform.register_to_module(name, module)
        module.transform_data = {name: {"call_args": dict, "class": hadamard_transform}}

        transformed_output = hadamard_transform.apply(input_tensor=module.weight)
        original_weight = hadamard_transform.inverse_apply(
            input_tensor=transformed_output)

        :param transform: transform (e.g. torch.Tensor, scalar) to be applied
        """
        if self.learnable:
            self.transform = torch.nn.Parameter(transform.to(dtype).to(device))
        else:
            self.transform = torch.nn.Buffer(transform.to(dtype).to(device))

    # register to class for easy offloading, serialization, deserialization
    def register_to_module(self, name: str, module: torch.nn.Module):
        if self.learnable:
            register_offload_parameter(module, name, self.transform)
        else:
            # TODO: have to verify serialization/offloading
            module.register_buffer(name, self.transform)

    def update_transform(
        self,
        data: torch.Tensor,
        module: Optional[torch.nn.Module] = None,
        name: Optional[str] = None,
    ):
        if module is None:
            self.transform.data.copy_(data)
        else:
            # If updating the module parameter data, assumes this is also the transform
            # data
            if name is None:
                raise ValueError("Name and module are required to update parma data")
            update_parameter_data(module, data, name)

    def apply(self, input_tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Apply the transform to the module
        """
        raise NotImplementedError()

    # TODO: potentially split into its own transform using the same shared set-up
    def inverse_apply(
        self, input_tensor: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Apply the inverse operation applied by the apply method
        """
        raise NotImplementedError()
