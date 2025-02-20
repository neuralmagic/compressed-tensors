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

from typing import Any, Optional

import torch
from compressed_tensors.registry.registry import RegistryMixin


__all__ = ["Transforms"]


# TODO: We don't need to save all the __call__ args for serialization or even have
# them defined by a recipe. Some of them, such as if the transformation should be the
# first or second matirx in torch.matmul depending on dimensions, can be inferred
# by the layer time likely.


class Transforms(RegistryMixin):
    def __init__(self, transform: Optional[Any] = None, *args, **kwargs):
        """
        :param transform: transform (e.g. matrix, scalar) to be applied
        """
        if transform is not None:
            self.transform = transform

    def __call__(self, input_tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        :param input_tensor: tensor to which the transformation is applied

        returns a transformed input tensor
        """
        raise NotImplementedError()
