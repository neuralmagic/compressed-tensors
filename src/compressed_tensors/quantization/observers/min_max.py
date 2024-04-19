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

from typing import Tuple

import torch
from compressed_tensors.quantization.observers.base import Observer
from compressed_tensors.quantization.observers.helpers import calculate_qparams
from compressed_tensors.quantization.quant_args import QuantizationArgs
from torch import FloatTensor, IntTensor, Tensor


__all__ = ["MinMaxObserver"]


@Observer.register("minmax")
class MinMaxObserver(Observer):
    """
    Implements a dynamic quantization observer that sets the scale and
    zero point based on the overall min and max value
    """

    def __init__(self, quantization_args: QuantizationArgs):
        super().__init__(quantization_args=quantization_args)

        self.min_val = float("inf")
        self.max_val = -float("inf")
        self.counter = 0

    def calculate_qparams(
        self, observed: Tensor, group_size: int = 0
    ) -> Tuple[FloatTensor, IntTensor]:
        """

        :param observed: observed tensor to calculate quantization parameters for
        :return: tuple of scale and zero point derived from the observed tensor
        """

        # quantize by groups
        if group_size > 0:
            columns = observed.shape[1]
            scales, zero_points = [], []
            for i in range(0, columns, self.quantization_args.group_size):
                scale, zero_point = self.calculate_qparams(
                    observed[:, i : (i + group_size)], 0
                )
                scales.append(scale)
                zero_points.append(zero_point)

            return torch.cat(scales), torch.cat(zero_points)

        # channel-wise quantization
        if group_size < 0:
            ...

        if group_size == 0:

            min_val = torch.tensor([observed.min()])
            max_val = torch.tensor([observed.max()])

            # update global min and max
            if self.counter > 0:
                self.min_val = torch.min(min_val, self.min_val)
                self.max_val = torch.max(max_val, self.max_val)
            else:
                self.min_val = min_val
                self.max_val = max_val

            # ensure that the zeros are in the range
            min_val = torch.min(self.min_val, torch.zeros_like(self.min_val))
            max_val = torch.max(self.max_val, torch.zeros_like(self.max_val))

            self.counter += 1
            return calculate_qparams(min_val, max_val, self.quantization_args)
