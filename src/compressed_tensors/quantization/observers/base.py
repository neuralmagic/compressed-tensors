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

import logging
from typing import Optional, Tuple

from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.registry.registry import RegistryMixin
from torch import FloatTensor, IntTensor, Tensor
from torch.nn import Module


_LOGGER = logging.getLogger(__name__)


__all__ = ["Observer"]


class Observer(Module, RegistryMixin):
    """
    Base Observer class to be subclassed for specific implementation.
    Subclasses should override `calculate_qparams` to return a scale, zero_point
    pair
    """

    def __init__(self, quantization_args: QuantizationArgs):
        self.quantization_args: QuantizationArgs = quantization_args
        super().__init__()
        self._scale = None
        self._zero_point = None
        # how many tokens were observed during the forward pass
        # (cannot set a property due to inheritance from torch.nn.Module)
        self._tokens_per_batch = None

    def forward(self, observed: Tensor) -> Tuple[FloatTensor, IntTensor]:
        """
        maps directly to get_qparams
        :param observed: optional observed tensor to calculate quantization parameters
            from
        :return: tuple of scale and zero point based on last observed value
        """
        self.record_tokens_per_batch(observed)
        return self.get_qparams(observed=observed)

    def calculate_qparams(self, observed: Tensor) -> Tuple[FloatTensor, IntTensor]:
        """
        :param observed: observed tensor to calculate quantization parameters for
        :return: tuple of scale and zero point derived from the observed tensor
        """
        raise NotImplementedError(f"{self.__class__} must implement calculate_qparams")

    def get_qparams(
        self, observed: Optional[Tensor] = None
    ) -> Tuple[FloatTensor, IntTensor]:
        """
        Convenience function to wrap overwritten calculate_qparams
        adds support to make observed tensor optional and support for tracking latest
        calculated scale and zero point
        :param observed: optional observed tensor to calculate quantization parameters
            from
        :return: tuple of scale and zero point based on last observed value
        """
        if observed is not None and observed.numel() > 0:
            # re-calculate scale and zero point, update the stored value
            self._scale, self._zero_point = self.calculate_qparams(observed)
        return self._scale, self._zero_point

    def record_tokens_per_batch(self, batch_tensor: Tensor):
        """
        Records the number of tokens observed during the forward pass, by
        setting the _tokens_per_batch attribute of the class.

        Note: The batch_tensor is expected to have two dimensions
            (batch_size * sequence_length, num_features). This is the
            general shape expected by the forward pass of the expert
            layers in a MOE model. If the input tensor does not have
            two dimensions, the _tokens_per_batch attribute will be set
            to None.
        """
        if not isinstance(batch_tensor, Tensor):
            raise ValueError(f"Expected value to be a tensor, got {type(batch_tensor)}")

        if batch_tensor.ndim != 2:
            _LOGGER.debug(
                "The input tensor is expected to have two dimensions "
                "(batch_size * sequence_length, num_features). "
                f"The input tensor has {batch_tensor.ndim} dimensions."
            )
            observed_tokens = None
        else:
            # batch_tensor (batch_size * sequence_length, num_features)
            # observed_tokens (batch_size * sequence_length)
            observed_tokens, _ = batch_tensor.shape

        self._tokens_per_batch = observed_tokens
