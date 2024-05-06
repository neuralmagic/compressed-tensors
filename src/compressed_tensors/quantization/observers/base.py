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


class TokenCounter:
    """
    The goal of the class is to keep track of the number of tokens
    observed during the forward pass through the module. The class is keeping
    track of the number of tokens passed through each expert layer of a MOE
    (Mixture of Experts) model.

    It has been implemented as a mixin class to be used by the Observer class,
    because classes that inherit from torch.nn.Module have hard time dealing
    with properties.
    """

    def __init__(self):
        self._tokens_per_batch = None

    @property
    def tokens_per_batch(self) -> int:
        """
        Returns the number of tokens observed
        during the forward pass of a model."""
        return self._tokens_per_batch

    @tokens_per_batch.setter
    def tokens_per_batch(self, batch_tensor: Tensor):
        if not isinstance(batch_tensor, Tensor):
            raise ValueError(f"Expected value to be a tensor, got {type(batch_tensor)}")

        if batch_tensor.ndim != 2:
            _LOGGER.debug(
                "The input tensor is expected to have two dimensions "
                "(batch_size * sequence_length, num_features). "
                f"The input tensor has {batch_tensor.ndim} dimensions."
            )
            return
        # batch_tensor (batch_size * sequence_length, num_features)
        # observed_tokens (batch_size * sequence_length)
        observed_tokens, _ = batch_tensor.shape
        self._tokens_per_batch = observed_tokens


class Observer(Module, RegistryMixin, TokenCounter):
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

    def forward(self, observed: Tensor) -> Tuple[FloatTensor, IntTensor]:
        """
        maps directly to get_qparams
        :param observed: optional observed tensor to calculate quantization parameters
            from
        :return: tuple of scale and zero point based on last observed value
        """
        self.tokens_per_batch = observed
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
        if observed is not None:
            # re-calculate scale and zero point, update the stored value
            self._scale, self._zero_point = self.calculate_qparams(observed)
        return self._scale, self._zero_point
