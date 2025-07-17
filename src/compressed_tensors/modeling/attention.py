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

from collections import OrderedDict
from typing import Callable, Optional

from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStatus,
    forward_quantize,
)
from compressed_tensors.transform import TransformBase, TransformLocation
from compressed_tensors.utils import getattr_chain
from torch import Module, Tensor
from torch.utils.hooks import RemovableHandle
from transformers import AttentionInterface, PreTrainedModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import eager_attention_forward


__all__ = [
    "CompressedAttentionImpl",
    "enable_compressed_attention",
    "get_compressed_attention_impl",
]


COMPRESSED_ATTENTION_NAME = "compressed_attention"


ActivationHookFn = Callable[[Module, Tensor]]


class CompressedAttentionImpl(Module):
    """
    Callable attention implementation which applies transforms, calibration, and
    quantization if applicable. Can be hooked with calibrations hooks in order to
    trigger quantization observers.

    In the future, the idea of making attention implementions hookable Modules
    could be upstreamed to transformers model definitions

    :param attn_implementation: original attention implementation to call after hooks
    """

    def __init__(self, attn_implementation: str):
        self.attn_implementation = attn_implementation
        self.query_hooks: OrderedDict[int, ActivationHookFn] = OrderedDict()
        self.key_hooks: OrderedDict[int, ActivationHookFn] = OrderedDict()
        self.value_hooks: OrderedDict[int, ActivationHookFn] = OrderedDict()

        # `eager_attention_forward` is duplicated across models by design
        # assume that llama implementation is representative of all attention functions
        # see: https://github.com/huggingface/transformers/issues/38541#issuecomment-2958567250  # noqa: 501
        self.attention_fn: Callable = (
            eager_attention_forward
            if self.attn_implementation == "eager"
            else ALL_ATTENTION_FUNCTIONS[self.attn_implementation]
        )

    def register_query_hook(self, hook: ActivationHookFn) -> RemovableHandle:
        handle = RemovableHandle(self.query_hooks)
        self.query_hooks[handle.id] = hook

        return handle

    def register_key_hook(self, hook: ActivationHookFn) -> RemovableHandle:
        handle = RemovableHandle(self.key_hooks)
        self.key_hooks[handle.id] = hook

        return handle

    def register_value_hook(self, hook: ActivationHookFn) -> RemovableHandle:
        handle = RemovableHandle(self.value_hooks)
        self.value_hooks[handle.id] = hook

        return handle

    def forward(
        self,
        module: Module,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        for hook in self.query_hooks():
            query = hook(self, query)

        for hook in self.key_hooks():
            key = hook(self, key)

        for hook in self.value_hooks():
            value = hook(self, value)

        # TODO: attnq
        # 2. calibrate/ apply quantization
        # args_path = "quantization_scheme.input_activations"
        # status_path = "quantization_status"
        # input_args: Optional[QuantizationArgs] = getattr_chain(
        #     module, args_path, None
        # )
        # status: Optional[QuantizationStatus] = getattr(module, status_path, None)
        # if input_args is not None and status in (
        #     QuantizationStatus.CALIBRATION,
        #     QuantizationStatus.FROZEN,
        # ):
        #     query = forward_quantize(module, query, "q", input_args)
        #     key = forward_quantize(module, key, "k", input_args)
        #     value = forward_quantize(module, value, "v", input_args)

        # 3. apply original attention function
        return self.attention_fn(
            module, query, key, value, attention_mask, scaling, dropout, **kwargs
        )


def enable_compressed_attention(model: PreTrainedModel):
    """
    Enables transforms, calibration, and quantization for an attention implementation.
    This function can safetly be called multiple times on the same model.

    :param model: model to enable compressed quantization for
    :return: singleton instance of `CompressedAttentionImpl`
    """
    attn_implementation = getattr(model.config, "attn_implementation", "eager")
    if attn_implementation != COMPRESSED_ATTENTION_NAME:
        compressed_attention = CompressedAttentionImpl(attn_implementation)
        AttentionInterface.register(COMPRESSED_ATTENTION_NAME, compressed_attention)
        model.config.attn_implementation = COMPRESSED_ATTENTION_NAME


def get_compressed_attention_impl() -> CompressedAttentionImpl:
    if COMPRESSED_ATTENTION_NAME not in ALL_ATTENTION_FUNCTIONS:
        raise ValueError(
            "Please call `enable_compressed_attention(model)` before attempting "
            "to get singleton instance of `CompressedAttentionImpl`"
        )
    return ALL_ATTENTION_FUNCTIONS[COMPRESSED_ATTENTION_NAME]
