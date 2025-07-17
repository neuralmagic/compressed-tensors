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

from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING, Callable, Optional

import torch
from compressed_tensors.utils import getattr_chain
from torch.utils.hooks import RemovableHandle
from transformers import AttentionInterface, PreTrainedModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import eager_attention_forward


if TYPE_CHECKING:
    from compressed_tensors.quantization import QuantizationArgs, QuantizationStatus


__all__ = ["CompressedAttentionImpl", "enable_compressed_attention", "call_attn_impl"]


ActivationHookFn = Callable[[torch.nn.Module, torch.Tensor], None]


class CompressedAttentionImpl(torch.nn.Module):
    """
    Callable attention implementation which applies transforms, calibration, and
    quantization if applicable. Can be hooked with calibrations hooks in order to
    trigger quantization observers.

    :param attn_implementation: original attention implementation to call after hooks
    """

    NAME = "compressed_attention"
    ATTN_IMPL = "eager"
    _ATTN_IMPLS = dict()

    @classmethod
    def from_module(cls, module: torch.nn.Module):
        if module not in cls._ATTN_IMPLS:
            cls._ATTN_IMPLS[module] = cls()
        return cls._ATTN_IMPLS[module]

    def __init__(self):
        super().__init__()
        self.query_hooks: OrderedDict[int, ActivationHookFn] = OrderedDict()
        self.key_hooks: OrderedDict[int, ActivationHookFn] = OrderedDict()
        self.value_hooks: OrderedDict[int, ActivationHookFn] = OrderedDict()

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
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        from compressed_tensors.quantization import forward_quantize

        for hook in self.query_hooks.values():
            output = hook(self, query)
            if output is not None:
                query = output

        for hook in self.key_hooks.values():
            output = hook(self, key)
            if output is not None:
                key = output

        for hook in self.value_hooks.values():
            output = hook(self, value)
            if output is not None:
                value = output

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
        # `eager_attention_forward` is duplicated across models by design
        # assume that llama implementation is representative of all attention functions
        # see: https://github.com/huggingface/transformers/issues/38541#issuecomment-2958567250  # noqa: 501

        attention_fn: Callable = (
            eager_attention_forward
            # if self.ATTN_IMPL == "eager"
            # else ALL_ATTENTION_FUNCTIONS[self.ATTN_IMPL]
        )
        # print(self.ATTN_IMPL)
        return attention_fn(
            module, query, key, value, attention_mask, scaling, dropout, **kwargs
        )


def enable_compressed_attention(model: torch.nn.Module):
    """
    Enables transforms, calibration, and quantization for an attention implementation.
    This function can safetly be called multiple times on the same model.

    :param model: model to enable compressed quantization for
    :return: singleton instance of `CompressedAttentionImpl`
    """
    if not isinstance(model, PreTrainedModel):
        return

    attn_impl = getattr(model.config, "_attn_implementation", "eager")

    CompressedAttentionImpl.ATTN_IMPL = attn_impl
    AttentionInterface.register(CompressedAttentionImpl.NAME, call_attn_impl)
    model.config._attn_implementation = CompressedAttentionImpl.NAME
    # model.set_attention_implementation(CompressedAttentionImpl.NAME)


def call_attn_impl(module: torch.nn.Module, *args, **kwargs):
    return CompressedAttentionImpl.from_module(module)(module, *args, **kwargs)
