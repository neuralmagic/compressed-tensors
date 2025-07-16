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


__all__ = ["CompressedAttentionImpl", "enable_compressed_attention"]


COMPRESSED_ATTENTION_NAME = "compressed_attention"


CalibHook = Callable[[Module, Tensor, Tensor, Tensor]]


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
        self.calib_hooks: OrderedDict[int, CalibHook] = OrderedDict()

        # `eager_attention_forward` is duplicated across models by design
        # assume that llama implementation is representative of all attention functions
        # see: https://github.com/huggingface/transformers/issues/38541#issuecomment-2958567250  # noqa: 501
        self.attention_fn: Callable = (
            eager_attention_forward
            if self.attn_implementation == "eager"
            else ALL_ATTENTION_FUNCTIONS[self.attn_implementation]
        )

    def register_calib_hook(self, hook: CalibHook) -> RemovableHandle:
        """
        Register a calibration hook which is called
        after transforms and before quantization

        :param hook: hook to be called
        :return: removable handle
        """
        handle = RemovableHandle(self.calib_hooks)
        self.calib_hooks[handle.id] = hook

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
        # 1. apply transforms
        for submodule in module.children():
            if isinstance(submodule, TransformBase):
                if TransformBase.args.location == TransformLocation.ATTN_Q:
                    query = submodule(query)

                if TransformBase.args.location == TransformLocation.ATTN_K:
                    key = submodule(key)

                # note that, unlike qk, v_proj does not undergo RoPE before attention
                # and can therefore be targeted directly

        # TODO: attnq
        # 2. calibrate/ apply quantization
        # args_path = "quantization_scheme.input_activations"
        # input_args: Optional[QuantizationArgs] = getattr_chain(module, args_path, None)  # noqa: 501
        # if input_args is not None:
        #     status_path = "quantization_status"
        #     status: Optional[QuantizationStatus] = getattr(module, status_path, None)

        #     # 2a. calibrate quantization
        #     if status == QuantizationStatus.CALIBRATION:
        #         assert len(self.calib_hooks) <= 1
        #         for hook in self.calib_hooks.items():
        #             hook(module, query, key, value)

        #     # 2b. apply quantization
        #     if status in (QuantizationStatus.CALIBRATION, QuantizationStatus.FROZEN):
        #         query = forward_quantize(module, query, "q", input_args)
        #         key = forward_quantize(module, key, "k", input_args)
        #         value = forward_quantize(module, value, "v", input_args)

        # 3. apply original attention function
        return self.attention_fn(
            module, query, key, value, attention_mask, scaling, dropout, **kwargs
        )


def enable_compressed_attention(model: PreTrainedModel) -> CompressedAttentionImpl:
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

    return ALL_ATTENTION_FUNCTIONS[COMPRESSED_ATTENTION_NAME]
