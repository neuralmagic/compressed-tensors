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

import inspect
from typing import Callable, Dict, Optional

import torch
from compressed_tensors.modeling.kvcache import initialize_hooked_kv_cache
from compressed_tensors.quantization import QuantizationScheme, forward_quantize
from torch.utils.hooks import RemovableHandle
from transformers import AttentionInterface, PreTrainedModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


_original_impl = "eager"  # mutable


class QuantizedAttentionImpl(torch.nn.Module):
    def __init__(self, attn_module: torch.nn.Module):
        super().__init__()
        self.attn_module_container = [attn_module]  # avoid circular reference
        self.quantization_enabled = False

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
        # quantization always gets applied last after hooks, in the same way that
        # quantized `wrapped_forward` always applies quantization last
        # because it does not use hooks
        scheme: Optional[QuantizationScheme] = getattr(
            module, "quantization_scheme", None
        )
        if scheme is not None and self.quantization_enabled:
            if scheme.input_activations is not None:
                query = forward_quantize(module, query, "q", scheme.input_activations)

            if scheme.weights is not None:
                raise ValueError("")

            if scheme.output_activations is not None:
                raise NotImplementedError("")

        return ALL_ATTENTION_FUNCTIONS[_original_impl](
            module, query, key, value, attention_mask, scaling, dropout, **kwargs
        )


def ct_hooked_attention(module: torch.nn.Module, *args, **kwargs):
    if hasattr(module, "impl"):
        return module.impl(module, *args, **kwargs)
    else:
        return ALL_ATTENTION_FUNCTIONS[_original_impl](module, *args, **kwargs)


def initialize_hooked_attention(
    model: PreTrainedModel, module: torch.nn.Module, quantize: bool = True
):
    if not hasattr(module, "impl"):
        module.register_module("impl", QuantizedAttentionImpl(module))

        if model.config._attn_implementation != "ct_hooked_attention":
            # assumes only one model at a time
            global _original_impl
            _original_impl = model.config._attn_implementation

            AttentionInterface.register("ct_hooked_attention", ct_hooked_attention)
            model.config._attn_implementation = "ct_hooked_attention"

    if quantize:
        # initialize q scale

        impl: QuantizedAttentionImpl = getattr(module, "impl")
        impl.quantization_enabled = True

        initialize_hooked_kv_cache(module, quantize=True)


def register_query_hook(module: torch.nn.Module, func: Callable) -> RemovableHandle:
    """
    Registers a forward pre-hook on `module.impl` that replaces the `query` argument
    with `func(mod, query)` (handles both positional and keyword forms).
    """
    impl = getattr(module, "impl")

    def _hook(mod: torch.nn.Module, args, kwargs):
        # Keyword case
        if "query" in kwargs:
            kwargs["query"] = func(mod, kwargs["query"])
            return args, kwargs

        # Positional case: find the index of `query` in impl.forward
        sig = inspect.signature(mod.forward)
        param_names = tuple(sig.parameters.keys())
        try:
            idx = param_names.index("query")
        except ValueError:
            # No `query` parameter; nothing to do
            return args, kwargs

        if idx < len(args):
            args = list(args)
            args[idx] = func(mod, args[idx])
            return tuple(args), kwargs

        # Not present explicitly (maybe defaulted)
        return args, kwargs

    return impl.register_forward_pre_hook(_hook, with_kwargs=True)
