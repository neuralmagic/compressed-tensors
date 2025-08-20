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
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from compressed_tensors.quantization import (
    KVCacheScaleType,
    QuantizationScheme,
    forward_quantize,
)
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from transformers import Cache, PretrainedConfig, PreTrainedModel


class QuantizedKVCache(torch.nn.Module):
    def __init__(self, attn_module: torch.nn.Module):
        super().__init__()
        self.attn_module_container = [attn_module]  # avoid nn.Module circular reference
        self.past_key_value: Optional[Cache] = None
        self._quantization_enabled = False

    def update(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        return self(*args, **kwargs)

    def forward(
        self,
        key_states: Tensor,
        value_states: Tensor,
        *args,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        # quantization always gets applied last after hooks, in the same way that
        # quantized `wrapped_forward` always applies quantization last
        # because it does not use hooks
        module = self.attn_module_container[0]
        scheme: Optional[QuantizationScheme] = getattr(
            module, "quantization_scheme", None
        )

        assert not self._quantization_enabled
        if scheme is not None and self._quantization_enabled:
            if scheme.input_activations is not None:
                key_states = forward_quantize(
                    module, key_states, "k", scheme.input_activations
                )
                value_states = forward_quantize(
                    module, value_states, "v", scheme.input_activations
                )

            if scheme.weights is not None:
                raise ValueError("")

            if scheme.output_activations is not None:
                raise NotImplementedError("")

        if self.past_key_value is not None:
            ret = self.past_key_value.update(key_states, value_states, *args, **kwargs)
            self.past_key_value = None
            return ret
        else:
            return key_states, value_states

    def enable_quantization(self):
        self._quantization_enabled = True


def initialize_hooked_kv_cache(
    model: PreTrainedModel, module: torch.nn.Module, quantize: bool = False
):
    if not hasattr(module, "kv_cache"):
        module.register_module("kv_cache", QuantizedKVCache(module))
        module.register_forward_pre_hook(kv_cache_attention_hook, with_kwargs=True)

    kv_cache: QuantizedKVCache = getattr(module, "kv_cache")
    if quantize and not kv_cache._quantization_enabled:
        # initialize k scale
        # initialize v scale
        kv_cache.enable_quantization()


def kv_cache_attention_hook(module: torch.nn.Module, args, kwargs):
    kv_cache: QuantizedKVCache = getattr(module, "kv_cache")
    kv_cache.past_key_value = kwargs.get("past_key_value", None)
    kwargs["past_key_value"] = kv_cache

    return args, kwargs


def register_key_hook(module: torch.nn.Module, hook: Callable) -> RemovableHandle:
    kv_cache: QuantizedKVCache = getattr(module, "kv_cache")

    def _hook(mod: torch.nn.Module, args, kwargs):
        # If passed as keyword, this is easy.
        if "key_states" in kwargs:
            kwargs["key_states"] = hook(mod, kwargs["key_states"])
            return args, kwargs

        # Otherwise, find where key_states would be in positional args.
        sig = inspect.signature(mod.forward)
        param_names = tuple(sig.parameters.keys())
        try:
            idx = param_names.index("key_states")
        except ValueError:
            # forward has no key_states parameter; do nothing
            return args, kwargs

        # If the position exists in args, replace it.
        if idx < len(args):
            args = list(args)
            args[idx] = hook(mod, args[idx])
            return tuple(args), kwargs

        # Not present positionally and not in kwargs (maybe defaulted) — do nothing.
        return args, kwargs

    return kv_cache.register_forward_pre_hook(_hook, with_kwargs=True)


def register_value_hook(
    module: torch.nn.Module, func: Callable, **kwargs
) -> RemovableHandle:
    kv_cache: QuantizedKVCache = getattr(module, "kv_cache")

    def hook(module: torch.nn.Module, args, kwargs):
        signature = inspect.signature(module.forward)
        bound_args = signature.bind_partial(*args, **kwargs)
        return func(module, bound_args.arguments["value_states"])

    return kv_cache.register_forward_pre_hook(hook, with_kwargs=True)
