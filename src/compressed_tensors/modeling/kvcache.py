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
from typing import Callable, Optional, Tuple

import torch
from compressed_tensors.quantization import QuantizationScheme, forward_quantize
from compressed_tensors.quantization.lifecycle.initialize import (
    _initialize_scale_zero_point,
)
from compressed_tensors.utils import getattr_chain
from compressed_tensors.utils.internal import InternalModule
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from transformers import Cache


__all__ = ["KV_CACHE_ATTR", "QuantizedKVCache"]


KV_CACHE_ATTR = "kv_cache"


class QuantizedKVCache(InternalModule):
    def __init__(self, attn_module: torch.nn.Module):
        super().__init__()
        self.attn_module_container = [attn_module]  # avoid nn.Module circular reference
        self.past_key_value: Optional[Cache] = None
        self._qparams_initialized = False

    def update(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        return self(*args, **kwargs)

    def forward(
        self,
        key_states: Tensor,
        value_states: Tensor,
        *args,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        # quantization
        module = self.attn_module_container[0]
        quant_args_attr = "quantization_scheme.input_activations"
        quant_args = getattr_chain(module, quant_args_attr, None)
        quant_enabled = getattr(module, "quantization_enabled", True)
        if quant_args is not None and quant_enabled and self._qparams_initialized:
            key_states = forward_quantize(module, key_states, "k", quant_args)
            value_states = forward_quantize(module, value_states, "v", quant_args)

        # original cache
        if self.past_key_value is not None:
            ret = self.past_key_value.update(key_states, value_states, *args, **kwargs)
        else:
            ret = (key_states, value_states)

        self.past_key_value = None
        return ret

    def initialize_qparams_once(self, module: torch.nn.Module):
        assert module is self.attn_module_container[0]
        scheme = getattr(module, "quantization_scheme", None)
        quant_args = getattr(scheme, "input_activations", None)

        if not self._qparams_initialized and quant_args is not None:
            _initialize_scale_zero_point(module, "k", quant_args)
            _initialize_scale_zero_point(module, "v", quant_args)
            self._qparams_initialized = True


# ----- initialize ----- #


def initialize_hooked_kv_cache(module: torch.nn.Module, quantize: bool = False):
    if not hasattr(module, KV_CACHE_ATTR):
        module.register_module(KV_CACHE_ATTR, QuantizedKVCache(module))
        module.register_forward_pre_hook(kv_cache_attention_hook, with_kwargs=True)

    kv_cache: QuantizedKVCache = getattr(module, KV_CACHE_ATTR)
    if quantize:
        kv_cache.initialize_qparams_once(module)


def kv_cache_attention_hook(module: torch.nn.Module, args, kwargs):
    kv_cache: QuantizedKVCache = getattr(module, KV_CACHE_ATTR)
    kv_cache.past_key_value = kwargs.get("past_key_value", None)
    kwargs["past_key_value"] = kv_cache

    return args, kwargs


# ----- hooks ----- #


def register_key_hook(module: torch.nn.Module, hook: Callable) -> RemovableHandle:
    kv_cache: QuantizedKVCache = getattr(module, KV_CACHE_ATTR)

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
            ret = hook(module, args[idx])
            if ret is not None:
                args[idx] = ret
            return tuple(args), kwargs

        # Not present positionally and not in kwargs (maybe defaulted) — do nothing.
        return args, kwargs

    return kv_cache.register_forward_pre_hook(_hook, with_kwargs=True)


def register_value_hook(module: torch.nn.Module, hook: Callable) -> RemovableHandle:
    kv_cache: QuantizedKVCache = getattr(module, KV_CACHE_ATTR)

    def _hook(mod: torch.nn.Module, args, kwargs):
        # If passed as keyword, this is easy.
        if "value_states" in kwargs:
            kwargs["value_states"] = hook(mod, kwargs["value_states"])
            return args, kwargs

        # Otherwise, find where value_states would be in positional args.
        sig = inspect.signature(mod.forward)
        param_names = tuple(sig.parameters.keys())
        try:
            idx = param_names.index("value_states")
        except ValueError:
            # forward has no value_states parameter; do nothing
            return args, kwargs

        # If the position exists in args, replace it.
        if idx < len(args):
            args = list(args)
            ret = hook(module, args[idx])
            if ret is not None:
                args[idx] = ret
            return tuple(args), kwargs

        # Not present positionally and not in kwargs (maybe defaulted) — do nothing.
        return args, kwargs

    return kv_cache.register_forward_pre_hook(_hook, with_kwargs=True)
