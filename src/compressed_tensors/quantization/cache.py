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


from typing import Any, Dict, List, Optional, Tuple

import torch
from compressed_tensors.quantization.observers import Observer
from compressed_tensors.quantization.quant_args import QuantizationArgs
from torch import Tensor
from transformers import DynamicCache as HFDyanmicCache


class QuantizedCache(HFDyanmicCache):
    """
    Quantized KV cache used in the forward call based on HF's dynamic cache.
    Singleton, so that the same cache gets reused in all forward call of self_attn.
    Each time forward is called, .update() is called, and ._quantize(), ._dequantize()
     gets called appropriately.
    Triggered by adding kv_cache_scheme in the recipe.

    Example:

    ```python3
    recipe = '''
    quant_stage:
        quant_modifiers:
            QuantizationModifier:
                kv_cache_scheme:
                    num_bits: 8
                    type: float
                    strategy: tensor
                    dynamic: false
                    symmetric: true
    '''

    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """Singleton"""
        if cls._instance is None:
            cls._instance = super(QuantizedCache, cls).__new__(cls)
        return cls._instance

    def __init__(self, quantization_args: QuantizationArgs):
        if not self._initialized:
            super().__init__()

            self._quantized_key_cache: List[Tensor] = []
            self._quantized_value_cache: List[Tensor] = []

            self.quantization_args = quantization_args

            self.k_observers: List[Observer] = []
            self.v_observers: List[Observer] = []

            self.k_scales: List[
                Tensor
            ] = []  # each index corresponds to layer_idx of the attention layer
            self.v_scales: List[Tensor] = []

            self.k_zps: List[Tensor] = []
            self.v_zps: List[Tensor] = []

            self._initialized = True

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            if len(self.k_observers) <= layer_idx:
                k_observer = self.quantization_args.get_observer()
                v_observer = self.quantization_args.get_observer()

                self.k_observers.append(k_observer)
                self.v_observers.append(v_observer)

            self._quantized_key_cache.append(
                self._quantize(key_states.contiguous(), "key", layer_idx),
            )
            self._quantized_value_cache.append(
                self._quantize(value_states.contiguous(), "value", layer_idx)
            )
            self.key_cache.append(
                torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
            )
            self.value_cache.append(
                torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
            )
            keys_to_return, values_to_return = key_states, value_states
        else:
            dequant_key = self._dequantize(
                self._quantized_key_cache[layer_idx], "key", layer_idx
            )
            dequant_value = self._dequantize(
                self._quantized_value_cache[layer_idx], "value", layer_idx
            )
            keys_to_return = [dequant_key, self.key_cache[layer_idx], key_states]
            values_to_return = [
                dequant_value,
                self.value_cache[layer_idx],
                value_states,
            ]

            keys_to_return = torch.cat(keys_to_return, dim=-2)
            values_to_return = torch.cat(values_to_return, dim=-2)
            if (
                self.key_cache[layer_idx].dim() == 4
                and self.key_cache[layer_idx].shape[-2] + 1 >= self.residual_length
            ):
                self._quantized_key_cache[layer_idx] = self._quantize(
                    keys_to_return.contiguous(), kv_type="key"
                )
                self._quantized_value_cache[layer_idx] = self._quantize(
                    values_to_return.contiguous(), kv_type="value"
                )
                self.key_cache[layer_idx] = torch.zeros(
                    0, dtype=key_states.dtype, device=key_states.device
                )
                self.value_cache[layer_idx] = torch.zeros(
                    0, dtype=key_states.dtype, device=key_states.device
                )
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )

        return keys_to_return, values_to_return

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        Returns the sequence length of the cached states.
        A layer index can be optionally passed.
        """
        if len(self.key_cache) <= layer_idx:
            return 0
        # since we cannot get the seq_length of each layer directly and
        # rely on `_seen_tokens` which is updated every "layer_idx" == 0,
        # this is a hack to get the actual seq_length for the given layer_idx
        # this part of code otherwise fails when used to
        # verify attn_weight shape in some models
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1

    def reset_states(self):
        """reset the kv states (used in calibration)"""
        self.key_cache: List[Tensor] = []
        self.value_cache: List[Tensor] = []
        self._seen_tokens = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen
        )
        self._quantized_key_cache: List[Tensor] = []
        self._quantized_value_cache: List[Tensor] = []

    def reset(self):
        """
        Reset the instantiation, create new instance on init
        """
        self._instance = None
        self._initialized = False

    def _quantize(self, tensor, kv_type, layer_idx):
        """Quantizes a key/value using a defined quantization method."""
        from compressed_tensors.quantization.lifecycle.forward import (
            process_quantization,
        )

        do_quantize = True
        do_dequantize = False
        if kv_type == "key":  # key type
            observer = self.k_observers[layer_idx]
            scales = self.k_scales
            zps = self.k_zps
        else:
            observer = self.v_observers[layer_idx]
            scales = self.v_scales
            zps = self.v_zps

        scale, zp = observer(tensor)
        if len(scales) <= layer_idx:
            scales.append(scale)
            zps.append(zp)
        else:
            scales[layer_idx] = scale
            zps[layer_idx] = scale

        q_tensor = process_quantization(
            x=tensor,
            scale=scale,
            zero_point=zp,
            args=self.quantization_args,
            do_quantize=do_quantize,
            do_dequantize=do_dequantize,
        )
        return q_tensor

    def _dequantize(self, qtensor, kv_type, layer_idx):
        """Dequantizes back the tensor that was quantized by `self._quantize()`"""
        from compressed_tensors.quantization.lifecycle.forward import (
            process_quantization,
        )

        do_quantize = False
        do_dequantize = True
        if kv_type == "key":
            scale = self.k_scales[layer_idx]
            zp = self.k_zps[layer_idx]
        else:
            scale = self.v_scales[layer_idx]
            zp = self.v_zps[layer_idx]

        qdq_tensor = process_quantization(
            x=qtensor,
            scale=scale,
            zero_point=zp,
            args=self.quantization_args,
            do_quantize=do_quantize,
            do_dequantize=do_dequantize,
        )
        return qdq_tensor
