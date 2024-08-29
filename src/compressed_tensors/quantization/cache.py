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
from compressed_tensors.quantization.lifecycle.forward import process_quantization
from compressed_tensors.quantization.observers import Observer
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.registry import RegistryMixin
from torch import Tensor
from transformers import DynamicCache as HFDyanmicCache


class QuantizedCache(HFDyanmicCache, RegistryMixin):
    """
    Quantized KV cache used in the forward call based on HF's dynamic cache.
    Each time forward is called, .update() is called, and ._quantize(), ._dequantize()
    gets called appropriately
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

    def __init__(self, quantization_args: QuantizationArgs):
        super().__init__()

        self.quantization_args = quantization_args

        self.k_observers = List[Observer] = []
        self.v_observers = List[Observer] = []

        self.k_scales = List[
            Tensor
        ] = []  # each index corresponds to layer_idx of the attention layer
        self.v_scales = List[Tensor] = []

        self.k_zps: List[Tensor] = []
        self.v_zps: List[Tensor] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            k_observer = self.quantization_args.get_observer()
            v_observer = self.quantization_args.get_observer()

            self.k_observers.append(k_observer)
            self.v_observers.append(v_observer)

            self._quantized_key_cache.append(
                self._quantize(key_states.contiguous(), axis=self.axis_key), layer_idx
            )
            self._quantized_value_cache.append(
                self._quantize(value_states.contiguous(), axis=self.axis_value),
                layer_idx,
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
                self._quantized_key_cache[layer_idx], self.axis_key, layer_idx
            )
            dequant_value = self._dequantize(
                self._quantized_value_cache[layer_idx], self.axis_value, layer_idx
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
                    keys_to_return.contiguous(), axis=self.axis_key
                )
                self._quantized_value_cache[layer_idx] = self._quantize(
                    values_to_return.contiguous(), axis=self.axis_value
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

    def _quantize(self, tensor, axis, layer_idx):
        """Quantizes a key/value using a defined quantization method."""
        do_quantize = True
        do_dequantize = False
        if axis == self.axis_key:  # key
            observer = self.k_observers[layer_idx]
            scales = self.k_scales
            zps = self.k_zps
        else:
            observer = self.v_observers[layer_idx]
            scales = self.v_scales
            zps = self.v_zps

        scale, zp = observer(tensor)
        scales.append(scale)
        zps.append(zp)

        q_tensor = process_quantization(
            x=tensor,
            scale=scale,
            zero_point=zp,
            args=self.quantization_args,
            do_quantize=do_quantize,
            do_dequantize=do_dequantize,
        )
        return q_tensor

    def _dequantize(self, qtensor, axis, layer_idx):
        """Dequantizes back the tensor that was quantized by `self._quantize()`"""

        do_quantize = False
        do_dequantize = True
        if axis == self.axis_key:
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
