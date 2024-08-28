

from transformers import QuantizedCache, QuantizedCacheConfig, DynamicCache
from transformers import DynamicCache
from torch.nn import Tensor
import torch


from typing import Optional, Dict, Tuple, Any, List

class KVCacheConfig(QuantizedCacheConfig):
    backend: str = "compressed-tensors", 
    nbits: Optional[int] = 4,
    axis_key: Optional[int] = 0,
    axis_value: Optional[int] = 0,
    q_group_size: Optional[int] = 64,
    residual_length: Optional[int] = 128,
    compute_dtype: Optional[torch.dtype] = torch.float16,
    device: Optional[str] = "cpu",
    
    ...
    
    
class QuantizedKVCache(QuantizedCache):
    def __init__(self, config: KVCacheConfig):
        super().__init__(self, config)
        
        self.k_scale = List[Tensor] = [] # each index corresponds to layer_idx of the attention layer
        self.v_scale = List[Tensor] = []
        
        self.observer = ...
        
        # self._quantized_key_cache: List[torch.Tensor] = []
        # self._quantized_value_cache: List[torch.Tensor] = []

        # self.nbits = cache_config.nbits
        # self.residual_length = cache_config.residual_length
        # self.q_group_size = cache_config.q_group_size
        # self.axis_key = cache_config.axis_key
        # self.axis_value = cache_config.axis_value
        # self.compute_dtype = cache_config.compute_dtype
        # self.device = cache_config.device
        
        
    # def update(self, layer_idx ):
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # save the key_states and value_states in their cache
        rtn = super().update(key_states, value_states, layer_idx, cache_kwargs) 
        
        if len(self.key_cache) <= layer_idx:
            k_scale, v_scale = self._quantize(...)
            self._k_scale.append(k_scale)
            self._v_scale.append(v_scale)
        
        return rtn
            
        


    def _quantize(self, key_state, value_state, axis):
        """Quantizes a key/value using a defined quantization method."""
        # update k_scale, v_scale based on the output of the observer
        
        # quantize the layer, update the _quantized_key_cache and 
        
        k_scale = self.key_observer(key_state, ...)
        v_scale = self.value_observer(value_state, ...)
        
        return k_scale, v_scale
       

    def _dequantize(self, q_tensor):
        """Dequantizes back the tensor that was quantized by `self._quantize()`"""
        
        
    
    

# class QuantizedCache(HFQuantizedCache):
#     def __init__(self, config: Optional[HFCacheConfig] = None):
#         config = config or HFCacheConfig()
#         super().__init__(config)
    
#     def _quantize(self, tensor, axis):
#         ...
        
#     def dequantize(self, q_tensor):
#         ...