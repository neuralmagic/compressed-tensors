
from transformers import CacheConfig, QuantizedCache


class KVCache(CacheConfig):
    """
    Save k_proj, v_proj values into the cache
    use its values and write to layer.attention layers
    
    before:
    model.layers.0.self_attn.k_proj.output_scale
    
    after:
    model.layers.0.self_attn.k_scale
    
    """
    def __init__(self, k_scale, v_scale):
        
        self.k_scale = k_scale
        self.v_scale = v_scale
        self.observer = None
        self.state = ""

    def _quantize(self, tensor):
        if self.state == "calibrate":
            self.k_scale.data = self.k_observer(tensor)
            self.v_scale.data = self.v_observer(tensor)
            
        return quantize(tensor, self.k_scale, self.v_scale)
            
    def _dequantize():
        ...
        
        
    
