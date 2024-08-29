import torch
import traceback
from transformers import LlamaTokenizer, LlamaForCausalLM
# from loguru import logger
from compressed_tensors.quantization.utils.helpers import extract_submodule_generator
from compressed_tensors._cache import AttetnionKVLinker
from torch import Tensor

# Load pre-trained LLaMA model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# #####################
# from torch.nn import Module, Parameter
# from transformers import QuantizedCache

# @Cache.register("quantize-kvcache")
# class KVCache(QuantizedCache, RegistryMixin): # nn.Module
#     """
#     For each key there is a kv cache
#     """
#     def __init__(self, module: Module, name: str):
#         super().__init__()
#         self.attn_module = module
#         self.name = name
#         self._k_scale = Parameter(
#             ...
#         )
#         self.observer = ... # either name or object
        
#         self._link(module, self._k_scale)
        
#     @k_scale.setter()
#     def k_scale(self, param):
#         self._k_scale = param
    
#     @property
#     def k_scale(self):
#         return self._k_scale
        
    
#     def quantize(): ...
#     def dequantize(): ...
#     def forward(self): 
#         self.quantize()
#         self.dequantize()

#     def inject(self, attn_module: Module, k_scale: Parameter):
#         attn_module.register_parameter(
#             k_scale
#         )
#         logger.debug(f"set kv cache for {self.name}")

names = []
for name, submodule in extract_submodule_generator(model, "self_attn"):
    # print(name, submodule)
    names.append(name)
    kv_cache = AttetnionKVLinker(submodule, name)
    AttetnionKVLinker.register_value(value=kv_cache, name=name)
    
for name in names:
    cache = AttetnionKVLinker.get_value_from_registry(name)
    print(cache)
    
    
tensor = torch.tensor([1, 2, 3, 4, 5])
cache.k_scale = tensor
print(model.model.layers[31].self_attn.k_scale)
breakpoint()


model.model.layers[31].self_attn.data
    

    
    # KVCache.register_value
    
    # ### initialize
    
    # kv_cache = KVCache.get_from_registry(name)
    # |