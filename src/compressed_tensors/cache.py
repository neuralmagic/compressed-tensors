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

from enum import Enum

from compressed_tensors.registry import RegistryMixin
from transformers import Cache as HFCache
from torch.nn import Module, Parameter
import torch


class AttentionLayerCache(HFCache, RegistryMixin):
    
    def __init__(self, module: Module, name: str):
        super().__init__()
        self.attn_module = module
        self.name = name
        self.observer = None
        
        self._k_scale = Parameter(
            torch.empty(1),
            requires_grad=False,
        )
        self._v_scale = Parameter(
            torch.empty(1),
            requires_grad=False,
        )
        
        self._register()
        self._reference()
        
    @property
    def k_scale(self):
        return self._k_scale
    
    @property
    def v_scale(self):
        return self._v_scale
    
    @k_scale.setter
    def k_scale(self, tensor):
        self._k_scale.data = tensor
        
    @v_scale.setter
    def v_scale(self, tensor):
        self._v_scale.data = tensor
    
    def link(self, leaf_module: Module, target: str):
        """
        Link the leaf module's scales to the appropiate self_attn module's
        k_scale or v_scale
        """
        basename = target.split(".")[-1] # one of k_proj, v_proj, qkv_proj
        
        if basename == "k_proj":
            self.attn_module.k_scale = leaf_module.out_scale.data
        elif basename == "v_proj":
            self.attn_module.v_scale = leaf_module.out_scale.data
        else:
            ... 
            # check how qkv proj is organized
        
        
        
        
        

    def quantize(self, tensor):
        ...
        # scale, zp = self.observer(tensor)
        # self._k_scale = scale

    def dequentize(self):
        ...
        
    def _register(self):
        self.attn_module.register_parameter("k_scale", self._k_scale)
        self.attn_module.register_parameter("v_scale", self._v_scale)
        
        
    def _reference(self):
        self.attn_module.k_scale = self._k_scale
        self.attn_module.v_scale = self._v_scale



