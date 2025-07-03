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

from typing import Optional

import torch
from compressed_tensors.transform import TransformBase, TransformLocation
from compressed_tensors.utils import patch_attr
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


"""
Attention interfaces are functions with the following signature
module, query, key, value, attention_mask, scaling, dropout, **kwargs
They're gotten `from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS`

Idea: Yield a custom attention function which injects 

Pros: relatively simple
Cons: ordering is hard, since submodules aren't ordered; a little harder if you want
      to do stuff like attention output hooks
We can just disable multiple attention transforms for now
"""

original_get_item = ALL_ATTENTION_FUNCTIONS.__getitem__


def make_hooked_attention(key):
    def hooked_attention(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        for submodule in module.children():
            if isinstance(submodule, TransformBase):
                if TransformBase.args.location == TransformLocation.Q_ATTN:
                    query = submodule(query)

                if TransformBase.args.location == TransformLocation.K_CACHE:
                    key = submodule(key)

        return original_get_item(key)(
            module, query, key, value, attention_mask, scaling, dropout, **kwargs
        )

    return hooked_attention


_cache = {}


def patched_get_item(self, key):
    if key not in _cache:
        _cache[key] = make_hooked_attention(key)

    return _cache[key]


patch_attr(ALL_ATTENTION_FUNCTIONS, "__getitem__", patched_get_item)
