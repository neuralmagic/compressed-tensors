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
from transformers import AttentionInterface
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


# TODO: HF acknowledgement
def transformable_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """
    Hook to potentially call transforms
    """
    from compressed_tensors.transform import TransformBase, TransformLocation

    for submodule in module.children():
        if isinstance(submodule, TransformBase):
            if TransformBase.args.location == TransformLocation.Q_ATTN:
                query = submodule(query)

            if TransformBase.args.location == TransformLocation.K_CACHE:
                key = submodule(key)

    return ALL_ATTENTION_FUNCTIONS["sdpa"](
        module, query, key, value, attention_mask, scaling, dropout, **kwargs
    )  # TODO: use original setting from config


# another way to do this, to get around random AttentionInterface register and messing with the config
# would be to just patch the ALL_ATTENTION_FUNCTIONS
# we already have to patch the config's attn_implementation anyways

AttentionInterface.register("transformable_attention", transformable_attention)
# model.config.attn_implementation = "transformable_attention"
