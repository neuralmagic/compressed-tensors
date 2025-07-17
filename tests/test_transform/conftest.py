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

import pytest
import torch
from compressed_tensors.modeling.attention import call_attn_impl
from compressed_tensors.transform import TransformArgs, TransformFactory
from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaRotaryEmbedding,
)


class TransformableModel(PreTrainedModel):
    def __init__(self, *sizes):
        super().__init__(config=PretrainedConfig())
        self.fcs = torch.nn.ModuleList(
            [
                torch.nn.Linear(sizes[index], sizes[index + 1], bias=False)
                for index in range(0, len(sizes) - 1)
            ]
        )

    def forward(self, x):
        for layer in self.fcs:
            x = layer(x)
        return x


class MockAttentionModel(PreTrainedModel):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        skip_pos_embeddings: bool = False,
        attn_implementation: str = "eager",
    ):
        config = PretrainedConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_dropout=0.0,
            attention_bias=False,
            max_position_embeddings=128,
            rope_theta=500000.0,
            _attn_implementation_internal=attn_implementation,
            _attn_implementation_autoset=False,
        )
        super().__init__(config)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.attn = LlamaAttention(config, layer_idx=0)
        self.skip_pos_embeddings = skip_pos_embeddings

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.size(1) <= self.config.max_position_embeddings

        if not self.skip_pos_embeddings:
            position_ids = torch.arange(hidden_states.size(1)).unsqueeze(0)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        else:
            zeros = torch.zeros(hidden_states.size(1), dtype=hidden_states.dtype)
            position_embeddings = (zeros, zeros)

        attn_output, _attn_weights = self.attn(
            hidden_states, position_embeddings=position_embeddings, attention_mask=None
        )

        return attn_output


@pytest.fixture(scope="function")
def model_apply():
    model = TransformableModel(2, 4, 8, 16, 32, 64)
    apply = [
        # weight output -> input
        TransformArgs(targets="fcs.0", location="weight_output"),
        TransformArgs(targets="fcs.1", location="input", inverse=True),
        # output -> weight input
        TransformArgs(targets="fcs.1", location="output"),
        TransformArgs(targets="fcs.2", location="weight_input", inverse=True),
        # output -> input
        TransformArgs(targets="fcs.2", location="output"),
        TransformArgs(targets="fcs.3", location="input", inverse=True),
        # weight output -> weight input
        TransformArgs(targets="fcs.3", location="weight_output"),
        TransformArgs(targets="fcs.4", location="weight_input", inverse=True),
    ]

    return model, apply
