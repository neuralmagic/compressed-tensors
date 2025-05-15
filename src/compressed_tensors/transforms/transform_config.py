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

from typing import Dict

from compressed_tensors.transforms.transform_args import TransformArgs
from compressed_tensors.transforms.transform_scheme import TransformsScheme
from pydantic import BaseModel


__all__ = ["TransformationConfig"]


class TransformsConfig(BaseModel):
    transform_groups: Dict[str, TransformsScheme]


quipsharp = TransformsConfig(
    transform_groups={
        "u": TransformsScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["Linear"], location="input", inverse=True  # non-mergable
                ),
                TransformArgs(
                    targets=["Linear"],
                    location="weight",
                    side="left",
                ),
            ],
            randomize_modules=True,
        ),
        "v": TransformsScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["Linear"],
                    location="weight",
                    side="right",
                    inverse=True,
                ),
                TransformArgs(
                    targets=["Linear"],
                    location="output",  # non-mergable
                ),
            ],
            randomize_modules=True,
        ),
    }
)

# spinquant
llama_spinquant = TransformsConfig(
    transform_groups={
        "R1": TransformsScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["embed_tokens", "o_proj", "down_proj"],
                    location="weight",
                    side="right",
                ),
                TransformArgs(
                    targets=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "up_proj",
                        "gate_proj",
                        "lm_head",
                    ],
                    location="weight",
                    side="left",
                    inverse=True,
                ),
            ],
        ),
        "R2": TransformsScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["v_proj"],
                    location="weight",
                    side="right",
                ),
                TransformArgs(
                    targets=["o_proj"], location="weight", side="left", inverse=True
                ),
            ],
        ),
        "R3": TransformsScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["self_attn"],
                    location="k_cache",
                ),
                TransformArgs(
                    targets=["self_attn"],
                    location="q_attn",
                ),
            ],
        ),
        "R4": TransformsScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["down_proj"],
                    location="input",
                ),
                TransformArgs(
                    targets=["down_proj"], location="weight", side="left", inverse=True
                ),
            ],
        ),
    }
)
