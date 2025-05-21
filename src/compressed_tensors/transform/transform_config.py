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

from compressed_tensors.transform import TransformArgs, TransformScheme
from pydantic import BaseModel


__all__ = ["TransformConfig"]


class TransformConfig(BaseModel):
    """
    Configuration of transforms to be applied to a model. This config is to be
    serialized within a model's `config.json` file

    :param transform_groups: A dictionary of `TransformSchemes` that should be applied
        to a particular model. The keys can be any arbitrary string
    """

    transform_groups: Dict[str, TransformScheme]


# quip / quip sharp
QUIP = TransformConfig(
    transform_groups={
        "v": TransformScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["Linear"],
                    location="input",  # non-mergable
                ),
                TransformArgs(
                    targets=["Linear"],
                    location="weight",
                    side="input",
                    inverse=True,
                ),
            ],
            randomize_modules=True,
        ),
        "u": TransformScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["Linear"],
                    location="weight",
                    side="output",
                ),
                TransformArgs(
                    targets=["Linear"], location="output", inverse=True  # non-mergable
                ),
            ],
            randomize_modules=True,
        ),
    }
)

# spinquant
LLAMA_SPINQUANT = TransformConfig(
    transform_groups={
        "R1": TransformScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["embed_tokens", "o_proj", "down_proj"],
                    location="weight",
                    side="output",
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
                    side="input",
                    inverse=True,
                ),
            ],
        ),
        "R2": TransformScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["v_proj"],
                    location="weight",
                    side="output",
                ),
                TransformArgs(
                    targets=["o_proj"], location="weight", side="input", inverse=True
                ),
            ],
        ),
        "R3": TransformScheme(
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
        "R4": TransformScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["down_proj"],
                    location="input",
                ),
                TransformArgs(
                    targets=["down_proj"], location="weight", side="input", inverse=True
                ),
            ],
        ),
    }
)


PRESET_CONFIGS = {
    "QUIP": QUIP,
    "LLAMA_SPINQUANT": LLAMA_SPINQUANT,
}
