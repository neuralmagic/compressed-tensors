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

import json

import torch


config_file_path = "my_config.json"
x = {
    "u_transform_q_o_down_proj": {
        "transform_type": "random-hadamard",
        "groups": [
            {
                "targets": [
                    "re:.*.attn.q_proj$",
                    "re:.*.attn.o_proj$",
                    "re:.*.mlp.down_proj$",
                ],
                "module_targets": ["weight"],
                "call_args": {},
                "ignore": [],
            }
        ],
        "global_transform": False,
        "transform_creation_args": {"size": 2048, "dtype": "torch.bfloat16"},
    }
}
with open(config_file_path, "w") as config_file:
    json.dump(x, config_file, indent=2, sort_keys=True)
