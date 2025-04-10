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

from safetensors import safe_open


model_path = "/home/dsikka/llm-compressor/examples/quantization_kv_cache/TinyLlama-1.1B-Chat-v1.0-W4A16-G128-Asym-Updated"
# model_path = "/home/dsikka/llm-compressor/examples/quantization_kv_cache/TinyLlama-1.1B-Chat-v1.0-W4A16-G128-Asym-Updated-Channel"

tensors = {}
with safe_open(f"{model_path}/model.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

for k, v in tensors.items():
    if "zero_point" in k:
        print(k, v.shape)
    if "shape" in k:
        print(k, v)

    print("\n")
