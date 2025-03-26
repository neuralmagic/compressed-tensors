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

import torch
from compressed_tensors.transforms.hadamard_utils import (
    SingletonHadamardRegistry,
    random_hadamard_matrix,
)


size = 2048
dtype = torch.bfloat16
hadamard_registry = SingletonHadamardRegistry()
deterministic_had = hadamard_registry.get_hadamard(size)
# fetch the deterministic had from the registry, if already precomputed
if deterministic_had is None:
    deterministic_had = random_hadamard_matrix(size=size).to(dtype)
    hadamard_registry.set_hadamard(size, deterministic_had)

out = random_hadamard_matrix(size)
Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
Q = Q * 2 - 1

breakpoint()
new_out = out * Q
new_out = new_out / torch.tensor(size).sqrt()
assert torch.equal(torch.eye(size), torch.round(new_out @ new_out.T))
breakpoint()
