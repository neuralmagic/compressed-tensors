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

from typing import Tuple

from torch.nn import Module


__all__ = ["is_module_quantized", "iter_named_leaf_modules", "module_type"]


def is_module_quantized(module: Module) -> bool:
    return hasattr(module, "quantization_scheme")


def module_type(module: Module) -> str:
    return type(module).__name__


def iter_named_leaf_modules(model: Module) -> Tuple[str, Module]:
    # yields modules that do not have any submodules
    # TODO: potentially expand to add list of allowed submodules such as observers
    for name, submodule in model.named_modules():
        if len(list(submodule.children())) == 0:
            yield name, submodule
