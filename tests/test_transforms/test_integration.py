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
import torch.nn as nn
from compressed_tensors.transforms import Hadamard, RandomHadamard, Transforms
from compressed_tensors.transforms.transform_args import (
    ModuleTarget,
    TransformationArgs,
)
from compressed_tensors.transforms.transform_config import TransformationConfig
from compressed_tensors.transforms.transform_data import TransformData
from compressed_tensors.transforms.transform_scheme import TransformationScheme


@pytest.fixture
def transform_recipe_basic():
    targets = ["Linear"]
    module_targets = [ModuleTarget.WEIGHTS]
    linear_layer_args = TransformationArgs(
        targets=targets, module_targets=module_targets
    )

    scheme = TransformationScheme(
        transform_type="hadamard",
        groups=[linear_layer_args],
        transform_creation_args={"size": 64},
    )
    config = TransformationConfig(
        transform_groups={
            "transform_0": scheme,
        }
    )
    return config


@pytest.fixture
def transform_recipe_complex_multiple(transform_recipe_basic):
    targets = ["Embedding"]
    module_targets = [ModuleTarget.OUTPUT_ACTIVATIONS]
    embedding_args = TransformationArgs(targets=targets, module_targets=module_targets)

    scheme = TransformationScheme(
        transform_type="hadamard",
        groups=[embedding_args],
        transform_creation_args={"size": 128},
    )
    transform_recipe_basic.transform_groups["transform_1"] = scheme
    return transform_recipe_basic


@pytest.fixture
def transform_recipe_complex(transform_recipe_basic):
    targets = ["Linear"]
    module_targets = [ModuleTarget.OUTPUT_ACTIVATIONS]
    linear_layer_args = TransformationArgs(
        targets=targets, module_targets=module_targets
    )

    scheme = TransformationScheme(
        transform_type="random-hadamard",
        groups=[linear_layer_args],
        transform_creation_args={"size": 64},
    )
    transform_recipe_basic.transform_groups["transform_1"] = scheme
    return transform_recipe_basic


@pytest.fixture
def basic_model():
    class BasicModel(nn.Module):
        def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
            super(BasicModel, self).__init__()

            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.block1 = nn.Sequential(
                nn.Linear(embed_size, hidden_size), nn.ReLU(), nn.Dropout(0.2)
            )
            self.block2 = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.2)
            )
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x = self.embedding(x)
            x = x.mean(dim=1)
            x = self.block1(x)
            x = self.block2(x)
            x = self.fc(x)

            return x

    vocab_size = 1000
    embed_size = 128
    hidden_size = 64
    num_classes = 10
    return BasicModel(vocab_size, embed_size, hidden_size, num_classes)


def _apply_transfoms_to_model(model, transform_groups):
    for _, group in transform_groups.items():
        # Each group/scheme targets one type of transform
        transform_type = group.transform_type
        transform_creation_args = group.transform_creation_args

        # Need a better name - too many groups
        for transform_arg in group.groups:
            target = transform_arg.targets
            module_targets = transform_arg.module_targets
            call_args = transform_arg.call_args

            transform = Transforms.load_from_registry(
                transform_type, **transform_creation_args
            )
            apply = Transforms.fetch_apply(transform_type)

            for _, submodule in model.named_modules():
                name = submodule.__class__.__name__
                if name in target:
                    # attach the transform to the submodule
                    transform_name = f"{module_targets[0]}_transform"
                    setattr(submodule, transform_name, transform)

                    # add relevant transform data to the submodule as well
                    data = {transform_name: {"apply": apply, "call_args": call_args}}

                    if hasattr(submodule, "transform_data"):
                        submodule.transform_data.data.update(data)
                    else:
                        transform_data = TransformData(data=data)
                        submodule.transform_data = transform_data
    return model


def _verify_correct_data(layer: torch.nn.Module):
    assert hasattr(layer, "transform_data")
    assert isinstance(layer.transform_data, TransformData)

    # data keys are all the different transforms relevant
    # to the module
    transform_data = layer.transform_data

    for k, v in transform_data.data.items():
        current_transform = getattr(layer, k)
        assert isinstance(current_transform, torch.nn.Parameter)
        assert "apply" in v
        assert "call_args" in v


def test_recipe_complex(basic_model, transform_recipe_complex):
    transform_groups = transform_recipe_complex.transform_groups
    model = _apply_transfoms_to_model(basic_model, transform_groups)

    blocks = [model.block1, model.block2]
    for block in blocks:
        for layer in block:
            if isinstance(layer, torch.nn.Linear):
                _verify_correct_data(layer)


def test_recipe_basic(basic_model, transform_recipe_basic):
    transform_groups = transform_recipe_basic.transform_groups
    model = _apply_transfoms_to_model(basic_model, transform_groups)

    blocks = [model.block1, model.block2]
    for block in blocks:
        for layer in block:
            if isinstance(layer, torch.nn.Linear):
                _verify_correct_data(layer)


def test_recipe_complex_multiple(basic_model, transform_recipe_complex_multiple):
    transform_groups = transform_recipe_complex_multiple.transform_groups
    model = _apply_transfoms_to_model(basic_model, transform_groups)

    # Should have the following structure:
    """
    >> basic_model.embedding.output_activations_transform
    Parameter containing:
        tensor([[ 1.,  1.,  1.,  ...,  1.,  1.,  1.],
                [ 1., -1.,  1.,  ..., -1.,  1., -1.],
                [ 1.,  1., -1.,  ...,  1., -1., -1.],
                ...,
                [ 1., -1.,  1.,  ..., -1.,  1., -1.],
                [ 1.,  1., -1.,  ...,  1., -1., -1.],
                [ 1., -1., -1.,  ..., -1., -1.,  1.]], dtype=torch.bfloat16)
    
    >> model.embedding.transform_data
        TransformData(data={'output_activations_transform': 
            {
                'apply': <function Hadamard.apply at 0x7c7bcc56caf0>, 
                'call_args': None
            }
        }
    )
    """

    # Verify Embedding layers and Linear Layers have the correct data attached to them
    _verify_correct_data(model.embedding)

    blocks = [model.block1, model.block2]
    for block in blocks:
        for layer in block:
            if isinstance(layer, torch.nn.Linear):
                _verify_correct_data(layer)
