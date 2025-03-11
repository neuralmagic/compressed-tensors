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
from compressed_tensors.quantization import process_transforms_config
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
    module_targets = [ModuleTarget.WEIGHT]
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
    module_targets = [ModuleTarget.WEIGHT]
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


def _verify_correct_data(layer: torch.nn.Module):
    assert hasattr(layer, "transform_data")
    assert isinstance(layer.transform_data, TransformData)

    # data keys are all the different transforms relevant
    # to the module
    transform_data = layer.transform_data

    for k, v in transform_data.data.items():
        current_transform = getattr(layer, k)
        assert isinstance(current_transform, torch.nn.Parameter)
        assert "call_args" in v


@pytest.mark.skip(reason="Skipping until activation transforms are supported")
def test_recipe_complex(basic_model, transform_recipe_complex):
    model = process_transforms_config(
        model=basic_model, transforms_config=transform_recipe_complex
    )

    blocks = [model.block1, model.block2]
    for block in blocks:
        for layer in block:
            if isinstance(layer, torch.nn.Linear):
                _verify_correct_data(layer)


def test_recipe_basic(basic_model, transform_recipe_basic):
    model = process_transforms_config(
        model=basic_model, transforms_config=transform_recipe_basic
    )

    blocks = [model.block1, model.block2]
    for block in blocks:
        for layer in block:
            if isinstance(layer, torch.nn.Linear):
                _verify_correct_data(layer)


def test_recipe_complex_multiple(basic_model, transform_recipe_complex_multiple):
    model = process_transforms_config(
        model=basic_model, transforms_config=transform_recipe_complex_multiple
    )

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
                'call_args': defaultdict()
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
