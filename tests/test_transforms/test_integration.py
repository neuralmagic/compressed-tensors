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
from compressed_tensors.transforms import Hadamard, Transforms
from compressed_tensors.transforms.transform_args import (
    ModuleTarget,
    TransformationArgs,
)
from compressed_tensors.transforms.transform_config import TransformationConfig
from compressed_tensors.transforms.transform_data import TransformData
from compressed_tensors.transforms.transform_scheme import TransformationScheme


@pytest.fixture
def transform_recipe_basic():
    targets = ["Embedding"]
    module_targets = [ModuleTarget.OUTPUT_ACTIVATIONS]
    embedding_args = TransformationArgs(targets=targets, module_targets=module_targets)

    targets = ["Linear"]
    module_targets = [ModuleTarget.WEIGHTS]
    linear_layer_args = TransformationArgs(
        targets=targets, module_targets=module_targets
    )

    scheme = TransformationScheme(
        transform_type="hadamard",
        groups=[embedding_args, linear_layer_args],
        transform_creation_args={"size": 64},
    )

    config = TransformationConfig(
        transform_groups={
            "transform_0": scheme,
        }
    )
    return config


@pytest.fixture
def transform_recipe_complex():
    # Multiple transform for a given module 
    # Weight Activations and Weight Activations for a given module
    pass


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


def test_basic_model(basic_model, transform_recipe_basic):
    transform_groups = transform_recipe_basic.transform_groups
    for _, group in transform_groups.items():
        # Each group/scheme targets one type of transform
        transform_type = group.transform_type
        transform_creation_args = group.transform_creation_args

        global_transform = None
        if group.global_transform:
            # create transform
            global_transform = Transforms.load_from_registry(
                transform_type, **transform_creation_args
            )

        # Need a better name - too many groups
        for transform_arg in group.groups:
            target = transform_arg.targets
            module_targets = transform_arg.module_targets
            call_args = transform_arg.call_args

            if global_transform is None:
                transform = Transforms.load_from_registry(
                    transform_type, **transform_creation_args
                )
                transform

            for _, submodule in basic_model.named_modules():
                name = submodule.__class__.__name__
                if name in target:
                    transform_data = TransformData(
                        data={
                            module_targets[0]: {
                                "transform": transform,
                                "call_args": call_args,
                            }
                        }
                    )
                    submodule.transform_data = transform_data

    # Should have the following structure:
    """
    >> basic_model.embedding.transform_data
    TransformData(
        data={
            <ModuleTarget.OUTPUT_ACTIVATIONS: 'output_activations'>: 
                {'transform': <compressed_tensors.transforms.hadamard.Hadamard object at 0x7ac77ef3fa00>, 
                'call_args': None
            }
        }
    )

    >> basic_model.block1[0].transform_data
    TransformData(
        data={
            <ModuleTarget.WEIGHTS: 'weights'>: 
                {'transform': <compressed_tensors.transforms.hadamard.Hadamard object at 0x7ac77ef3f940>, 
                'call_args': None
            }
        }
    )
    """

    # Verify Embedding layers and Linear Layers have the correct data attached to them
    assert hasattr(basic_model.embedding, "transform_data")
    assert isinstance(basic_model.embedding.transform_data, TransformData)
    activation_transform = basic_model.embedding.transform_data.data.get(
        "output_activations"
    )
    assert isinstance(activation_transform.get("transform"), Hadamard)

    blocks = [basic_model.block1, basic_model.block2]
    for block in blocks:
        for layer in block:
            if isinstance(layer, torch.nn.Linear):
                assert hasattr(layer, "transform_data")
                assert isinstance(layer.transform_data, TransformData)
                weight_transform = layer.transform_data.data.get("weights")
                assert isinstance(weight_transform.get("transform"), Hadamard)
