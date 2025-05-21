from typing import List

import torch

from transformers import AutoModelForCausalLM

from compressed_tensors.transforms.transform_config import quipsharp
from compressed_tensors.transforms.hadamard import HadamardFactory
from compressed_tensors.transforms.random_hadamard import RandomHadamardFactory
from compressed_tensors.transforms.base import MatrixTransformBase


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fcs = torch.nn.ModuleList([
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Linear(hidden_size, output_size)
        ])
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fcs[0](x)
        x = self.relu(x)
        x = self.fcs[1](x)
        return x


def apply_transforms(config, model):
    factories = []

    for name, scheme in config.transform_groups.items():
        transform_factory = RandomHadamardFactory(name, scheme, seed=42)
        transform_factory.apply_to_model(model)
        factories.append(transform_factory)

    return factories


def test_apply():
    model = Model(2, 8, 4)
    factories = apply_transforms(quipsharp, model)

    # test shared memory
    #assert model.fcs[0].u_input.weight is model.fcs[0].u_input.weight == 

    breakpoint()

    # # check memory
    # for module in model.modules()
    #     if isinstance(module, MatrixTransformBase):
    #         assert module.weight

test_apply()
exit(0)


factories: List[HadamardFactory] = [
    HadamardFactory(name, scheme, seed=42)
    for name, scheme in quipsharp.transform_groups.items()
]

factories[0].apply_to_model(model)
factories[1].apply_to_model(model)
breakpoint()


