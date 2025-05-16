from typing import List

import torch

from compressed_tensors.transforms.transform_config import quipsharp
from compressed_tensors.transforms.hadamard import HadamardFactory

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
    

model = Model(2, 8, 4)


factories: List[HadamardFactory] = [
    HadamardFactory(name, scheme, seed=42)
    for name, scheme in quipsharp.transform_groups.items()
]

factories[0].apply_to_model(model)
factories[1].apply_to_model(model)
breakpoint()