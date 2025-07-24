import torch
from transformers import AutoModelForCausalLM

# Load the model
model_id = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

# Find the first up_proj and down_proj weights
up_proj_weight = None
down_proj_weight = None

for name, module in model.named_modules():
    if up_proj_weight is not None and down_proj_weight is not None:
        break
    if hasattr(module, "up_proj") and up_proj_weight is None:
        up_proj_weight = module.up_proj.weight.detach().cpu()
    if hasattr(module, "down_proj") and down_proj_weight is None:
        down_proj_weight = module.down_proj.weight.detach().cpu()

assert up_proj_weight is not None and down_proj_weight is not None, "Projection weights not found!"

# Save to file
torch.save({
    "up_proj_weight": up_proj_weight,
    "down_proj_weight": down_proj_weight
}, "proj_weights.pt")

print("Saved weights to proj_weights.pt")

# Example: Loading into new Linear layers
state = torch.load("proj_weights.pt")
print(state["up_proj_weight"].shape)
print(state["down_proj_weight"].shape)

up_layer = torch.nn.Linear(state["up_proj_weight"].size(1), state["up_proj_weight"].size(0), bias=False)
up_layer.weight.data.copy_(state["up_proj_weight"])

down_layer = torch.nn.Linear(state["down_proj_weight"].size(1), state["down_proj_weight"].size(0), bias=False)
down_layer.weight.data.copy_(state["down_proj_weight"])

print("Restored weights into Linear layers.")
