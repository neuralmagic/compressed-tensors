from typing import List, Literal, Optional, Dict

class TransformArgs:
    targets: List[str]
    location: Literal["input", "weight", "output"]
    side: Optional[Literal["left", "right"]]
    inverse: bool

class TransformsScheme:
    type: str
    apply: List[TransformArgs]
    randomize_modules: bool
    requires_grad: bool

class TransformsConfig:
    transform_groups: Dict[str, TransformsScheme]

class MatrixTransformFactory:
    def apply_to_model():
        pass

class TransformsModifier:
    def __init__(self, config: TransformsConfig):
        self.config = config
        self.seed = 42
        self.transform_factories = []

    def on_initialize(self):
        for name, scheme in self.config.transform_groups:
            factory = MatrixTransformFactory.from_registry(scheme, (name, scheme, self.seed))
            self.transform_factories.append(factory)

    def on_start(self, model):
        for factory in self.transform_factories:
            factory.apply_to_model(model)

    def on_finalize():
        self.transform_factories = []


# quip#
quipsharp = TransformsConfig(
    transform_groups={
        "weights_u": TransformsScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["Linear"],
                    location="input",  # non-mergable
                    inverse=True
                ),
                TransformArgs(
                    targets=["Linear"],
                    location="weight",
                    side="left",
                ),
            ],
            randomize_modules=True,
        ),
        "weights_v": TransformsScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["Linear"],
                    location="weight",
                    side="right",
                    inverse=True,
                ),
                TransformArgs(
                    targets=["Linear"],
                    location="output",  # non-mergable
                )
            ],
            randomize_modules=True
        ),
    }
)

# spinquant
llama_spinquant = TransformsConfig(
    transform_groups={
        "R1": TransformsScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["embed_tokens", "o_proj", "down_proj"],
                    location="weight",
                    side="right",
                ),
                TransformArgs(
                    targets=["q_proj", "k_proj", "v_proj", "up_proj", "gate_proj", "lm_head"],
                    location="weight",
                    side="left",
                    inverse=True
                ),
            ]
        ),
        "R2": TransformsScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["v_proj"],
                    location="weight",
                    side="right",
                ),
                TransformArgs(
                    targets=["o_proj"],
                    location="weight",
                    side="left",
                    inverse=True
                ),
            ]
        ),
        "R3": TransformsScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["self_attn"],
                    location=["k_cache"],
                ),
                TransformArgs(
                    targets=["self_attn"],
                    location=["q_attn"],
                )
            ]
        ),
        "R4": TransformsScheme(
            type="hadamard",
            apply=[
                TransformArgs(
                    targets=["down_proj"],
                    location=["input"],
                ),
                TransformArgs(
                    targets=["down_proj"],
                    location=["weight"],
                    side="left",
                    inverse=True
                ),
            ]
        ),
    }
)

# llm compressor
TransformsModifier(config=llama_spinquant)

# [transform locations] and {quant locations}:
# input  <- {input @ [input]}
# weight <- {[weight_left] @ weight @ [weight_right]}
# output <- {input @ weight @ [output]}
# [] is a transform
# {} means evaluate inner, then quantize value
# only "weight_left" and "weight_right" locations can be merged, everything else is online

# procedure for handling global/randomized modules
# for each randomize_module, create a new matrix/permutation
# when a transform is registered to a module, check if that module already has a unique matrix/permutation created for that module, otherwise use a global one