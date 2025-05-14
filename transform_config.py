class TransformationScheme:
    pass

class TransformsConfig:
    pass

class TransformsArgs:
    pass

class TransformsModifier:
    pass


# quip#
quipsharp = TransformsConfig(
    transform_groups={
        "weights_u": TransformationScheme(
            type="hadamard",
            apply=[
                TransformsArgs(
                    targets=["Linear"],
                    location="input",  # non-mergable
                    inverse=True
                ),
                TransformsArgs(
                    targets=["Linear"],
                    location="weight",
                    side="left",
                ),
            ],
            randomize_modules=True,
        ),
        "weights_v": TransformationScheme(
            type="hadamard",
            apply=[
                TransformsArgs(
                    targets=["Linear"],
                    location="weight",
                    side="right",
                    inverse=True,
                ),
                TransformsArgs(
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
        "R1": TransformationScheme(
            type="hadamard",
            apply=[
                TransformsArgs(
                    targets=["embed_tokens", "o_proj", "down_proj"],
                    location="weight",
                    side="right",
                ),
                TransformsArgs(
                    targets=["q_proj", "k_proj", "v_proj", "up_proj", "gate_proj", "lm_head"],
                    location="weight",
                    side="left",
                    inverse=True
                ),
            ]
        ),
        "R2": TransformationScheme(
            type="hadamard",
            apply=[
                TransformsArgs(
                    targets=["v_proj"],
                    location="weight",
                    side="right",
                ),
                TransformsArgs(
                    targets=["o_proj"],
                    location="weight",
                    side="left",
                    inverse=True
                ),
            ]
        ),
        "R3": TransformationScheme(
            type="hadamard",
            apply=[
                TransformsArgs(
                    targets=["self_attn"],
                    location=["k_cache"],
                ),
                TransformsArgs(
                    targets=["self_attn"],
                    location=["q_attn"],
                )
            ]
        ),
        "R4": TransformationScheme(
            type="hadamard",
            apply=[
                TransformsArgs(
                    targets=["down_proj"],
                    location=["input"],
                ),
                TransformsArgs(
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