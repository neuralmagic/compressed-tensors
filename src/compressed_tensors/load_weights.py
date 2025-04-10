from safetensors import safe_open

model_path = "/home/dsikka/llm-compressor/examples/quantization_kv_cache/TinyLlama-1.1B-Chat-v1.0-W4A16-G128-Asym-Updated"
#model_path = "/home/dsikka/llm-compressor/examples/quantization_kv_cache/TinyLlama-1.1B-Chat-v1.0-W4A16-G128-Asym-Updated-Channel"

tensors = {}
with safe_open(f"{model_path}/model.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

for k, v in tensors.items():
    if "zero_point" in k:
        print(k, v.shape)
    if "shape" in k:
        print(k, v)
    
    print("\n")