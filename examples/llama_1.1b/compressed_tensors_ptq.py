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

from tqdm import tqdm

from sparsetensors.quantization import (
    apply_quantization_config,
    freeze_module_quantization,
    QuantizationConfig,
    QuantizationStatus,
)
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from sparseml.transformers.finetune.data.base import TextGenerationDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

config_file = "example_quant_config.json"
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
dataset_name = "open_platypus"
split = "train"
num_calibration_samples = 10
max_seq_length = 1024
pad_to_max_length = False


model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()  # no grad or updates needed for base model
config = QuantizationConfig.parse_file(config_file)

# set status to calibration
config.quantization_status = QuantizationStatus.CALIBRATION

# initialize quantization
apply_quantization_config(model, config)

# create dataset
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_args = DataTrainingArguments(
    dataset=dataset_name,
    dataset_config_name="main",
    max_seq_length=max_seq_length,
    pad_to_max_length=pad_to_max_length,
)
dataset_manager = TextGenerationDataset.load_from_registry(
    data_args.dataset,
    data_args=data_args,
    split=split,
    tokenizer=tokenizer,
)
calib_dataset = dataset_manager.tokenize_and_process(
    load_dataset(dataset_name, split=split)
)

# run calibration
for _ in tqdm(num_calibration_samples(10)):
    _ = model(**tokenizer("", return_tensors="pt"))

# freeze params after calibration
model.apply(freeze_module_quantization)

# TODO: save