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


from setuptools import setup, find_packages
from typing import List, Dict, Tuple

def _setup_long_description() -> Tuple[str, str]:
    return open("README.md", "r", encoding="utf-8").read(), "text/markdown"

def _setup_packages() -> List:
    return find_packages(
        "src", include=["compressed_tensors", "compressed_tensors.*"], exclude=["*.__pycache__.*"]
    )
    
def _setup_install_requires() -> List:
    return ["torch>=1.7.0", "transformers<4.41", "pydantic<2.7"]

def _setup_extras() -> Dict:
    return {"dev": ["black==22.12.0", "isort==5.8.0", "wheel>=0.36.2", "flake8>=3.8.3", "pytest>=6.0.0", "nbconvert>=7.16.3"]}

setup(
    name="compressed-tensors",
    version="0.3.1",
    author="Neuralmagic, Inc.",
    author_email="support@neuralmagic.com",
    license="Apache 2.0",
    description="Library for utilization of compressed safetensors of neural network models",
    long_description=_setup_long_description()[0],
    long_description_content_type=_setup_long_description()[1],
    url="https://github.com/neuralmagic/compressed-tensors",
    extras_require=_setup_extras(),
    install_requires=_setup_install_requires(),
    package_dir={"": "src"},
    packages=_setup_packages(),
)
