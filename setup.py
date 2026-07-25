#   Copyright (c) 2022 DeepEvolution Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io

from robofm import __version__
from setuptools import setup, find_packages

with io.open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='robofm',
    version=__version__,  
    packages=find_packages(include=['robofm', 'robofm.*']),
    package_dir={'': '.'},  
    install_requires=[
        'numpy>=1.18.0',
        'torch>=2.4.0',
    ],
    extras_require={
        'fla': ['flash-linear-attention>=0.1.0'],
        'logging': ['tensorboard>=2.14.0'],
        'data': [
            'gymnasium>=1.0.0',
            'stable-baselines3>=2.0.0',
            'sb3-contrib>=2.0.0',
        ],
    },
)
