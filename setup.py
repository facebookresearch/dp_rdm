#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages

setup(
    name='rdm',
    version='0.0.1',
    description='',
    packages=["rdm"],
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
