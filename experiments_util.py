#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path
import albumentations
import glob
import pickle
import pandas as pd
import numpy as np
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
from rdm.util import COLORS 
import seaborn as sns
from functools import partial
from matplotlib import pyplot as plt
from scipy import optimize
import cv2

DEFAULT_PATH = "models/rdm/imagenet/model.ckpt"

# Retrieval Dataset Configuration
RETRIEVAL_DB_MSCOCO_FACEBLURRED_METACLIP = (
    "models/rdm/mscoco/config2014_metaclip_faceblurred.yaml"
)
RETRIEVAL_DB_MSCOCO_METACLIP = "models/rdm/mscoco/config2014_metaclip.yaml"
RETRIEVAL_DB_MSCOCO_CLIP = "models/rdm/mscoco/config2014_clip.yaml"
RETRIEVAL_DB_CIFAR10 = "models/rdm/cifar10/config_metaclip.yaml"
RETRIEVAL_DB_CIFAR10_CLIP = "models/rdm/cifar10/config_clip.yaml"

RETRIEVAL_DB_IMAGENET_METACLIP = "models/rdm/imagenet/faceblurred.yaml"

RETRIEVAL_DB_SHUTTERSTOCK = "models/rdm/shutterstock/full.yaml"
RETRIEVAL_DB_SHUTTERSTOCK_SMALL = "models/rdm/shutterstock/small.yaml"

# delta parameter used for computing the privacy loss wrt the dataset
N_COCO = 82_800
N_SHUTTERSTOCK = 239_000_000
N_CIFAR10 = 50_000
N_IMAGENET = 1_200_000

RETRIEVAL_DBNAME_CIFAR10 = "cifar10"
RETRIEVAL_DBNAME_MSCOCO = "mscoco"
RETRIEVAL_DBNAME_MSCOCO_FACEBLURRED = "mscoco_faceblurred"
RETRIEVAL_DBNAME_SHUTTERSTOCK_SMALL = "shutterstock_sm"
RETRIEVAL_DBNAME_SHUTTERSTOCK = "shutterstock"
RETRIEVAL_DBNAME_IMAGENET_FACEBLURRED = "imagenet_faceblurred"

## Image processing

cifar10_processor = albumentations.Resize(32, 32, interpolation=cv2.INTER_LANCZOS4)
mscoco_processor = albumentations.Compose(
    [
        albumentations.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_LANCZOS4),
        albumentations.CenterCrop(height=256, width=256),
    ]
)
DEMO_PROMPTS = [
  'A photo of a modern car',
  'A beautiful photo of a llama',
  'A goose',
  'A laptop',
  'A photo of a smart watch, high quality',
  'A ship',
] * 80

CIFAR10_PROMPTS = [
    "an airplane, high quality photograph",
    "an automobile, high quality photograph",
    "a bird, high quality photograph",
    "a cat, high quality photograph",
    "a deer, high quality photograph",
    "a dog, high quality photograph",
    "a frog, high quality photograph",
    "a horse, high quality photograph",
    "a ship, high quality photograph",
    "a truck, high quality photograph",
]

MSCOCO_VALIDATION_BASEPATH = "/datasets01/COCO/060817/val2014"
MSCOCO_VALIDATION_FACEBLURRED_BASEPATH = (
    "/checkpoint/lebensold/COCO/val2014_faceblurred"
)


MSCOCO_VALIDATION_CAPTIONS = "/datasets01/COCO/060817/annotations/captions_val2014.json"
json_captions = json.load(open(MSCOCO_VALIDATION_CAPTIONS, "rb"))
MSCOCO_PROMPTS = list(
    dict(
        [(int(rec["image_id"]), rec["caption"]) for rec in json_captions["annotations"]]
    ).items()
)

SHUTTERSTOCK_VALIDATION_PATH = '/checkpoint/lebensold/shutterstock-validation'
shutterstock_captions = json.load(open(f'{SHUTTERSTOCK_VALIDATION_PATH}/captions.json', 'rb'))

SHUTTERSTOCK_PROMPTS = [(int(key), caption) for key, caption in shutterstock_captions.items()]


cfg_map = {
    RETRIEVAL_DB_CIFAR10: "CIFAR10",
    RETRIEVAL_DB_CIFAR10_CLIP: "CIFAR10 CLIP",
    RETRIEVAL_DB_MSCOCO_CLIP: "COCO",
    RETRIEVAL_DB_MSCOCO_METACLIP: "COCO",
    RETRIEVAL_DB_MSCOCO_FACEBLURRED_METACLIP: "MS-COCO FB",
    RETRIEVAL_DB_SHUTTERSTOCK_SMALL: "Shutterstock 1M",
    RETRIEVAL_DB_SHUTTERSTOCK: "Shutterstock 239M",
    RETRIEVAL_DB_IMAGENET_METACLIP: "ImageNet FB"
}

# Privacy Analysis
def calc_eps(sigma, subsample_rate, knn, n_queries, delta):
    """
    Calculate the differential privacy budget (epsilon) for a mechanism using the RDP (Rényi Differential Privacy) accountant method.

    This function estimates the privacy budget (epsilon) and the optimal order of Rényi divergence (alpha) for a query under the Gaussian mechanism, based on provided parameters like noise level, subsampling rate, and number of queries.

    Parameters:
        sigma (float): The standard deviation of the Gaussian noise to be added for privacy.
        subsample_rate (float): The proportion of the dataset that each query accesses (subsample rate).
        knn (int): The parameter 'k' used to determine the granularity of the Gaussian noise; effectively used to calculate the global sensitivity.
        n_queries (int): The total number of queries over which the privacy loss is accumulated.
        delta (float): The target delta value of differential privacy.

    Returns:
        tuple: A tuple containing two elements:
            - eps (float): The calculated epsilon value, representing the privacy budget.
            - opt_alpha (float): The optimal order of Rényi divergence at which the minimum epsilon is achieved.

    Examples:
        # Calculate epsilon for a given configuration.
        eps, alpha = calc_eps(sigma=1.0, subsample_rate=0.01, knn=5, n_queries=1000, delta=1e-5)

    """
    GS = 2 / knn
    noise_multiplier = sigma / GS

    alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))
    rdp = compute_rdp(
        q=subsample_rate,
        noise_multiplier=noise_multiplier,
        steps=n_queries,
        orders=alphas,
    )
    eps, opt_alpha = get_privacy_spent(orders=alphas, rdp=rdp, delta=delta)
    return eps, opt_alpha

def binary_search(f, x_min, x_max, target, prec=1e-4):
    """
    Precondition:
    f is an increasing function in x with f(x_min) <= target and f(x_max) >= target
    """
    x_mid = (x_min + x_max) / 2
    y_mid = f(x_mid)
    if abs(y_mid - target) < prec:
        return x_mid
    elif y_mid > target:
        return binary_search(f, x_min, x_mid, target, prec)
    else:
        return binary_search(f, x_mid, x_max, target, prec)
    
def find_k(epsilon, q, sigma, n_queries, N):
    f = lambda k: -calc_eps(sigma, q, k, n_queries, delta=1/N)[0]
    return binary_search(f, 1, N, -epsilon)

def find_q(epsilon, k, sigma, n_queries, N):
    f = lambda q: calc_eps(sigma, q, k, n_queries, delta=1/N)[0]
    return binary_search(f, 0, 1, epsilon)


# Experiment Mapping
def dataset_from_config(path):
    if "cifar10" in path:
        return RETRIEVAL_DBNAME_CIFAR10
    if "mscoco_faceblurred" in path:
        return RETRIEVAL_DBNAME_MSCOCO_FACEBLURRED
    if "mscoco" in path:
        return RETRIEVAL_DBNAME_MSCOCO
    if "shutterstock" in path:
        return RETRIEVAL_DBNAME_SHUTTERSTOCK
    if "imagenet" in path:
        return RETRIEVAL_DBNAME_IMAGENET_FACEBLURRED
    
    assert "undefined", path


def dp_delta_from_config(path):
    return 1 / ds_size_from_config(path)
    
def ds_size_from_config(path):
    if "mscoco" in path:
        return N_COCO
    if "imagenet" in path:
        return N_IMAGENET
    if "shutterstock" in path:
        return N_SHUTTERSTOCK
    if "cifar10" in path:
        return N_CIFAR10
    return 0