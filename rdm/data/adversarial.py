#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
import numpy as np
from PIL import Image
import albumentations

def preprocessor(size):
    rescaler = albumentations.SmallestMaxSize(max_size = size)
    cropper = albumentations.CenterCrop(height=size,width=size)
    return albumentations.Compose([rescaler, cropper])

class AdversarialDataset(Dataset):
    def __init__(self,size, root_dir, ignore_sample=[], **kwargs):
        self.size = size

        self.preprocessor = preprocessor(self.size) 
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)
        self.images = [img for img in self.images if img not in ignore_sample]


    def preprocess(self, image):
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)


        return {'image': self.preprocess(image)}

class AdversarialDatasetMinusOne(AdversarialDataset):
    def __init__(self,size, root_dir, **kwargs):
        # 009 -- pumpkins
        # 004 -- test tubes
        super().__init__(size=size, root_dir=root_dir, ignore_sample=['004.png'], **kwargs)

class White(AdversarialDataset):
    def __init__(self,size, root_dir, **kwargs):
        super().__init__(size=size, root_dir=root_dir, **kwargs)

class WhiteMinusOne(White):
    def __init__(self,size, root_dir, **kwargs):
        super().__init__(size=size, root_dir=root_dir, ignore_sample=['001.png'], **kwargs)
