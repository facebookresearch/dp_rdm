#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torchvision
import numpy as np
from PIL import Image
import albumentations
import cv2


class MSCOCO(torchvision.datasets.CocoDetection):
    def __init__(self,size, interpolation=cv2.INTER_LANCZOS4, *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.size = size
        self.interpolation = interpolation

        self.rescaler = albumentations.SmallestMaxSize(max_size = self.size, interpolation=self.interpolation)
        self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def preprocess(self, image):
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image


    def __getitem__(self, index):
        item = super().__getitem__(index)
        return {'image': self.preprocess(item[0])}
