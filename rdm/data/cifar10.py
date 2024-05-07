import torchvision
import numpy as np
from PIL import Image
import albumentations
import cv2

class Cifar10(torchvision.datasets.CIFAR10):
    def __init__(self,size, interpolation=cv2.INTER_CUBIC, *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.size = size
        self.interpolation = interpolation
        self.preprocessor = albumentations.Resize(width=32, height=32, interpolation=self.interpolation)

    def preprocess(self, image):
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image


    def __getitem__(self, index):
        item = super().__getitem__(index)
        return {'image': self.preprocess(item[0]), 'target': item[1]}

class Cifar10Train(Cifar10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, train=True)

class Cifar10Validation(Cifar10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, train=False)

    def __getitem__(self, index):
        # cycle through the validation set...
        return super().__getitem__(index % 10_000)