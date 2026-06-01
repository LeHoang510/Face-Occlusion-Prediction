import os
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


class OcclusionMaskDataset(Dataset):
    """Face occlusion dataset that returns RGB images with generated segmentation masks."""

    def __init__(
        self,
        csv_path: str,
        img_root: str,
        mask_root: str,
        transform=None,
        is_test: bool = False,
    ):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.mask_root = mask_root
        self.transform = transform
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        filename = row["filename"]
        image = Image.open(os.path.join(self.img_root, filename)).convert("RGB")
        mask = Image.open(self._mask_path(filename)).convert("L")

        if self.transform:
            image, mask = self.transform(image, mask)

        if self.is_test:
            return image, mask, filename

        label = torch.tensor(float(row["FaceOcclusion"]), dtype=torch.float32)
        gender = torch.tensor(float(row["gender"]), dtype=torch.float32)
        return image, mask, label, gender

    def _mask_path(self, filename: str) -> str:
        mask_path = Path(self.mask_root, filename).with_suffix(".png")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing segmentation mask: {mask_path}")
        return str(mask_path)


class SegmentationAuxTransform:
    def __init__(self, train: bool = True, img_size: int = 224):
        self.train = train
        self.img_size = img_size
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = F.resize(image, [self.img_size, self.img_size], interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, [self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST)

        if self.train and random.random() < 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        if self.train:
            image = self.color_jitter(image)

        image = F.normalize(F.to_tensor(image), mean=self.mean, std=self.std)
        mask = F.to_tensor(mask)
        return image, mask


def get_segmentation_aux_transforms(train: bool = True, img_size: int = 224):
    return SegmentationAuxTransform(train=train, img_size=img_size)
