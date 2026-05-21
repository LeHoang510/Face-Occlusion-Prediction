import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class OcclusionDataset(Dataset):
    """Face occlusion regression dataset."""

    def __init__(self, csv_path: str, img_root: str, transform=None, is_test: bool = False):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row["filename"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, row["filename"]

        label = torch.tensor(float(row["FaceOcclusion"]), dtype=torch.float32)
        gender = torch.tensor(float(row["gender"]), dtype=torch.float32)
        return image, label, gender


def get_transforms(train: bool = True, img_size: int = 224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
