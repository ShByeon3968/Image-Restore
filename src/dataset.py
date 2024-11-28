import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CustomImageDataset(Dataset):
    def __init__(self, input_dir, gt_dir, mask_dir, transform=None, mode='train', split_ratio=0.7):
        """
        Args:
            input_dir (str): Path to masked input images directory.
            gt_dir (str): Path to ground truth images directory.
            mask_dir (str): Path to mask binary images directory.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (str): Either 'train' or 'val'.
            split_ratio (float): Ratio of data to be used for training. Default is 0.7.
        """
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.mask_dir = mask_dir
        self.input_files = sorted(os.listdir(input_dir))
        self.gt_files = sorted(os.listdir(gt_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.transform = transform

        # Split dataset into train and validation
        total_samples = len(self.input_files)
        train_count = int(total_samples * split_ratio)

        if mode == 'train':
            self.input_files = self.input_files[:train_count]
            self.gt_files = self.gt_files[:train_count]
            self.mask_files = self.mask_files[:train_count]
        elif mode == 'val':
            self.input_files = self.input_files[train_count:]
            self.gt_files = self.gt_files[train_count:]
            self.mask_files = self.mask_files[train_count:]
        else:
            raise ValueError("Mode must be 'train' or 'val'")

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Load images
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        input_image = Image.open(input_path).convert("L")  # Grayscale
        gt_image = Image.open(gt_path).convert("RGB")  # Ground Truth (Color)
        mask_image = Image.open(mask_path).convert("L")  # Binary mask

        if self.transform:
            input_image = self.transform(input_image)
            gt_image = self.transform(gt_image)
            mask_image = self.transform(mask_image)

        return input_image, gt_image, mask_image
