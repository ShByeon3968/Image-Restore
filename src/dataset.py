import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
from skimage import color

from colorization.colorizers import util

class CustomImageDataset(Dataset):
    def __init__(self, input_dir, input_unmasked_dir,gt_dir, mask_dir, transform=None, mode='train', split_ratio=0.8):
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
        self.input_unmasked_dir = input_unmasked_dir
        self.gt_dir = gt_dir
        self.mask_dir = mask_dir

        self.input_unmask = sorted(os.listdir(self.input_unmasked_dir))
        self.input_files = sorted(os.listdir(input_dir))
        self.gt_files = sorted(os.listdir(gt_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.transform = transform

        # Split dataset into train and validation
        total_samples = len(self.input_files)
        train_count = int(total_samples * split_ratio)

        if mode == 'train':
            self.input_files = self.input_files[:train_count]
            self.input_unmask = self.input_unmask[:train_count]
            self.gt_files = self.gt_files[:train_count]
            self.mask_files = self.mask_files[:train_count]
        elif mode == 'val':
            self.input_unmask = self.input_unmask[train_count:]
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
        input_unmask_path = os.path.join(self.input_unmasked_dir, self.input_unmask[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        input_image = Image.open(input_path).convert("RGB")  # Grayscale, 3채널
        input_unmask_image = Image.open(input_unmask_path).convert("L")  # Grayscale
        gt_RGB_image = Image.open(gt_path).convert("RGB")  # Ground Truth (Color)
        gt_GRAY_image = Image.open(gt_path).convert("L")  # Ground Truth (GRAY)
        gt_GRAY_image = gt_GRAY_image.convert("RGB")
        mask_image = Image.open(mask_path).convert("L")  # Binary mask

        if self.transform:
            input_image = self.transform(input_image)
            input_unmask_image = self.transform(input_unmask_image)
            gt_RGB_image = self.transform(gt_RGB_image)
            gt_GRAY_image = self.transform(gt_GRAY_image)
            mask_image = self.transform(mask_image)

        return input_image, input_unmask_image,gt_RGB_image,gt_GRAY_image,mask_image


class EdgeImageDataset(Dataset):
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

        # Load images using cv2
        input_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)  # Grayscale
        gt_image = cv2.imread(gt_path, cv2.IMREAD_COLOR)  # Ground Truth (Color)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Binary mask

        if self.transform:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
            input_image = self.transform(Image.fromarray(input_image.astype(np.uint8)))
            gt_image = self.transform(Image.fromarray(gt_image.astype(np.uint8)))
            mask_image = self.transform(Image.fromarray(mask_image.astype(np.uint8)))

        return input_image, gt_image, mask_image

class TestImageDataset(CustomImageDataset):
    def __init__(self, test_dir, transform):
        self.image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # 이미지를 cv2로 로드
        image = cv2.imread(image_path)
        image = Image.fromarray(image)
        # Transform 적용
        image = self.transform(image)
        return image, os.path.basename(image_path)
    
class ColorizationImageDataset(Dataset):
    def __init__(self, input_dir, transform=None):
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

        self.input_files = sorted(os.listdir(input_dir))
        self.transform = transform
    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Load images
        input_path = os.path.join(self.input_dir, self.input_files[idx])

        image = util.load_img(input_path)
        (tens_l_orig, tens_l_rs) = util.preprocess_img(image, HW=(256,256))

        return (tens_l_orig, tens_l_rs)
    

