import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class HDRDataset(Dataset):
    """
    Custom HDR dataset that returns a dictionary of LDR input image, HDR ground truth image and file path. 
    """

    def __init__(self, mode, opt):

        self.batch_size = opt.batch_size

        if mode == "train":
            self.dataset_path = os.path.join("./dataset", "train")
        else:
            self.dataset_path = os.path.join("./dataset", "test")

        self.ldr_data_path = os.path.join(self.dataset_path, "LDR")
        self.hdr_data_path = os.path.join(self.dataset_path, "HDR")

        # paths to LDR and HDR images ->

        self.ldr_image_names = sorted(os.listdir(self.ldr_data_path))
        self.hdr_image_names = sorted(os.listdir(self.hdr_data_path))

    def __getitem__(self, index):

        self.ldr_image_path = os.path.join(
            self.ldr_data_path, self.ldr_image_names[index]
        )

        # transformations on LDR input ->

        ldr_sample = Image.open(self.ldr_image_path).convert("RGB")
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        transform_ldr = transforms.Compose(transform_list)
        ldr_tensor = transform_ldr(ldr_sample)

        # transformations on HDR ground truth ->

        self.hdr_image_path = os.path.join(
            self.hdr_data_path, self.hdr_image_names[index]
        )

        hdr_sample = cv2.imread(self.hdr_image_path, -1).astype(np.float32)

        # transforms.ToTensor() is used for 8-bit [0, 255] range images; can't be used for [0, ∞) HDR images
        transform_list = [
            transforms.Lambda(lambda img: torch.from_numpy(img.transpose((2, 0, 1)))),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        transform_hdr = transforms.Compose(transform_list)
        hdr_tensor = transform_hdr(hdr_sample)

        sample_dict = {
            "ldr_image": ldr_tensor,
            "hdr_image": hdr_tensor,
            "path": self.ldr_image_path,
        }

        return sample_dict

    def __len__(self):
        return len(self.ldr_image_names) // self.batch_size * self.batch_size
