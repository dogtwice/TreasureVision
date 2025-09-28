#定义公开接口, 只有all中的东西才会被暴露
__all__=['FolderDataset']

from PIL import Image  # 导入PIL库中的Image模块，用于图像处理
import os  # 导入os库，用于处理文件和目录路径
from torch.utils.data import Dataset  # 从PyTorch的工具库中导入Dataset类，用于自定义数据集
import denoising_config
import torch
import numpy as np


import re #正则表达式相关库


def sorted_alphanumeric(data):
    """对文件名进行自然排序.
    参数:
        data (list): 文件名列表.
    返回:
        list: 排序后的文件名列表.
    """
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(data, key=alphanum_key)

class FolderDataset(Dataset):
    def __init__(self, main_dir, transform=None):
        super().__init__()
        self.main_dir = main_dir
        self.transform = transform
        self.image_files = sorted_alphanumeric(os.listdir(main_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.main_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            original_img = self.transform(image)
        else:
            raise ValueError("Transform is not defined.")
        
        noise_img= original_img + denoising_config.NOISE_FACTOR * torch.randn(*original_img.shape)
        noise_img = torch.clip(noise_img, 0., 1.)
        return noise_img, original_img