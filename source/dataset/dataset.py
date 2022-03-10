from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """自定义数据集"""

    # Todo image is gray file and need to tranfrom
    def __init__(self, images_list: list, images_class: list, transform=None):
        self.images_list = images_list
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, item):
        img = self.images_list[item]
        img = torch.unsqueeze(img, dim=0)
        label = self.images_class[item]
        label = torch.unsqueeze(label, dim=0)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
