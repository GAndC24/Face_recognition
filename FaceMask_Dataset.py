import torch
from torch.utils.data import Dataset
import os
from PIL import Image

class FaceMask_Dataset_pretrain(Dataset):
    def __init__(self, img_dir, transform=None):
        '''
        初始化数据集
        Args:
            :param img_dir: 图像文件夹路径
            :param transform: 数据预处理
        '''
        # 获取当前工作目录
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建相对路径
        img_dir = os.path.join(current_file_dir, img_dir)

        self.img_dir = img_dir
        self.transform = transform
        self.image_paths = [os.path.join(img_dir, img) for img in os.listdir(img_dir)
                            if img.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        '''
        获取数据集中的数据
        Args:
            :param idx: 数据索引
        Returns:
            :return: 图像数据和标签
        '''
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        label = img     # 图像本身作为标签

        if self.transform:
            img = self.transform(img)
            label = self.transform(label)

        return img, label


class FaceMask_Dataset(Dataset):
    def __init__(self, img_dir, transform=None):
        '''
        初始化数据集
        :param img_dir: 图像文件夹路径
        :param transform: 数据预处理
        '''
        self.img_dir = img_dir
        self.transform = transform
        self.img_list = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        '''
        获取数据集中的数据
        :param idx:  数据索引
        :return:  图像数据和标签
        '''
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        img_name = self.img_list[idx]

        img_name = img_name.split('.')[0]   # 去掉图像文件名的后缀
        img_name_decimal = int(img_name)       # 图像文件名的十进制数
        # 转为十七位二进制表示作为标签
        label = torch.zeros(17)
        for i in range(17):
            label[i] = img_name_decimal % 2
            img_name_decimal = img_name_decimal // 2

        return img, label

class Temp_dataset(Dataset):
    '''
    临时数据集，用于获取 generator 的 FID 分数
    '''
    def __init__(self, images):
        '''
        初始化数据集

        Args:
            :param images: 图像数据, [N, C, H, W]
        '''
        self.images = images

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        '''
        获取数据集中的数据
        Args:
            :param idx:  数据索引
            :return:  图像数据
        '''
        return self.images[idx]
