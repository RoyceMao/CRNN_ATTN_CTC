# -*- coding:utf-8 -*-
"""
   File Name：     dataset.py
   Description :   数据加载
   Author :        royce.mao
   date：          2020/3/2 14:46
"""

import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import torch as t
import re

nums = ''.join(['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])


# lower_char = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
#               'v', 'w', 'x', 'y', 'z']
# upper_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
#               'V', 'W', 'X', 'Y', 'Z']

def str_to_label(str_, num_class, num_char, one_hot=True):
    """

    :param str_:
    :param num_class:
    :param num_char:
    :param one_hot:
    :return:
    """
    global nums
    target_str = str_.split('_')[0]
    # print(target_str)
    assert len(target_str) == num_char

    if one_hot:
        label = []
        for char in target_str:
            vec = [0] * num_class
            vec[nums.find(char)] = 1
            label += vec
        return t.tensor(label).float()

    else:
        label = [nums.find(char) for char in target_str]
        return t.tensor(label).long()


def label_to_str(label):
    """

    :param label:
    :return:
    """
    str_ = ""
    for i in label:
        if i <= 9:
            str_ += chr(ord('0') + i)
        elif i <= 35:
            str_ += chr(ord('a') + i - 10)
        else:
            str_ += chr(ord('A') + i - 36)
    return str_


class AugDigitSet(data.Dataset):
    """ 已增广训练数据加载类 """
    def __init__(self, root, cfg, one_hot=True, train=True):
        self.num_class = cfg.N_CLASSES
        self.num_char = cfg.N_CHAR
        self.one_hot = one_hot
        self.imgs_path = [os.path.join(root, img) for img in os.listdir(root)]

        self.num_input = cfg.N_RNN_LEN

        self.transform = T.Compose([
            T.ColorJitter(brightness=0.5, contrast=0.8, saturation=0.8, hue=0.5),
            T.Resize((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)),
            # p.torch_transform(),
            T.ToTensor(),
            # T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        img_name = os.path.basename(img_path)
        pattern = re.compile(r'\w+_original_(\d*\w*)')
        label = pattern.search(img_name).groups()[0]

        target = str_to_label(label, self.num_class, self.num_char, self.one_hot)

        img = Image.open(img_path)
        img = img.convert('L')
        image = self.transform(img)

        input_length = t.full(size=(1,), fill_value=self.num_input, dtype=t.long)
        target_length = t.full(size=(1,), fill_value=self.num_char, dtype=t.long)

        if self.one_hot:
            return image, target
        else:
            return image, target, input_length, target_length

    def __len__(self):
        return len(self.imgs_path)


class DigitSet(data.Dataset):
    """ 原数据加载类 """
    def __init__(self, root, cfg, one_hot=True, train=True):
        self.num_class = cfg.N_CLASSES
        self.num_char = cfg.N_CHAR
        self.one_hot = one_hot
        self.imgs_path = [os.path.join(root, img) for img in os.listdir(root)]

        self.num_input = 17

        self.transform = T.Compose([
            T.ColorJitter(brightness=0.5, contrast=0.8, saturation=0.8, hue=0.5),
            T.Resize((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)),
            # p.torch_transform(),
            T.ToTensor(),
            # T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        label = os.path.basename(img_path).split(".")[0]

        target = str_to_label(label, self.num_class, self.num_char, self.one_hot)

        img = Image.open(img_path)
        img = img.convert('L')
        image = self.transform(img)

        input_length = t.full(size=(1,), fill_value=self.num_input, dtype=t.long)
        target_length = t.full(size=(1,), fill_value=self.num_char, dtype=t.long)

        if self.one_hot:
            return image, target
        else:
            return image, target, input_length, target_length

    def __len__(self):
        return len(self.imgs_path)
