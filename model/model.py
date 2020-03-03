# -*- coding:utf-8 -*-
"""
   File Name：     model.py
   Description :   目前测试了2种CRNN的基网络，1）VGG+BiLSTM、2）ResNet+BiLSTM+Attention
   Author :        royce.mao
   date：          2020/3/2 14:27
"""
import torch
import torch.nn as nn
from collections import OrderedDict

import model.ResNet as res


class VGG_LSTM(nn.Module):
    """ 普通CRNN """
    def __init__(self, n_classes, input_shape=(1, 40, 140)):  # 灰度图
        super(VGG_LSTM, self).__init__()
        self.input_shape = input_shape
        channels = [32, 64, 128, 256]
        layers = [2, 2, 2, 2]
        kernels = [3, 3, 3, 3]
        pools = [2, 2, 2, (2, 1)]
        modules = OrderedDict()

        def cba(name, in_channels, out_channels, kernel_size):
            modules[f'conv{name}'] = nn.Conv2d(in_channels, out_channels, kernel_size,
                                               padding=(1, 1) if kernel_size == 3 else 0)
            modules[f'bn{name}'] = nn.BatchNorm2d(out_channels)
            modules[f'relu{name}'] = nn.ReLU(inplace=True)

        last_channel = 1
        for block, (n_channel, n_layer, n_kernel, k_pool) in enumerate(zip(channels, layers, kernels, pools)):
            for layer in range(1, n_layer + 1):
                cba(f'{block + 1}{layer}', last_channel, n_channel, n_kernel)
                last_channel = n_channel
            modules[f'pool{block + 1}'] = nn.MaxPool2d(k_pool)
        modules[f'dropout'] = nn.Dropout(0.25, inplace=True)

        self.cnn = nn.Sequential(modules)
        self.lstm = nn.LSTM(input_size=self.infer_features(), hidden_size=128, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(in_features=256, out_features=n_classes)  # 输出的标签长度为n_input_length，不等于n_char，需要根据置信度解码

    def infer_features(self):
        x = torch.zeros((1,) + self.input_shape)
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x.shape[1]

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


class ResNet_LSTM(nn.Module):
    """ 带残差、Attention的CRNN """
    def __init__(self, n_classes, input_shape=(1, 40, 140)):  # 灰度图
        super(ResNet_LSTM, self).__init__()
        self.input_shape = input_shape
        self.resnet_cbam = res.resnet18_cbam(pretrained=False)
        self.dropout = nn.Dropout(0.25, inplace=True)
        self.lstm = nn.LSTM(input_size=self.infer_features(), hidden_size=128, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(in_features=256, out_features=n_classes)

    def infer_features(self):
        x = torch.zeros((1,) + self.input_shape)
        x = self.resnet_cbam(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x.shape[1]

    def forward(self, x):
        x = self.resnet_cbam(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
