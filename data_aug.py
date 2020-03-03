# -*- coding:utf-8 -*-
"""
   File Name：     data_aug.py
   Description :   数据准备
   Author :        royce.mao
   date：          2020/3/2 12:01
"""

import cv2
import Augmentor
import numpy as np

from lib.dataset import *
from config import cur_config as cfg


def get_aug(path, num):
    """
    最终采用 增广策略：1）比例缩放、2）随机扭曲、3）随机透视、4）随机遮挡
    :param path:
    :param num:
    :return:
    """
    p = Augmentor.Pipeline(path)
    # 比例缩放
    p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    # 随机扭曲
    p.random_distortion(probability=1, grid_width=6, grid_height=2, magnitude=3)
    # 随机透视（视角倾斜）
    p.skew_tilt(probability=0.7, magnitude=0.1)
    # 随机遮挡
    p.random_erasing(probability=0.7, rectangle_area=0.2)

    p.sample(num)

    return p


def get_distortion_pipline(path, num):
    """
    之前采用 增广策略：随机扭曲
    :param path:
    :param num:
    :return:
    """
    p = Augmentor.Pipeline(path)
    p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    p.random_distortion(probability=1, grid_width=6, grid_height=2, magnitude=3)
    p.sample(num)
    return p


class Enhance(object):
    """ 图像增强 """
    def __init__(self, ):
        self.scale = 100
        self.num_scales = 1
        self.dynamic = 2
        self.bin_thresh = 180
        super(Enhance, self).__init__()

    @staticmethod
    def RSD(max_scale, nscales):
        """ 生成长度为nscales的高斯滤波的尺度集 """
        scales = []
        scale_step = max_scale / nscales
        for s in range(nscales):
            scales.append(scale_step * s + 2.0)

        return scales

    @staticmethod
    def CR(im_ori, im_log, alpha=128., gain=1., offset=0.):
        """ 色彩恢复 """
        im_cr = im_log * gain * (
                    np.log(alpha * (im_ori + 1.0)) - np.log(np.sum(im_ori, axis=2) + 3.0)[:, :, np.newaxis]) + offset
        return im_cr

    @staticmethod
    def NOR(im_cr, dynamic):
        """ 无色偏量化调节 """
        im_rtx_mean = np.mean(im_cr)
        im_rtx_std = np.std(im_cr)

        im_rtx_min = im_rtx_mean - dynamic * im_rtx_std
        im_rtx_max = im_rtx_mean + dynamic * im_rtx_std

        im_rtx_range = im_rtx_max - im_rtx_min

        im_out = np.uint8(np.clip((im_cr - im_rtx_min) / im_rtx_range * 255.0, 0, 255))

        return im_out

    @staticmethod
    def OPEN(img):
        """
        图像开运算，去除噪点类干扰以及补全验证码缺口
        :param img:
        :return:
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        new_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        new_img = cv2.dilate(new_img, kernel2)
        return new_img

    def enhanced(self, img):
        scales = self.RSD(self.scale, self.num_scales)
        im_blur = np.zeros([len(scales), img.shape[0], img.shape[1], img.shape[2]])
        im_mlog = np.zeros([len(scales), img.shape[0], img.shape[1], img.shape[2]])
        # MSR增强
        for channel in range(3):
            for s, scale in enumerate(scales):
                # 先对接收到的图像，做高斯滤波（3通道，每个通道采用3种不同的尺度）
                im_blur[s, :, :, channel] = cv2.GaussianBlur(img[:, :, channel], (0, 0), scale)
                # 然后取图像的对数减去图像高斯滤波之后的对数
                im_mlog[s, :, :, channel] = np.log((img[:, :, channel] + 1.) / (im_blur[s, :, :, channel] + 1.))

        # 各通道均按1/3的权重，加权求和
        img_msr = np.mean(im_mlog, 0)

        # 色彩恢复
        im_cr = self.CR(img, img_msr)

        # 无色偏量化调节
        img_enhanced = self.NOR(im_cr, self.dynamic)

        # 灰度二值化
        img_gray = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2GRAY)
        ret, img_bin = cv2.threshold(img_gray, self.bin_thresh, 255, cv2.THRESH_BINARY)

        # 开运算
        img_out = self.OPEN(img_bin)

        return img_out


if __name__ == '__main__':
    """
    usage: python data_aug.py
    """
    times = 2  # 训练集扩充2倍
    num_ = len(os.listdir(cfg.TRAIN_PATH)) * times

    pro = get_aug(cfg.TRAIN_PATH, num_)  # get_distortion_pipline(cfg.TRAIN_PATH, num)

    pro.process()
