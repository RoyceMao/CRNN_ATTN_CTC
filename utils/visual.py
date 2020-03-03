# -*- coding:utf-8 -*-
"""
   File Name：     visual.py
   Description :   可视化
   Author :        royce.mao
   date：          2020/3/2 16:20
"""
import cv2
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# CTPN定位之后的数字串小图
class DigitPic(object):
    def __init__(self, ):
        super(DigitPic, self).__init__()

    @staticmethod
    def padding(img, pad_size):
        """
        顶部加padding（黑色背景，便于put_txt）
        :param img: numpy
        :param pad_size: size标量
        :return:
        """
        # 增加padding
        padding_size = int(max(0, pad_size))
        img_padding = np.pad(img, ((padding_size, 0), (0, 0)), mode='constant', constant_values=0)

        return img_padding

    def to_txt_image(self, img, pred):
        """
        put_txt英文
        :param img: 加了padding的img
        :param pred: 预测的字符串
        :return:
        """
        left_up = (0, 0)
        text_size, baseline = cv2.getTextSize(
            pred, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)

        p1 = (left_up[0], left_up[1] + text_size[1])
        img_padding = self.padding(img, p1[1] + baseline)

        cv2.putText(img_padding, 'pred:' + pred, (p1[0], p1[
            1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, 8)

        return img_padding


# CTPN定位数字串之前的原图
class OriginPic(object):
    def __init__(self, ):
        super(OriginPic, self).__init__()

    @staticmethod
    def json_parse(annotation_condidate):
        """
        labeme标注的json解析
        :param annotation_condidate:
        :return:
        """
        with open(annotation_condidate, 'r') as fp:
            data = json.load(fp)  # 加载json文件
            boxes = []
            for shape in data['shapes']:
                x1, y1 = shape['points'][0]  # 第一个点是左上角
                x2, y2 = shape['points'][1]  # 第二个点是右下角
                boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
            bbox = [int(np.min(np.array(boxes)[:, 0])),
                    int(np.min(np.array(boxes)[:, 1])),
                    int(np.max(np.array(boxes)[:, 2])),
                    int(np.max(np.array(boxes)[:, 3])),
                    ]

        return bbox

    @staticmethod
    def add_cn_text(img, text, left_top, color=(0, 0, 255), size=50):
        """
        中文支持
        :param img:
        :param text:
        :param left_top:
        :param color:
        :param size:
        :return:
        """
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('./simhei.ttf', size, encoding="utf-8")
        draw.text(left_top, text, color, font=font)

        return np.asarray(img)

    def to_bbox_txt_image(self, img, json_path, pred, score):
        """
        put_bbox and put_txt中文
        :param img:
        :param json_path:
        :param pred:
        :param score:
        :return:
        """
        pt = self.json_parse(json_path)

        # 画框
        left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
        cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 5)

        # 添加中文文字
        conf = "{:.2f}".format(score)

        text_size, baseline = cv2.getTextSize(
            'Pred:' + pred + '丨' + 'Score:' + conf, cv2.FONT_HERSHEY_SIMPLEX, 5, 1)

        p1 = (left_up[0], left_up[1] - text_size[1])

        return self.add_cn_text(img, '预测:' + pred + ' ' + '得分:' + conf, (p1[0], p1[1]), color=(0, 0, 255), size=100)
