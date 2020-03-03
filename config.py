# -*- coding:utf-8 -*-
"""
   File Name：     config.py
   Description :   参数配置
   Author :        royce.mao
   date：          2020/3/2 10:43
"""


class Config(object):
    # 相关路径
    NUMS = ''.join(['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    N_CLASSES = 11  # 数字类别数 len(NUM)
    N_CHAR = 7  # 定长的数字串长度
    N_RNN_LEN = 17  # RNN序列的长度
    IMAGE_WIDTH = 140
    IMAGE_HEIGHT = 40

    LR = 0.001
    EPOCH = 100
    BATCH_SIZE = 16

    TRAIN_PATH = "./data/train"
    AUGED_TRAIN_PATH = "./data/auged_train"
    TEST_PATH = "./data/test"

    LOG_DIR = "./weights"
    CKPT_PATH = "./weights/crnn_ctc.pth"
    OUT_DIR = "./out"


cur_config = Config()
