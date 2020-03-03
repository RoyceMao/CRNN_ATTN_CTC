# -*- coding:utf-8 -*-
"""
   File Name：     train.py
   Description :   crnn+ctc数字序列学习 - 训练脚本
   Author :        royce.mao
   date：          2020/3/2 10:08
"""

from config import cur_config as cfg
from model.model import *
from lib.dataset import *
from lib.trainer import *

import torch
from torch.utils.data import DataLoader

import argparse
from tensorboardX import SummaryWriter


def main(args):
    #
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    if args.display:
        writer = SummaryWriter(log_dir=cfg.LOG_DIR, comment='train')
    else:
        writer = None

    # Dataloader
    torch.manual_seed(42)

    augtrain_datasets = AugDigitSet(cfg.AUGED_TRAIN_PATH, cfg, one_hot=False, train=True)
    # train_datasets = DigitSet(cfg.TRAIN_PATH, cfg, one_hot=False, train=False)
    test_datasets = DigitSet(cfg.TEST_PATH, cfg, one_hot=False, train=False)

    augtrain_dataloader = DataLoader(augtrain_datasets,
                                     batch_size=cfg.BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=4)
    '''
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=1)
    '''
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1)

    # Model
    if args.model == 'resnet':
        model = ResNet_LSTM(cfg.N_CLASSES, input_shape=(1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH))
        model = model.to(device)
    elif args.model == 'vgg':
        model = VGG_LSTM(cfg.N_CLASSES, input_shape=(1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH))
        model = model.to(device)
    else:
        raise ModuleNotFoundError

    # Train and Valid
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, amsgrad=True)

    acc_eval_best = 0  # 初始化

    for epoch in range(1, args.epoch + 1):
        # 训练
        train(model, optimizer, epoch, augtrain_dataloader, device, cfg, writer)
        # 评估测试
        acc = valid(model, epoch, test_dataloader, device, cfg, writer)
        # 保存最佳eval_acc的模型
        if acc > acc_eval_best:
            acc_eval_best = acc
            torch.save(model, cfg.CKPT_PATH)


if __name__ == '__main__':
    """
    usage: python train.py --[params]
    """
    # 可调参数
    parser = argparse.ArgumentParser(description='PyTorch CRNN+CTC Training')

    parser.add_argument('--gpu', default='1', type=str,
                        help='GPU ID Select')
    parser.add_argument('--epoch', default=100, type=int,
                        help='training epochs')
    parser.add_argument('--model', default='resnet',
                        type=str, help='model type')
    parser.add_argument('--display', action='store_true', dest='display',
                        help='Use TensorboardX to Display')
    parsers = parser.parse_args()

    main(parsers)
