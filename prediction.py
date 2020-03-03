# -*- coding:utf-8 -*-
"""
   File Name：     prediction.py
   Description :   预测脚本
   Author :        royce.mao
   date：          2020/3/2 16:20
"""
import argparse
import torch
import torchvision.transforms as transforms

from lib.dataset import *
from lib.metrics import *
from utils.visual import *
from config import cur_config as cfg

ckpt = torch.load(cfg.CKPT_PATH)

test_datasets = DigitSet(cfg.TEST_PATH, cfg, one_hot=False, train=False)


def demo(model, dataset, *args):
    """
    数字序列预测demo，并可视化
    :param model:
    :param dataset:
    :return:
    """
    #
    device = torch.device("cuda:{}".format(args[0].gpu) if torch.cuda.is_available() else "cpu")

    digitimg = DigitPic()  # 数字串小图可视化 对象
    originimg = OriginPic()  # 数字串所在场景大图可视化 对象

    model.eval()
    for batch_index, (image, target, input_lengths, target_lengths) in enumerate(dataset):
        # 真实值
        true_ = decode_target(target, cfg)
        print('true:', true_)

        output = model(image.unsqueeze(0).to(device))
        score_matrix = F.softmax(output.detach().permute(1, 0, 2), dim=-1)
        score_values, output_argmax = score_matrix.max(dim=-1)

        # tensor转numpy
        score_values = score_values.cpu().numpy()
        output_argmax = output_argmax.cpu().numpy()

        # 预测值
        pred_ = decode_logit(score_values.flatten(), output_argmax.flatten(), cfg)
        print('pred:', pred_)

        # 可视化
        if args[0].vis == 'digit':
            image = transforms.ToPILImage()(image)
            image_txt = digitimg.to_txt_image(np.array(image), pred_)
            cv2.imwrite(os.path.join(cfg.OUT_DIR, 'test_image_{}_{}.jpg'.format(true_, batch_index)), image_txt)
        elif args[0].vis == 'origin':
            # todo:根据具体文件的分布完成
            image_txt = originimg.to_bbox_txt_image(*args)  # todo:修改
            cv2.imwrite(os.path.join(cfg.OUT_DIR, 'test_image_{}_{}.jpg'.format(true_, batch_index)), image_txt)
        else:
            raise ModuleNotFoundError


if __name__ == '__main__':
    """
        usage: python prediction.py --[params]
        """
    # 可调参数
    parser = argparse.ArgumentParser(description='PyTorch CRNN+CTC Training')

    parser.add_argument('--gpu', default='1', type=str,
                        help='GPU ID Select')
    parser.add_argument('--vis', default='digit',
                        type=str, help='visualization - digit or origin')
    parsers = parser.parse_args()

    demo(ckpt, test_datasets, parsers)
