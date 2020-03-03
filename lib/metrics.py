# -*- coding:utf-8 -*-
"""
   File Name：     metric.py
   Description :   Acc计算，输出RNN序列到最终label序列的解码
   Author :        royce.mao
   date：          2020/3/2 11:10
"""
import numpy as np
import torch.nn.functional as F


def decode_logit(seq_score, seq_index, cfg):
    """
    n_rnn_len长度的seq_char -> 最终长度为n_char的seq_remain（定长：假设为7）
    :param seq_score: len == 7 每个位置的得分
    :param seq_index: len == 7 每个位置的索引
    :param cfg:
    :return:
    """
    # crnn的输出序列
    seq_char = ''.join([cfg.NUMS[x] for x in seq_index])

    # 过滤两两'-'之间，时间点连续字符，再过滤所有'-'空白字符
    seq_remain = ''.join(
        [x for i, x in enumerate(seq_char[:-1]) if x != cfg.NUMS[0] and x != seq_char[i + 1]] + [seq_char[-1]])
    score_remain = np.array(
        [seq_score[i] for i, x in enumerate(seq_char[:-1]) if x != cfg.NUMS[0] and x != seq_char[i + 1]] + [
            seq_score[-1]])

    # 没有数字检测到
    if len(score_remain) == 0:
        return ''

    # 检测的数字串长度大于n_char（剔除softmax概率较低的）
    elif len(score_remain) > cfg.N_CHAR:
        inds_remain = score_remain > np.sort(score_remain)[len(score_remain) - cfg.N_CHAR - 1]
        seq_remain = ''.join([seq_remain[i] for i, ind in enumerate(inds_remain) if ind])

    # 检测的数字串长度小于n_char（拿crnn输出序列的末尾补全）
    elif len(score_remain) < cfg.N_CHAR:
        seq_remain += seq_char[-(cfg.N_CHAR - len(score_remain)):]

    return seq_remain


def decode_target(sequence, cfg):
    """
    n_char长度的target sequence -> 直接解码
    :param sequence:
    :param cfg:
    :return:
    """
    return ''.join([cfg.NUMS[x] for x in sequence]).replace(' ', '')


def calc_acc(target, output, cfg):
    """
    Acc指标的计算：1）acc要求数字串全对，2）acc_relax要求放松，每个位置一一比对
    :param target: [batch_size, n_char]
    :param output: [n_rnn_len, batch_size, n_classes]
    :param cfg:
    :return:
    """
    # softmax激活，概率标准化处理
    score_matrix = F.softmax(output.detach().permute(1, 0, 2), dim=-1)
    # 取softmax概率值最高，对应的score与index
    score_values, output_argmax = score_matrix.max(dim=-1)

    # tensor转numpy
    score_values = score_values.cpu().numpy()
    output_argmax = output_argmax.cpu().numpy()
    target = target.cpu().numpy()

    # batch维度decode_logit，decode_target
    logit_decoded = [decode_logit(score_value, output_sequence, cfg) for score_value, output_sequence in
                     zip(score_values, output_argmax)]
    target_decoded = [decode_target(sequence, cfg) for sequence in target]

    # 1、数字串整体比对求acc
    acc_matrix = np.array([true == pred for true, pred in zip(target_decoded, logit_decoded)])

    # 2、位置一一比对求acc_relax
    acc_relax_matrix = np.array([np.array(list(true)) == np.array(list(pred)) if len(true) == len(pred) else
                                 np.concatenate([np.array(list(true))[:min(len(true), len(pred))] == np.array(
                                     list(pred))[:min(len(true), len(pred))],
                                                 np.tile(np.array([False]),
                                                         max(len(true), len(pred)) - min(len(true), len(pred)))])
                                 for true, pred in zip(target_decoded, logit_decoded)])

    return acc_matrix.mean(), acc_relax_matrix.mean()
