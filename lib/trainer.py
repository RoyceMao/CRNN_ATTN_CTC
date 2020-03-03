# -*- coding:utf-8 -*-
"""
   File Name：     trainer.py
   Description :   
   Author :        royce.mao
   date：          2020/3/2 14:42
"""
import torch
from tqdm import tqdm

from lib.metrics import *


def train(model, optimizer, epoch, dataloader, device, cfg, writer=None):
    model.train()
    with tqdm(dataloader) as pbar:
        loss_mean = 0
        acc_mean = 0
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)

            output_log_softmax = F.log_softmax(output, dim=-1)

            loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)

            loss.backward()
            optimizer.step()

            loss = loss.item()
            acc, _ = calc_acc(target, output, cfg)

            if batch_index == 0:
                loss_mean = loss
                acc_mean = acc

            loss_mean = 0.1 * loss + 0.9 * loss_mean
            acc_mean = 0.1 * acc + 0.9 * acc_mean

            pbar.set_description(f'Epoch: {epoch} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')
        if not writer:
            writer.add_scalar('Train/Loss', loss_mean, epoch)
            writer.add_scalar('Train/Acc', acc_mean, epoch)


def valid(model, epoch, dataloader, device, cfg, writer=None):
    model.eval()
    with tqdm(dataloader) as pbar, torch.no_grad():
        loss_sum = 0
        acc_sum = 0
        acc_sum_ = 0
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            output = model(data)
            output_log_softmax = F.log_softmax(output, dim=-1)
            loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)

            loss = loss.item()
            acc, acc_ = calc_acc(target, output, cfg)

            loss_sum += loss
            acc_sum += acc
            acc_sum_ += acc_

            loss_mean = loss_sum / (batch_index + 1)
            acc_mean = acc_sum / (batch_index + 1)
            acc_mean_ = acc_sum_ / (batch_index + 1)

            pbar.set_description(
                f'Test : {epoch} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} Acc_relax: {acc_mean_:.4f} ')

        if not writer:
            writer.add_scalar('Test/Acc', acc_mean, epoch)

        return acc_mean
