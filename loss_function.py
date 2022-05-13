import numpy as np
import pandas as pd
from torch import optim
import torch
import torch.nn as nn
import random
from tqdm import tqdm
from configure.config import config


def bpr_loss(X, Y, batch_size):
    loss = 0
    cos1 = nn.CosineSimilarity(dim=0, eps=1e-6)
    cos2 = nn.CosineSimilarity(dim=1, eps=1e-6)
    for idx in range(batch_size):
        l = list(range(batch_size))
        del l[idx]
        num_neg = batch_size - 1
        neg_idx = random.sample(l, num_neg)

        res_pos = torch.exp(cos1(X[idx], Y[idx]) / config['temperature'])#.unsqueeze(0).repeat(num_neg, 1)
        res_neg = torch.exp(
            cos2(X[idx].unsqueeze(0).expand(num_neg, -1), Y[neg_idx]) / config['temperature'])#.unsqueeze(1)
        res = -1 * torch.log(res_pos / torch.sum(res_neg))

        # res_pos = cos1(X[idx], Y[idx]) / config['temperature']
        # res_neg = cos2(X[idx].unsqueeze(0).expand(num_neg, -1), Y[neg_idx]) / config['temperature']
        # res = -1 * (res_pos -  torch.sum(res_neg))

        loss = loss + res

    return loss / (batch_size - 1)


def bpr_loss1(X, Y, Y_dic,batch_size):
    loss = 0
    for idx in range(batch_size):
        l = list(range(config['num_of_labels']))
        l1 = torch.nonzero(Y[idx] == Y_dic)[0][0]
        l.remove(l1.item())
        # l = random.sample(l, batch_size - 1)
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        res_pos = torch.exp(cos(X[idx], Y[idx]) / config['temperature'])#.unsqueeze(0).repeat(num_neg, 1)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        res_neg = torch.exp(
            cos(X[idx].unsqueeze(0).expand(config['num_of_labels'] - 1, -1), Y_dic[l]) / config['temperature'])#.unsqueeze(1)
        # res_neg = torch.exp(
        #     cos(X[idx].unsqueeze(0).expand(batch_size - 1, -1), Y_dic[l]) / config['temperature'])#.unsqueeze(1)
        res = -1 * torch.log(res_pos / torch.sum(res_neg))
        loss = loss + res
    return loss / (batch_size - 1)


def bpr_cros_loss(X, Y, batch_size,Y_dic):
    loss = 0
    cos1 = nn.CosineSimilarity(dim=0, eps=1e-6)
    cos2 = nn.CosineSimilarity(dim=1, eps=1e-6)
    onehottensor = []
    for idx in range(batch_size):
        l = list(range(batch_size))
        del l[idx]
        num_neg = batch_size - 1
        neg_idx = random.sample(l, num_neg)
        res_pos = torch.exp(cos1(X[idx], Y[idx]) / config['temperature'])#.unsqueeze(0).repeat(num_neg, 1)
        res_neg = torch.exp(
            cos2(X[idx].unsqueeze(0).expand(num_neg, -1), Y[neg_idx]) / config['temperature'])#.unsqueeze(1)
        bpr_res = -1 * torch.log(res_pos / torch.sum(res_neg))

        l1 = torch.nonzero(Y[idx] == Y_dic)[0][0]
        onehot = [0.0]*config['num_of_labels']
        onehot[l1] = 1.0
        onehottensor.append(onehot)

        loss = loss + bpr_res

    return loss / (batch_size - 1), torch.tensor(onehottensor)