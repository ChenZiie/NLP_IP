import torch
import torch.nn as nn
from torch.nn import Flatten
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import random
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from configure.config import config


class DSSM_model(nn.Module):
    def __init__(self):
        super(DSSM_model, self).__init__()
        self.X_bert = my_bert()
        self.Y_bert = my_bert()
        self.batch_size = config['train_batch_size']

        self.mlp = nn.Sequential(
            nn.Linear(config["seq_feature_dim"], config['num_of_labels']))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, X_feat, Y_feat):
        # logits = self.mlp(torch.cat([X_feat, Y_feat], dim=-1))
        logits = self.mlp(X_feat)
        return logits
        # return Q_feat, A_feat

    def predictor_cos(self, X_feat, label_feat):
        X_feat = X_feat.unsqueeze(1).expand(-1, label_feat.shape[0], -1)
        label_feat = label_feat.unsqueeze(0).expand(X_feat.shape[0], -1, -1)
        # score = F.kl_div(q_feat.softmax(dim=-1).log(), label_feat.softmax(dim=-1), reduction='none')
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        score = torch.exp(cos(X_feat, label_feat))
        return score

    def predictor_cos_top_k(self, X_feat, label_feat, index, device):
        X_feat = X_feat.unsqueeze(1).expand(-1, index.shape[1], -1)
        #print(X_feat.size())
        topk = []
        for i in range(X_feat.shape[0]):
            topk_ = label_feat[index[i]].cpu().detach().numpy().tolist()
            topk.append(topk_)
            # if i == 0:
            #     topk = label_feat[index[i]]
            # else:
            #     topk = torch.stack([topk, label_feat[index[i]]], dim=0)
        #print(len(topk))
        topk = torch.tensor(topk).to(device)
        #print(topk.size())

        # score = F.kl_div(q_feat.softmax(dim=-1).log(), label_feat.softmax(dim=-1), reduction='none')
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        score = torch.exp(cos(X_feat, topk))
        return score


class my_bert(nn.Module):
    def __init__(self):
        super(my_bert, self).__init__()
        self.bert = AutoModel.from_pretrained(config["pretrain_model"])
        # self.flatten = Flatten()
        # self.l1= nn.Sequential(
        #     nn.Linear(config["seq_feature_dim"] * config["max_seq_Q_len"], config["seq_feature_dim"] * 10),
        #     nn.ReLU(),
        #     nn.Linear(config["seq_feature_dim"]* 10, config["seq_feature_dim"] * 5),
        #     nn.ReLU(),
        #     nn.Linear(config["seq_feature_dim"] * 5, config["seq_feature_dim"])
        # )
        # self.criterion = nn.MSELoss()

    def mean_pooling(self, output, attention_mask):
        token_embeddings = output  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, x):
        output = self.bert(**x)
        # output = self.flatten(output)
        # sentence_embeddings = self.l1(output)
        # Perform pooling. In this case, mean pooling
        sentence_embeddings = self.mean_pooling(output[0], x['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.batch_size = config['train_batch_size']
        self.X_mlp = nn.Sequential(
            nn.Linear(config["seq_feature_dim"], config["seq_feature_dim"] + 20))
        self.Y_mlp = nn.Sequential(
            nn.Linear(config["seq_feature_dim"], config["seq_feature_dim"] + 20))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, X_feat, Y_feat):
        # logits = self.mlp(torch.cat([X_feat, Y_feat], dim=-1))
        logits = self.mlp(X_feat)
        return logits