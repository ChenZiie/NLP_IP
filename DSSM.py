import warnings

warnings.filterwarnings('ignore')
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from torch import optim
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
from configure.config import config
from loss_function import bpr_loss, bpr_loss1
from model import DSSM_model
from data_prepare import data_dict_train,data_dict_test,encoder_train,encoder_test,data_augmentation_translation,data_augmentation_Paraphrase,data_augmentation_shuffle
import glob




def training(epoch, model, train_iter, label_data, optimizer, device):
    num, loss_ = 0, 0.0
    criterion = nn.CrossEntropyLoss()
    # for i, batch in tqdm(next(iter(train_iter))):
    for i, batch in tqdm(enumerate(train_iter)):
        optimizer.zero_grad()
        num += 1
        q, a, y = batch
        for key in q.keys():
            q[key] = q[key].to(device)
            a[key] = a[key].to(device)
        X_feat = model.X_bert(q)
        Y_feat = model.Y_bert(a)
        # loss = bpr_loss1(X_feat, Y_feat, model.Y_bert(label_data), X_feat.shape[0])
        loss = bpr_loss(X_feat, Y_feat, X_feat.shape[0])
        loss_ += loss.item()
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
            optimizer.step()

    print("epoch: " + str(epoch) + " average loss: " + str(loss_ / num))

def testing(model, test_iter, label_feat, device):
    model.eval()
    label = []
    pred = []
    for i, batch in tqdm(enumerate(test_iter)):
        q, a_id = batch
        #print(q,a_id)
        for key in q.keys():
            q[key] = q[key].to(device)
        X_feat = model.X_bert(q)
        res = model.predictor_cos(X_feat, label_feat)
        pred += torch.argmax(res, dim=1).cpu().numpy().tolist()
        label += a_id.cpu().numpy().tolist()

    acc = accuracy_score(label, pred) * 100
    f1 = f1_score(label, pred, average='macro') * 100
    print('Accuracy: ', acc, ', F1 score: ',f1)
    return acc, f1


if __name__ == '__main__':
    seed_val = 1024
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # train = pd.read_csv('data9/train.tsv', sep='\t', header=None, names=['X', 'Y'])

    # train = data_augmentation_Paraphrase(train,30)
    # train = data_augmentation_shuffle(train,30)

    train = pd.read_csv('data9/Translation_30_train_data.csv',names=['X', 'Y'])
    #train = data_augmentation_translation(train,30)
    #train.to_csv('data9/Translation_30_train_data.csv',index=False)
    test = pd.read_csv('data9/test.tsv', sep='\t', header=None, names=['X', 'Y'])

    train, x_train_dic, y_dic = data_dict_train(train)
    test, x_test_dic = data_dict_test(test,y_dic)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(config['pretrain_model'])
    train_data = encoder_train(train, tokenizer)
    test_data = encoder_test(test,tokenizer)
    train_iter = data.DataLoader(
        dataset=train_data,
        batch_size=config['train_batch_size'],
        shuffle=True,
        num_workers=2)
    test_iter = data.DataLoader(
        dataset=test_data,
        batch_size=config['test_batch_size'],
        shuffle=True,
        num_workers=2)

    model = DSSM_model()
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["learn_rate"])
    scheduler = StepLR(optimizer, step_size=config["lr_dc_step"], gamma=config["lr_dc"])

    label_data = tokenizer(
        list(y_dic.keys()),
        add_special_tokens=True,
        max_length=config['max_seq_Q_len'],
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    for label in label_data.keys():
        label_data[label] = label_data[label].to(device)

    print("Train start")
    max = 0
    for filename in glob.glob('*.pth'):
        if max < float(filename.split('_')[3]):
            max = float(filename.split('_')[3])

    for epoch in range(1, config['train_epoch'] + 1):
        training(epoch, model, train_iter, label_data, optimizer, device)
        scheduler.step(epoch=epoch)
        if epoch % 1 == 0:
            print('Test result in epoch', epoch)
            # label_feat = model.Y_bert(label_data)
            acc, f1 =testing(model, test_iter, model.Y_bert(label_data), device)
            if acc > max:
                torch.save(model.state_dict(),'Best_DSSM_acc_'+str(acc)+'_f1_'+str(f1)+'.pth')


