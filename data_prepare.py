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
from tqdm import tqdm
from configure.config import config
from google_trans_new import google_translator

def func_for_low_Paraphrase(dataset,groups,target,tokenizer,model,device):
    new_x = []
    new_y = []
    for i in groups.keys():
        y = i
        str = groups[i][0]
        end = groups[i][-1]
        cnt = int(dataset[str:str+1]['cnt'])
        needed_generat = target - cnt
        per_data = needed_generat//cnt
        temp = dataset[str:end+1]
        temp = list(temp[0])
        generated_count = 0
        for j in temp:
            encoding = tokenizer.encode_plus(j, pad_to_max_length=True, return_tensors="pt")
            input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
            if j != temp[-1]:
                outputs = model.generate(
                    input_ids=input_ids, attention_mask=attention_masks,
                    max_length=64,
                    do_sample=True,
                    top_k=120,
                    top_p=0.95,
                    early_stopping=True,
                    num_return_sequences=per_data
                )
                generated_count += per_data
            else:
                outputs = model.generate(
                    input_ids=input_ids, attention_mask=attention_masks,
                    max_length=64,
                    do_sample=True,
                    top_k=120,
                    top_p=0.95,
                    early_stopping=True,
                    num_return_sequences= needed_generat - generated_count
                )
            for output in outputs:
                line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                new_x.append(line)
                new_y.append(y)
    return pd.DataFrame({0:new_x,1:new_y})

def func_for_heigh_Paraphrase(data,tokenizer,model,device):
    encoding = tokenizer.encode_plus(data, pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=64,
        do_sample=True,
        top_k=120,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=1
    )
    line = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return line

def data_augmentation_Paraphrase(dataset,target): #max:429
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    model.to(device)

    aug_data = dataset.groupby([1], as_index=False)[1].agg({'cnt': 'count'})

    aug_data = pd.merge(dataset, aug_data, how='inner', on=[1])

    aug_data_low = aug_data[aug_data['cnt'] <= target // 2]

    if len(aug_data_low) > 0:
        aug_data_low.index = range(0, len(aug_data_low))
        aug_data_low[0] = aug_data_low[0].apply(lambda x:"paraphrase: " + x + " </s>")
        grouped = aug_data_low.groupby(1, group_keys=True).groups
        aug_data_low = func_for_low_Paraphrase(aug_data_low,grouped,target,tokenizer,model,device)
        if len(aug_data_low) > 0:
            dataset = pd.concat([dataset, aug_data_low])


    aug_data_heigh = aug_data[(aug_data['cnt'] > target // 2) & (aug_data['cnt'] < target)]
    if len(aug_data_heigh) > 0:
        grouped = aug_data_heigh.groupby(1, group_keys=False)
        aug_data_heigh = grouped.apply(lambda x: x.sample(target-x['cnt'].iloc[0]))
        aug_data_heigh[0] = aug_data_heigh[0].apply(lambda x:"paraphrase: " + x + " </s>")
        aug_data_heigh[0] = aug_data_heigh[0].apply(lambda x: func_for_heigh_Paraphrase(x,tokenizer,model,device))

    if len(aug_data_heigh) > 0:
        aug_data_heigh = aug_data_heigh.drop('cnt', axis=1)
        dataset = pd.concat([dataset, aug_data_heigh])

    dataset.index = range(0, len(dataset))
    return dataset


def pre_translation(aug_data, target,translator):
    lang_list = ['af', 'sq', 'az', 'eu', 'be', 'bn', 'bs', 'bg', 'ca', 'ceb', 'ny', 'zh-CN', 'zh-TW', 'co', 'cs', 'da',
                 'nl', 'eo', 'et', 'fi', 'fr', 'fy', 'gl', 'ka', 'de', 'el', 'gu', 'ht', 'ha', 'haw', 'hi', 'hmn', 'hu',
                 'is', 'ig', 'id', 'ga', 'jw', 'kn', 'kk', 'km', 'ku', 'lo', 'la', 'lv', 'lt', 'lb', 'mk', 'mg', 'ms',
                 'ml', 'mt', 'mi', 'mr', 'mn', 'ne', 'no', 'or', 'pl', 'pt', 'sm', 'sr', 'st', 'sn', 'sd', 'si', 'sk',
                 'sl', 'su', 'sw', 'sv', 'tg', 'ta', 'te', 'th', 'tr', 'uk', 'ur', 'vi', 'cy', 'yi', 'zu']
    lang = random.choice(lang_list)
    aug_data_low = aug_data[aug_data['cnt'] <= target//2]
    if len(aug_data_low) > 0 :
        aug_data_low[0] = aug_data_low[0].apply(lambda x:
            translator.translate(translator.translate(x, lang_src='en', lang_tgt=lang), lang_src=lang, lang_tgt='en'))

    aug_data_heigh = aug_data[aug_data['cnt'] > target//2]
    if len(aug_data_heigh) > 0:
        grouped = aug_data_heigh.groupby(1, group_keys=False)
        aug_data_heigh = grouped.apply(lambda x: x.sample(target-x['cnt'].iloc[0]))
        aug_data_heigh[0] = aug_data_heigh[0].apply(lambda x:
            translator.translate(translator.translate(x, lang_src='en', lang_tgt=lang), lang_src=lang, lang_tgt='en'))

    return aug_data_low, aug_data_heigh


def data_augmentation_translation(dataset,target): #max:429
    translator = google_translator()  #new

    aug_data = dataset.groupby([1], as_index=False)[1].agg({'cnt': 'count'})

    aug_data = pd.merge(dataset, aug_data, how='inner', on=[1])
    aug_data = aug_data[aug_data['cnt'] < target]
    while len(aug_data) > 0:
        aug_data_low, aug_data_heigh = pre_translation(aug_data, target,translator)

        if len(aug_data_low) > 0:
            aug_data_low = aug_data_low.drop('cnt', axis=1)
            dataset = pd.concat([dataset, aug_data_low])
        if len(aug_data_heigh) > 0:
            aug_data_heigh = aug_data_heigh.drop('cnt', axis=1)
            dataset = pd.concat([dataset, aug_data_heigh])

        aug_data = dataset.groupby([1], as_index=False)[1].agg({'cnt': 'count'})
        aug_data = pd.merge(dataset, aug_data, how='inner', on=[1])
        aug_data = aug_data[aug_data['cnt'] < target]

    dataset.index = range(0, len(dataset))
    return dataset


def data_augmentation_shuffle(dataset):
    count = dataset.groupby(['Y'], as_index=False)['Y'].agg({'cnt': 'count'})
    count = pd.merge(dataset, count, how='inner', on='Y')
    new = count[count['cnt'] < count['cnt'].max()//2]
    aug_data = list(new['X'])
    for i in range(len(aug_data)):
        temp1 = ''
        temp2 = aug_data[i].split()
        random.shuffle(temp2)
        for j in temp2:
            temp1 += j + ' '
        temp1 = temp1[:-1]
        aug_data[i] = temp1
    new['X'] = aug_data
    new = new.drop('cnt',axis=1)
    # print(len(new),len(dataset))
    dataset = pd.concat([dataset,new])
    dataset.index = range(0,len(dataset))
    return dataset

def data_dict_train(dataset):
    x = set(dataset['X'])
    y = set(dataset['Y'])
    x_dict = dict()
    y_dict = dict()
    for key, value in enumerate(x):
        x_dict[value] = key
    for key, value in enumerate(y):
        y_dict[value] = key
    xid = [x_dict[v] for v in dataset['X']]
    dataset['xid'] = xid
    yid = [y_dict[v] for v in dataset['Y']]
    dataset['yid'] = yid
    return dataset, x_dict, y_dict

def data_dict_test(dataset,y_dict):
    x = set(dataset['X'])
    y = set(dataset['Y'])
    x_dict = dict()
    for key, value in enumerate(x):
        x_dict[value] = key
    xid = [x_dict[v] for v in dataset['X']]
    dataset['xid'] = xid
    yid = [y_dict[v] for v in dataset['Y']]
    dataset['yid'] = yid
    return dataset, x_dict

class encoder_train(Dataset):
    def __init__(self, dataest, tokenizer):
        self.x = dataest['X']
        self.y = dataest['Y']
        self.config = config
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        data_x = self.tokenizer.encode_plus(
            self.x[idx],
            add_special_tokens=True,
            max_length=self.config['max_seq_Q_len'],
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        data_x = {
            'input_ids': data_x['input_ids'].flatten(),
            'attention_mask': data_x['attention_mask'].flatten(),
        }

        data_y = self.tokenizer.encode_plus(
            self.y[idx],
            add_special_tokens=True,
            max_length=self.config['max_seq_A_len'],
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        data_y = {
            'input_ids': data_y['input_ids'].flatten(),
            'attention_mask': data_y['attention_mask'].flatten(),
        }

        return data_x, data_y, 1

    def __len__(self):
        return len(self.x)

class encoder_test(Dataset):
    def __init__(self, dataset, tokenizer):
        self.x = dataset['X']
        self.yid = dataset['yid']
        self.config = config
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        data_x = self.tokenizer.encode_plus(
            self.x[idx],
            add_special_tokens=True,
            max_length=self.config['max_seq_Q_len'],
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        data_x = {
            'input_ids': data_x['input_ids'].flatten(),
            'attention_mask': data_x['attention_mask'].flatten(),
            'token_type_ids': data_x['token_type_ids'].flatten()

        }
        return data_x, self.yid[idx]

    def __len__(self):
        return len(self.yid)

