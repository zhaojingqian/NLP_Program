#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/7/7 上午10:41
  @ Author   : Vodka
  @ File     : config .py
  @ Software : PyCharm
"""
import os
import json
import logging
from tqdm import tqdm
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from transformers import (
    DataProcessor,
    InputExample,
    BertConfig,
    BertTokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    glue_convert_examples_to_features,
)

import pytorch_lightning as pl

DEVICE = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
MAX_NEGS = 3 # Linking_data的最多负样本

# 预训练模型路径
PRETRAINED_PATH = './roberta-chinese-large' # "./ernie-3.0-base-zh"  # './roberta-chinese-large'

# Pytorch路径
PYTORCH_PATH = './torch-checkpoints/'
if not os.path.exists(PYTORCH_PATH):
    os.mkdir(PYTORCH_PATH)

# 实体链接训练路径
EL_SAVE_PATH = PYTORCH_PATH + 'EntityLinking/'
if not os.path.exists(EL_SAVE_PATH):
    os.mkdir(EL_SAVE_PATH)

# 实体类别推断训练路径
ET_SAVE_PATH = PYTORCH_PATH + 'EntityTyping/'
if not os.path.exists(ET_SAVE_PATH):
    os.mkdir(ET_SAVE_PATH)

# 项目数据路径
DATA_PATH = './data/'

# CCKS2020实体链指竞赛原始路径
RAW_PATH = DATA_PATH

# 预处理后导出的pickle文件路径
PICKLE_PATH = DATA_PATH + 'pickle/'
if not os.path.exists(PICKLE_PATH):
    os.mkdir(PICKLE_PATH)

# 预测结果的文件路径
RESULT_PATH = DATA_PATH + 'result/'
if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)

# 训练、验证、推断所需的tsv文件路径
TSV_PATH = DATA_PATH + 'tsv/'
if not os.path.exists(TSV_PATH):
    os.mkdir(TSV_PATH)

# 训练结果的CheckPoint文件路径
CKPT_PATH = './ckpt/'
if not os.path.exists(CKPT_PATH):
    os.mkdir(CKPT_PATH)

PICKLE_DATA = {
    # 实体名称对应的KBID列表
    'ENTITY_TO_KBIDS': None,
    # KBID对应的实体名称列表
    'KBID_TO_ENTITIES': None,
    # KBID对应的属性文本
    'KBID_TO_TEXT': None,
    # KBID对应的实体类型列表（注意：一个实体可能对应'|'分割的多个类型）
    'KBID_TO_TYPES': None,
    # KBID对应的属性列表
    'KBID_TO_PREDICATES': None,

    # 索引类型映射列表
    'IDX_TO_TYPE': None,
    # 类型索引映射字典
    'TYPE_TO_IDX': None,
}


def read_pickle():
    for k in PICKLE_DATA:
        filename = k + '.pkl'
        if os.path.exists(PICKLE_PATH + filename):
            PICKLE_DATA[k] = pd.read_pickle(PICKLE_PATH + filename)
        else:
            print(f'File {filename} not Exist!')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
