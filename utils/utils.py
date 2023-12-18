import os
import json
import random
import torch
import numpy as np
import pandas as pd
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

def get_logger(filename):
    '''
    return output log
    '''
    logger = getLogger(__name__) # 定义logger
    logger.setLevel(INFO) # 定义过滤级别
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def seed_everything(seed=42):
    '''
    setting random seed for reproduction & multiple runs
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

