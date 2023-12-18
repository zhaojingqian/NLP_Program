# %%
from utils.config import *
PRETRAINED_PATH = "./ernie-3.0-base-zh" # set PLM
from utils.evaluate import Eval
from utils.data_utils import *

data_pro = False
if data_pro:
    set_random_seed(3407) # seed everything
    preprocess_pickle_file()  # 生成全局变量索引pickle文件
    preprocess_tsv_file()  # 生成模型训练、验证、推断所需的tsv文件

gen_dataloader = False
if gen_dataloader:
    read_pickle() # 读取全局变量索引pickle文件
    generate_feature_pickle() # 生成特征pickle文件

# training >>EL_log.txt 2>&1 &
train_EL = True
if train_EL:
    train_entity_linking_model(f'{PRETRAINED_PATH}_EL_EPOCH8.ckpt')
    generate_link_tsv_result(f'{PRETRAINED_PATH}_EL_EPOCH8.ckpt')


