from utils.evaluate import Eval
from utils.config import *
from utils.data_utils import *

## 在这里设置超参数
epochs = 6
max_len = 64
bs = 64
precision = "bf16-mixed"
model_name = f'{PRETRAINED_PATH}_ET_EPOCH{epochs}_1stage_v1.ckpt' # version 0

data_pro = False
if data_pro:
    set_random_seed(3407) # seed everything
    preprocess_pickle_file()  # 生成全局变量索引pickle文件
    preprocess_tsv_file()  # 生成模型训练、验证、推断所需的tsv文件


gen_dataloader = False
if gen_dataloader:
    read_pickle() # 读取全局变量索引pickle文件
    generate_feature_pickle() # 生成特征pickle文件

checkpoint_callback_acc = ModelCheckpoint(
    monitor='val_acc', #我们想要监视的指标 
    mode="max",
    dirpath=ET_SAVE_PATH,  # 模型缓存目录
    filename=model_name+'{epoch:02d}-{val_acc:.4f}', # 模型名称
    verbose=True
)

checkpoint_callback_f1 = ModelCheckpoint(
    monitor='val_f1', #我们想要监视的指标 
    mode="max",
    dirpath=ET_SAVE_PATH,  # 模型缓存目录
    filename=model_name+'-{epoch:02d}-{val_f1:.4f}', # 模型名称
    verbose=True
)

def train_entity_typing_model(ckpt_name):
    print("train_entity_typing_model .......")
    model = EntityTypingModel(max_length=max_len, batch_size=bs)
    trainer = pl.Trainer(
        max_epochs=epochs,
        precision=precision,
        # accumulate_grad_batches=2,
        devices=[0],
        default_root_dir=ET_SAVE_PATH,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback_acc, checkpoint_callback_f1]
    )
    trainer.fit(model)
    trainer.save_checkpoint(CKPT_PATH + ckpt_name)
    
# training
train_ET = True
if train_ET:
    read_pickle() # 读取全局变量索引pickle文件
    train_entity_typing_model(model_name)
    generate_type_tsv_result(model_name)


