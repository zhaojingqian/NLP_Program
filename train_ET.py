from utils.evaluate import Eval
from utils.config import *
from utils.data_utils import *
from pytorch_lightning.callbacks import ModelCheckpoint

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
    dirpath=ET_SAVE_PATH,  #模型缓存目录
    filename='{PRETRAINED_PATH}_ET-v1-{epoch:02d}-{val_acc:.4f}', # 模型名称
    verbose=True
)

checkpoint_callback_f1 = ModelCheckpoint(
    monitor='val_f1', #我们想要监视的指标 
    mode="max",
    dirpath=ET_SAVE_PATH,  #模型缓存目录
    filename='{PRETRAINED_PATH}_ET-v1-{epoch:02d}-{val_f1:.4f}', # 模型名称
    verbose=True
)

def train_entity_typing_model(ckpt_name):
    print("train_entity_typing_model .......")
    model = EntityTypingModel(max_length=64, batch_size=64)
    trainer = pl.Trainer(
        max_epochs=6,
        precision="bf16-mixed",
        # accumulate_grad_batches=2,
        devices=[1],
        default_root_dir=ET_SAVE_PATH,
        # profiler='simple',
        enable_checkpointing=True,
        callbacks=[checkpoint_callback_acc, checkpoint_callback_f1]
    )
    trainer.fit(model)
    trainer.save_checkpoint(CKPT_PATH + ckpt_name)
    
# training
train_ET = True
if train_ET:
    read_pickle() # 读取全局变量索引pickle文件
    train_entity_typing_model(f'{PRETRAINED_PATH}_ET_EPOCH6.ckpt')
    generate_type_tsv_result(f'{PRETRAINED_PATH}_ET_EPOCH6.ckpt')


