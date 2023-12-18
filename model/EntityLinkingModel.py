#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/7/7 上午10:50
  @ Author   : Vodka
  @ File     : EntityLinkingModel .py
  @ Software : PyCharm
"""
from utils.config import *
from processor.EntityLinkingProcessor import EntityLinkingProcessor
from sklearn.metrics import accuracy_score, f1_score

class EntityLinkingModel(pl.LightningModule):
    """实体链接模型"""

    def __init__(self, max_length=384, batch_size=32, use_pickle=True):
        super(EntityLinkingModel, self).__init__()
        # 输入最大长度
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_pickle = use_pickle

        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH)

        self.bert = AutoModelForSequenceClassification.from_pretrained(
            PRETRAINED_PATH,
            num_labels=1,
        )

        # 二分类损失函数
        self.criterion = nn.BCEWithLogitsLoss()
        self.validation_losss = []
        self.validation_accs = []
        self.preds_list = []
        self.labels_list = []

    def forward(self, input_ids, attention_mask, token_type_ids):
        logits = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]
        return logits.squeeze()

    def prepare_data(self):
        self.processor = EntityLinkingProcessor()
        self.train_examples = self.processor.get_train_examples(TSV_PATH + 'EL_TRAIN.tsv')
        self.valid_examples = self.processor.get_dev_examples(TSV_PATH + 'EL_VALID.tsv')
        self.test_examples = self.processor.get_test_examples(TSV_PATH + 'EL_TEST.tsv')

        self.train_loader = self.processor.create_dataloader(
            examples=self.train_examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            shuffle=True,
            batch_size=self.batch_size,
            use_pickle=self.use_pickle,
        )
        self.valid_loader = self.processor.create_dataloader(
            examples=self.valid_examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            shuffle=False,
            batch_size=self.batch_size,
            use_pickle=self.use_pickle,
        )
        self.test_loader = self.processor.create_dataloader(
            examples=self.test_examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            shuffle=False,
            batch_size=self.batch_size,
            use_pickle=self.use_pickle,
        )
        print("finish")

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        logits = self(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(logits, labels.float())

        preds = (logits > 0).int()
        acc = (preds == labels).float().mean()

        tensorboard_logs = {'train_loss': loss, 'train_acc': acc}
        print(tensorboard_logs)
        return {'loss': loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        logits = self(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(logits, labels.float())

        preds = (logits > 0).int()
        acc = (preds == labels).float().mean()
        
        self.validation_losss.append(loss)
        self.validation_accs.append(acc)
        self.preds_list.extend(preds.tolist())
        self.labels_list.extend(labels.tolist())

        return {'val_loss': loss, 'val_acc': acc}
    
    def on_validation_epoch_end(self):
        
        val_loss = torch.stack(self.validation_losss).mean()
        val_acc = torch.stack(self.validation_accs).mean()
        # 计算f1
        val_f1 = f1_score(self.labels_list, self.preds_list)

        tensorboard_logs = {'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1}
        print(f"Val Results: {tensorboard_logs}")
        self.log('val_acc', val_acc)
        self.log('val_f1', val_f1)
        self.validation_losss.clear()
        self.validation_accs.clear()
        self.preds_list.clear()
        self.labels_list.clear()
        
        return {'val_loss': val_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}
        
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-5, eps=1e-8)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader
