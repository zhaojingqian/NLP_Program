#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/7/7 上午10:49
  @ Author   : Vodka
  @ File     : EntityLinkingProcessor .py
  @ Software : PyCharm
"""
from utils.config import *


class EntityLinkingProcessor(DataProcessor):
    """实体链接数据处理"""

    def get_train_examples(self, file_path):
        return self._create_examples(
            self._read_tsv(file_path),
            set_type='train',
        )

    def get_dev_examples(self, file_path):
        return self._create_examples(
            self._read_tsv(file_path),
            set_type='valid',
        )

    def get_test_examples(self, file_path):
        return self._create_examples(
            self._read_tsv(file_path),
            set_type='test',
        )

    def get_labels(self):
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = f'{set_type}-{i}'
            text_a = line[1] + ' [SEP] ' + line[3] ## add SEP token
            text_b = line[5]
            label = line[-1]
            examples.append(InputExample(
                guid=guid,
                text_a=text_a,
                text_b=text_b,
                label=label,
            ))
        return examples

    def create_dataloader(self, examples, tokenizer, max_length=384,
                          shuffle=False, batch_size=32, use_pickle=False):
        pickle_name = 'EL_FEATURE_' + examples[0].guid.split('-')[0].upper() + '.pkl'
        if use_pickle:
            features = pd.read_pickle(PICKLE_PATH + pickle_name)
        else:
            features = glue_convert_examples_to_features(
                examples,
                tokenizer,
                label_list=['0', '1'],
                max_length=max_length,
                output_mode='classification',
            )
            pd.to_pickle(features, PICKLE_PATH + pickle_name)

        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor([f.input_ids for f in features]),
            torch.LongTensor([f.attention_mask for f in features]),
            torch.LongTensor([f.token_type_ids for f in features]),
            torch.LongTensor([f.label for f in features]),
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=4,
        )
        return dataloader

    def generate_feature_pickle(self, max_length):
        tokenizer = BertTokenizer.from_pretrained(PRETRAINED_PATH)

        train_examples = self.get_train_examples(TSV_PATH + 'EL_TRAIN.tsv')
        valid_examples = self.get_dev_examples(TSV_PATH + 'EL_VALID.tsv')
        test_examples = self.get_test_examples(TSV_PATH + 'EL_TEST.tsv')

        self.create_dataloader(
            examples=train_examples,
            tokenizer=tokenizer,
            max_length=max_length,
            shuffle=True,
            batch_size=32,
            use_pickle=False,
        )
        self.create_dataloader(
            examples=valid_examples,
            tokenizer=tokenizer,
            max_length=max_length,
            shuffle=False,
            batch_size=32,
            use_pickle=False,
        )
        self.create_dataloader(
            examples=test_examples,
            tokenizer=tokenizer,
            max_length=max_length,
            shuffle=False,
            batch_size=32,
            use_pickle=False,
        )
