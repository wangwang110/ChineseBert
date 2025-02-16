#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : bert_mask_dataset.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2021/6/26 15:27
@version: 1.0
@desc  : Dataset for fill mask sentence
"""
import json
import os
from typing import List

import tokenizers
import torch
from pypinyin import pinyin, Style
from tokenizers import BertWordPieceTokenizer


class BertMaskDataset(object):

    def __init__(self, vocab_file, config_path, max_length: int = 512):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = BertWordPieceTokenizer(vocab_file)

        # load pinyin map dict
        with open(os.path.join(config_path, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # load char id map tensor
        with open(os.path.join(config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)  # 汉字对应的拼音，有可能对应两个
        # load pinyin map tensor
        with open(os.path.join(config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)

    def mask_sentence(self, sentence, location):
        # convert sentence to ids
        tokenizer_output = self.tokenizer.encode(sentence)
        bert_tokens = tokenizer_output.ids
        pinyin_tokens = self.convert_sentence_to_pinyin_ids(sentence, tokenizer_output)
        # add mask 要预测的？ 没有考虑字形的特征？字形特征在哪，字形特征在哪？字形特征在哪
        bert_tokens[location + 1] = 103
        pinyin_tokens[location + 1] = [0] * 8
        # assert，token nums should be same as pinyin token nums
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)
        # convert list to tensor
        input_ids = torch.LongTensor(bert_tokens)
        pinyin_ids = torch.LongTensor(pinyin_tokens).view(-1)
        return input_ids, pinyin_ids

    def convert_sentence_to_pinyin_ids(self, sentence: str, tokenizer_output: tokenizers.Encoding) -> List[List[int]]:
        """
        self.tokenizer.convert_sentence_to_pinyin_ids(src, tokenizer_output)
        :param sentence:
        :param tokenizer_output: 如何用起来的
        :return:
        """
        # get pinyin of a sentence
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        # find chinese character location, and generate pinyin ids
        pinyin_ids = []
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_ids.append([0] * 8)
                continue
            if offset[0] in pinyin_locs:
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                pinyin_ids.append([0] * 8)

        return pinyin_ids

    # def convert_sentence_to_pinyin_ids(self, sentence: str, tokenizer_output: tokenizers.Encoding) -> List[List[int]]:
    #     """
    #     self.tokenizer.convert_sentence_to_pinyin_ids(src, tokenizer_output)
    #     :param sentence:
    #     :param tokenizer_output: 如何用起来的
    #     :return:
    #     """
    #     # get pinyin of a sentence
    #     pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
    #     pinyin_locs = {}
    #     # get pinyin of each location
    #     for index, item in enumerate(pinyin_list):
    #         pinyin_string = item[0]
    #         # not a Chinese character, pass
    #         if pinyin_string == "not chinese":
    #             continue
    #         if pinyin_string in self.pinyin2tensor:
    #             pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
    #         else:
    #             ids = [0] * 8
    #             for i, p in enumerate(pinyin_string):
    #                 if p not in self.pinyin_dict["char2idx"]:
    #                     ids = [0] * 8
    #                     break
    #                 ids[i] = self.pinyin_dict["char2idx"][p]
    #             pinyin_locs[index] = ids
    #
    #     # find chinese character location, and generate pinyin ids
    #     pinyin_ids = []
    #     for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
    #         if offset[1] - offset[0] != 1:
    #             pinyin_ids.append([0] * 8)
    #             continue
    #         if offset[0] in pinyin_locs:
    #             pinyin_ids.append(pinyin_locs[offset[0]])
    #         else:
    #             pinyin_ids.append([0] * 8)
    #
    #     return pinyin_ids


if __name__ == '__main__':
    vocab_file = "/data/nfsdata2/sunzijun/glyce/glyce/bert_chinese_base_large_vocab/vocab.txt"
    config_path = "/data/nfsdata2/sunzijun/glyce/glyce/config"
    sentence = "我喜欢猫。"
    tokenizer = BertMaskDataset(vocab_file, config_path)
    input_ids, pinyin_ids = tokenizer.mask_sentence(sentence, 1)
    print(pinyin_ids)
    print(input_ids)
