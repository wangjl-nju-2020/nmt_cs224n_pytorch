# @Author: wangjl
# @Email: If you need any help, feel free to contact me via wangjl.nju.2020@gmail.com.

from typing import Any, List

import os
import nltk
import vocab
import torch
import random
import numpy as np


def read_corpus(path: str, split_func: Any = None):
    """
    读取语料，并使用特定的数据行分割函数处理

    Args:
        path (str): 语料库存储地址
        split_func (Any): 数据行分割函数

    Returns:
        data (List): 分割过的数据行列表
    """
    data = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                if split_func is not None:
                    words = split_func(line)
                    data.append(words)
                else:
                    data.append([line])
        return data
    raise FileNotFoundError


def split_by_space(data_line: str):
    """
    使用空格分割数据行

    Args:
        data_line (str): 数据行

    Returns:
        (List[str]): 分割后的数据行
    """
    return str(data_line).strip().split(' ')


def split_by_tokenize(data_line: str):
    """
    使用NLTK包分割数据行

    Args:
        data_line (str): 数据行

    Returns:
        (List[str]): 分割后的数据行
    """
    return nltk.tokenize.word_tokenize(data_line.lower())


def split_and_pad(data_line: str):
    """
    按分词划分数据行并填充起始、终止符号

    Args:
        data_line (str): 数据行

    Returns:
        data (List[str): 分割并填充后的数据行
    """
    data = [vocab.Vocabulary.start_token()]
    data.extend(split_by_tokenize(data_line.lower()))
    data.append(vocab.Vocabulary.end_token())
    return data


def pad_sents(sents: List, pad_id: int):
    """
    按照最长的句子长度填充短句子

    Args:
        sents (List): 待填充的句子列表
        pad_id (int): 填充符号的ID

    Returns:
        sents_padded (List): 填充完成的句子
    """
    lens = [len(sent) for sent in sents]
    max_len = max(lens)
    sents_padded = []
    for i, sent in enumerate(sents):
        pad_sent = [pad_id] * max_len
        pad_sent[:lens[i]] = sent[:lens[i]]
        sents_padded.append(pad_sent)
    return sents_padded


def set_seed(seed):
    """
    固定随机种子

    Args:
        seed: 种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
