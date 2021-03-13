# @Author: wangjl
# @Email: If you need any help, feel free to contact me via wangjl.nju.2020@gmail.com.

from typing import Any
from torch.utils.data import Dataset

import utils


class DataPair:
    """
    源语言-目标语言平行语料数据对

    Args:
        src (Any): 源语言数据
        dst (Any): 目标语言数据
    """

    def __init__(self, src: Any, dst: Any):
        self.src = src
        self.dst = dst


class EsEnDataset(Dataset):
    """
    自定义Dataset，用于读取英-西双语语料

    Args:
        es_path (str): 原语言语料存储地址
        en_path (str): 目标语言语料存储地址
    """

    def __init__(self, es_path, en_path):
        src_data = utils.read_corpus(es_path, utils.split_and_pad)
        dst_data = utils.read_corpus(en_path, utils.split_by_tokenize)
        self.data = []
        for i in range(len(src_data)):
            self.data.append(DataPair(src_data[i], dst_data[i]))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
