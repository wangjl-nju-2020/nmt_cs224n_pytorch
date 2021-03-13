# @Author: wangjl
# @Email: If you need any help, feel free to contact me via wangjl.nju.2020@gmail.com.

from vocab import Vocabulary

import torch.nn as nn


class ModelEmbeddings(nn.Module):
    """
    源语言和目标语言的embedding层

    Args:
        embed_size (int): 词embedding的维度
        src_vocab (Vocabulary): 源语言词表
        dst_vocab (Vocabulary): 目标语言词表
    """
    def __init__(self, embed_size: int, src_vocab: Vocabulary, dst_vocab: Vocabulary):
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size

        # 默认值
        src_padding_idx = src_vocab.word2idx[Vocabulary.pad_token()]
        dst_padding_idx = dst_vocab.word2idx[Vocabulary.pad_token()]
        self.src_embedding = nn.Embedding(len(src_vocab), embed_size, padding_idx=src_padding_idx)
        self.dst_embedding = nn.Embedding(len(dst_vocab), embed_size, padding_idx=dst_padding_idx)
