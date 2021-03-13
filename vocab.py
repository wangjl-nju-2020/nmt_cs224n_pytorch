import os
import torch
import utils
import pickle
import argparse

from typing import List, Any
from config import hparams


class Vocabulary:
    """
    词表类
    """

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.add_word(self.start_token())
        self.add_word(self.end_token())
        self.add_word(self.pad_token())
        self.add_word(self.unk_token())

    @staticmethod
    def start_token():
        return '<start>'

    @staticmethod
    def end_token():
        return '<end>'

    @staticmethod
    def pad_token():
        return '<pad>'

    @staticmethod
    def unk_token():
        return '<unk>'

    def add_word(self, word: Any):
        """
        添加单词进词表，并赋予唯一索引
        Args:
            word (Any): 待添加的单词
        """
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word: Any):
        """
        根据输入单词寻找索引，若不存在则返回<unk>对应的索引

        Args:
            word (Any): 待查询的单词

        Returns:
            (int): 单词的索引
        """
        if word not in self.word2idx:
            return self.word2idx[self.unk_token()]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def words2indices(self, sentences):
        """
        将单词序列转换为成索引列表

        Args:
            sentences: 待转换的单词序列

        Returns:
            索引序列
        """
        if type(sentences[0]) == list:
            return [[self.__call__(w) for w in s] for s in sentences]
        else:
            return [self.__call__(w) for w in sentences]

    def indices2words(self, word_ids):
        """
        将索引序列转换为成单词列表

        Args:
            word_ids: 待转换的索引序列

        Returns:
            单词序列
        """
        return [self.idx2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sentences: List[List[str]], device: torch.device) -> torch.Tensor:
        """
        将句子序列转换为指定设备上的tensor

        Args:
            sentences (List[List[str]]): 待转换的一批句子
            device (torch.device): 指定的设备

        Returns:
            (torch.Tensor): 返回tensor，大小为 (max_sen_len, batch_size)
        """
        word_ids = self.words2indices(sentences)
        sent_t = utils.pad_sents(word_ids, self.word2idx[self.pad_token()])
        sent_var = torch.tensor(sent_t, dtype=torch.long, device=device)
        # (max_sen_len, batch_size)
        return torch.t(sent_var)


def build_vocab(vocab_path, split_func=None, threshold=5):
    vocab = Vocabulary()
    for word in get_filtered_words(vocab_path, split_func, threshold):
        vocab.add_word(word)

    print('Total %d words in vocabulary.' % len(vocab))
    return vocab


def get_filtered_words(vocab_path: str, split_func: Any, threshold: int):
    """
    按照词表长最大值，出现次数最小值过滤词汇

    Args:
        vocab_path (str): 存储语料的地址
        split_func: 分割数据行的函数
        threshold: 单词最少出现的次数，达到该阈值才会被加入词表

    Returns:
        words (List[str]): 过滤后的单词列表
    """
    from collections import Counter
    sens = utils.read_corpus(path=vocab_path, split_func=split_func)
    counter = Counter()
    for idx, sen in enumerate(sens):
        counter.update(sen)
        if idx % 10000 == 0:
            print('[%d/%d] Tokenized the sentences.' % (idx, len(sens)))
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    return words


def dump_vocab(vocab_root, src_path, dst_path, threshold=5):
    """
    持久化词表

    Args:
        vocab_root: 存储词表的目录根路径
        src_path: 源语言语料的路径
        dst_path: 目标语言语料的路径
        threshold: 词频阈值，出现次数高于该次数的词才被收入进词表
    """
    print('*' * 20, 'dump vocab', '*' * 20)
    paths = [src_path, dst_path]
    split_funcs = [utils.split_and_pad, utils.split_by_tokenize]
    for idx, path in enumerate(paths):
        file_name = os.path.join(vocab_root, os.path.basename(path) + '.pkl')
        if not os.path.exists(file_name):
            vocab = build_vocab(path, split_funcs[idx], threshold)
            with open(file_name, 'wb') as f:
                pickle.dump(vocab, f)

            print('Total vocabulary size: %d' % len(vocab))
            print('Saved the vocabulary to: %s' % file_name)
        else:
            print('Vocabulary already exists.')


def load_vocab(src_pkl: str, dst_pkl: str):
    """
    读取源语言，目标语言词表

    Args:
        src_pkl (str): 源语言词表存储地址
        dst_pkl (str): 目标语言词表存储地址

    Returns:
        (Tuple[Vocabulary, Vocabulary]): 源语言词表，目标语言词表
    """
    print('*' * 20, 'load vocab', '*' * 20)
    with open(src_pkl, 'rb') as f1:
        src_vocab = VocabularyUnpickler(f1).load()
    with open(dst_pkl, 'rb') as f2:
        dst_vocab = VocabularyUnpickler(f2).load()
    return src_vocab, dst_vocab


class VocabularyUnpickler(pickle.Unpickler):
    """
    辅助类，解决pickle.load()反序列化找不到类定义的问题
    """

    def find_class(self, module, name):
        if name == 'Vocabulary':
            return Vocabulary
        return super().find_class(module, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vocab_root', default=hparams.vocab_root)
    parser.add_argument('--train_src_path', default=hparams.train_src_path)
    parser.add_argument('--train_dst_path', default=hparams.train_dst_path)
    parser.add_argument('--val_src_path', default=hparams.val_src_path)
    parser.add_argument('--val_dst_path', default=hparams.val_dst_path)
    parser.add_argument('--test_src_path', default=hparams.test_src_path)
    parser.add_argument('--test_dst_path', default=hparams.test_dst_path)
    parser.add_argument('--train_src_pkl', default=hparams.train_src_pkl)
    parser.add_argument('--train_dst_pkl', default=hparams.train_dst_pkl)
    args = parser.parse_args()

    src_paths = [args.train_src_path, args.val_src_path, args.test_src_path]
    dst_paths = [args.train_dst_path, args.val_dst_path, args.test_dst_path]
    for i in range(3):
        dump_vocab(args.vocab_root, src_paths[i], dst_paths[i], threshold=1)

    s_vocab, d_vocab = load_vocab(args.train_src_pkl, args.train_dst_pkl)
    for i in range(0, 10):
        print(s_vocab.idx2word[i], ', ')
        print(d_vocab.idx2word[i], '\n')
