# @Author: wangjl
# @Email: If you need any help, feel free to contact me via wangjl.nju.2020@gmail.com.

from typing import List
from torch.utils.data import DataLoader
from dataset import EsEnDataset


def collate_fn(data: List[EsEnDataset]):
    """
    按降序排列，pack_padded_sequence要求按照序列按照长度从大到小排列

    Args:
        data (List[EsEnDataset]): 需要排序的数据

    Returns:
        data (List[EsEnDataset]): 排序好的数据
    """
    data.sort(key=lambda x: len(x.src), reverse=True)
    return data


def get_dataloader(es_path: str, en_path: str, batch_size: int, num_workers: int = 5):
    """
    训练阶段的DataLoader

    Args:
        es_path (str): 原语言的语料存储地址
        en_path (str): 目标语言的语料存储地址
        batch_size (int): 每批数据加载条数
        num_workers (int): 加载数据使用的子进程数

    Returns:
        (Dataloader): 数据加载器
    """
    return DataLoader(dataset=EsEnDataset(es_path, en_path),
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers,
                      collate_fn=collate_fn)


def get_test_dataloader(test_src_path: str, test_dst_path: str):
    """
    测试阶段的DataLoader

    Args:
        test_src_path (str): 原语言的语料存储地址
        test_dst_path (str): 目标语言的语料存储地址

    Returns:
        (Dataloader): 数据加载器
    """
    return DataLoader(dataset=EsEnDataset(test_src_path, test_dst_path),
                      batch_size=1,
                      shuffle=False,
                      num_workers=1,
                      collate_fn=collate_fn)


def get_val_dataloader(val_src_path: str, val_dst_path: str, batch_size: int, num_workers: int = 5):
    """
    训练阶段的DataLoader

    Args:
        val_src_path (str): 原语言的语料存储地址
        val_dst_path (str): 目标语言的语料存储地址
        batch_size (int): 每批数据加载条数
        num_workers (int): 加载数据使用的子进程数

    Returns:
        (Dataloader): 数据加载器
    """
    return DataLoader(dataset=EsEnDataset(val_src_path, val_dst_path),
                      batch_size=batch_size,
                      shuffle=False,
                      num_workers=num_workers,
                      collate_fn=collate_fn)
