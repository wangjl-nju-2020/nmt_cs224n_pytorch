# @Author: wangjl
# @Email: If you need any help, feel free to contact me via wangjl.nju.2020@gmail.com.

from tqdm import tqdm
from model import NMT
from utils import set_seed
from vocab import load_vocab
from data_loader import get_dataloader, get_val_dataloader

import torch
import numpy as np


def evaluate_ppl(model, val_src_path, val_dst_path, batch_size, num_workers):
    """
    批量计算模型在源句子和目标句子上的困惑度

    Args:
        model (NMT): 模型
        batch_size (int): 成批计算的大小
        val_src_path (str): 验证集源句子文件地址
        val_dst_path (str): 验证集目标句子文件地址
        num_workers (int): 读取数据使用线程数

    Returns:
        模型在源句子和目标句子上的困惑度
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    with torch.no_grad():
        for data_pairs in get_val_dataloader(val_src_path, val_dst_path, batch_size, num_workers):
            sents = [(dp.src, dp.dst) for dp in data_pairs]
            src_sents, tgt_sents = zip(*sents)
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


class Trainer:
    """
    训练类，使用训练集训练模型

    Args:
        _hparams (NameSpace): 人为设定的超参数，默认值见config.py，也可以在命令行指定。
    """

    def __init__(self, _hparams):
        self.hparams = _hparams
        set_seed(_hparams.fixed_seed)
        self.train_loader = get_dataloader(_hparams.train_src_path, _hparams.train_dst_path,
                                           _hparams.batch_size, _hparams.num_workers)
        self.src_vocab, self.dst_vocab = load_vocab(_hparams.train_src_pkl, _hparams.train_dst_pkl)
        self.device = torch.device(_hparams.device)
        self.model = NMT(_hparams.embed_size, _hparams.hidden_size,
                         self.src_vocab, self.dst_vocab, self.device,
                         _hparams.dropout_rate).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=_hparams.lr)

    def train(self):
        print('*' * 20, 'train', '*' * 20)
        hist_valid_scores = []
        patience = 0
        num_trial = 0

        for epoch in range(int(self.hparams.max_epochs)):
            self.model.train()

            epoch_loss_val = 0
            epoch_steps = len(self.train_loader)
            for step, data_pairs in tqdm(enumerate(self.train_loader)):
                sents = [(dp.src, dp.dst) for dp in data_pairs]
                src_sents, tgt_sents = zip(*sents)

                self.optimizer.zero_grad()

                batch_size = len(src_sents)
                example_losses = -self.model(src_sents, tgt_sents)
                batch_loss = example_losses.sum()
                train_loss = batch_loss / batch_size
                epoch_loss_val += train_loss.item()
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.clip_gradient)
                self.optimizer.step()

            epoch_loss_val /= epoch_steps
            print('epoch: {}, epoch_loss_val: {}'.format(epoch, epoch_loss_val))

            # perform validation
            if epoch % self.hparams.valid_niter == 0:
                print('*' * 20, 'validate', '*' * 20)
                dev_ppl = evaluate_ppl(self.model, self.hparams.val_src_path, self.hparams.val_dst_path,
                                       self.hparams.batch_val_size, self.hparams.num_workers)
                valid_metric = -dev_ppl

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to {}'.format(self.hparams.model_save_path))
                    self.model.save(self.hparams.model_save_path)
                    torch.save(self.optimizer.state_dict(), self.hparams.optimizer_save_path)
                elif patience < self.hparams.patience:
                    patience += 1
                    print('hit patience %d' % patience)

                    if patience == self.hparams.patience:
                        num_trial += 1
                        print('hit #{} trial'.format(num_trial))
                        if num_trial == self.hparams.max_num_trial:
                            print('early stop!')
                            exit(0)

                        # 兼容设计，考虑Adam不需要人工调整lr，而其他优化器需要
                        if hasattr(self.optimizer, 'param_group'):
                            # decay lr, and restore from previously best checkpoint
                            lr = self.optimizer.param_groups[0]['lr'] * self.hparams.lr_decay
                            print('load previously best model and decay learning rate to %f' % lr)

                            params = torch.load(self.hparams.model_save_path, map_location=lambda storage, loc: storage)
                            self.model.load_state_dict(params['state_dict'])
                            self.model = self.model.to(self.device)

                            print('restore parameters of the optimizers')
                            self.optimizer.load_state_dict(torch.load(self.hparams.optimizer_save_path))

                            # set new lr
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = lr

                        # reset patience
                        patience = 0
                print('*' * 20, 'end validate', '*' * 20)
        print('*' * 20, 'end train', '*' * 20)
