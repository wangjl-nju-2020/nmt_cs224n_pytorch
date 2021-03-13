# @Author: wangjl
# @Email: If you need any help, feel free to contact me via wangjl.nju.2020@gmail.com.

from tqdm import tqdm
from model import NMT, Hypothesis
from typing import List, Tuple
from vocab import load_vocab, Vocabulary
from data_loader import get_test_dataloader
from nltk.translate.bleu_score import corpus_bleu

import torch


class Tester:
    """
    测试类，使用测试集验证模型

    Args:
        _hparams (NameSpace): 人为设定的超参数，默认值见config.py，也可以在命令行指定。
    """

    def __init__(self, _hparams):
        self.hparams = _hparams
        self.src_vocab, self.dst_vocab = load_vocab(_hparams.train_src_pkl,
                                                    _hparams.train_dst_pkl)
        self.device = torch.device(_hparams.device)

    def test(self):
        print('*' * 20, 'start test', '*' * 20)
        self.model = NMT.load(self.hparams.model_save_path, self.device)
        sources, references, hypotheses = self.beam_search()
        bleu_score = compute_corpus_level_bleu_score(references, hypotheses)
        print('Corpus BLEU: {}'.format(bleu_score * 100))

        with open(self.hparams.test_res_path, 'w') as f:
            for src_sent, hypo in zip(sources, hypotheses):
                src_sent = ' '.join(src_sent)
                hypo_sent = ' '.join(hypo.value)
                f.write(src_sent + '\n' + hypo_sent + '\n\n')
        print('save test result to {}'.format(self.hparams.test_res_path))
        print('*' * 20, 'end test', '*' * 20)

    def beam_search(self) -> Tuple[List[List[str]], List[List[str]], List[Hypothesis]]:
        """
        测试由beam search生成假设的句子集合

        Returns:
            sources, references, hypotheses (Tuple[List[List[str]], List[List[str]], List[Hypothesis]]): 返回原始句子、参考句子和生成的假设
        """
        self.model.eval()
        sources = []
        hypotheses = []
        references = []
        with torch.no_grad():
            for step, data_pairs in tqdm(
                    enumerate(get_test_dataloader(self.hparams.test_src_path, self.hparams.test_dst_path))):
                sents = [(dp.src, dp.dst) for dp in data_pairs]
                src_sents, tgt_sents = zip(*sents)
                sources.append(src_sents[0])
                references.append(tgt_sents[0])
                hypos = self.model.beam_search(src_sents[0],
                                               beam_size=self.hparams.beam_size,
                                               max_decoding_time_step=self.hparams.max_decoding_time_step)
                hypotheses.append(hypos[0])
        return sources, references, hypotheses


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    根据推理结果和参考句子，计算语料库级别的BLEU

    Args:
        references (List[List[str]]): 参考句子
        hypotheses (List[Hypothesis]): 生成的假设

    Returns:
        bleu_score (float): BLEU得分
    """
    if references[0][0] == Vocabulary.start_token():
        references = [ref[1:-1] for ref in references]
    # TODO：似乎计算不准确，BLEU得分偏低？
    bleu_score = corpus_bleu([[ref] for ref in references], [hyp.value for hyp in hypotheses])
    return bleu_score
