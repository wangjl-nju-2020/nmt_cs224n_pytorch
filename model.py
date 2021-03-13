# @Author: wangjl
# @Email: If you need any help, feel free to contact me via wangjl.nju.2020@gmail.com.

from vocab import Vocabulary
from collections import namedtuple
from typing import List, Tuple, Any
from model_embeddings import ModelEmbeddings
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# beam search得到的假设，包括句子和对应的得分
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):
    """
    简单的NMT模型：编码器为双向LSTM， 解码器为单向LSTM，使用全局attention。
    （参考论文：'Effective Approaches to Attention-based Neural Machine Translation'）

    Args:
        embed_size (int): 单词的embeddings维度
        hidden_size (int): encoder隐藏层的维度
        src_vocab (Vocabulary): 平行语料源语言词汇表
        dst_vocab (Vocabulary): 平行语料目标语言词汇表
        dropout_rate (float): 用于正则化的丢弃率
        device (torch.device): 指定训练所用的GPU
    """

    def __init__(self, embed_size, hidden_size, src_vocab: Vocabulary, dst_vocab: Vocabulary, device, dropout_rate=0.2):
        super(NMT, self).__init__()
        self.device = device
        self.model_embeddings = ModelEmbeddings(embed_size, src_vocab, dst_vocab)
        self.hidden_size = hidden_size
        self.src_vocab = src_vocab
        self.dst_vocab = dst_vocab
        self.dropout_rate = dropout_rate
        # encoder是双向LSTM，有bias
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=hidden_size,
                               bidirectional=True,
                               dropout=dropout_rate,
                               bias=True)
        # decoder是单向LSTM，有bias
        self.decoder = nn.LSTMCell(input_size=embed_size + hidden_size,
                                   # input-feeding方法：将注意力向量和下一个时间步的输入连接在一起，使模型在做对齐决策时，也会考虑过去的对齐信息
                                   hidden_size=hidden_size,
                                   bias=True)
        # h_projection, c_projection分别是src对decoder状态和cell的初始化
        self.h_projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.c_projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        # att_projection是src对decoder隐空间的映射（到context vector）
        self.att_projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        # attention向量和下个时间步的输入连接在一起输入decoder
        self.combined_output_projection = nn.Linear(hidden_size * 2 + hidden_size, hidden_size, bias=False)
        # decoder神经网络的输入到vocab的映射
        self.target_vocab_projection = nn.Linear(hidden_size, len(dst_vocab), bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """
        计算一批平行语料中，输入源句子，使用NMT语言模型生成目标句子的对数概率

        Args:
            source (List[List(str)]): (batch_size, src_lens)
            target (List[List(str)]): (batch_size, tgt_lens)

        Returns:
            torch.Tensor: 生成目标句子的概率scores
        """
        source_lengths = [len(s) for s in source]
        # source_padded: (max_src_sen_len, batch_size)
        source_padded = self.src_vocab.to_input_tensor(source, self.device)
        # target_padded: (max_tgt_sen_len, batch_size)
        target_padded = self.dst_vocab.to_input_tensor(target, self.device)
        # enc_hiddens:
        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
        # 按照最后一维为轴，做log_softmax
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)
        # 广播操作：每一位是否是<pad>，<pad>对应为0，非<pad>对应为1
        target_masks = (target_padded != self.dst_vocab.word2idx[self.dst_vocab.pad_token()]).float()
        # 按照概率，指定维度，取矩阵中特定位置的数。计算生成target_gold_words的概率，其中去掉第一个<start>
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1),
                                                  dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores

    def encode(self, source_padded: torch.Tensor,
               source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        输入平行语料源句子，计算encoder得到隐藏状态，再由投影得到decoder的隐藏状态

        Args:
            source_padded (torch.Tensor): (max_sen_len, batch_size)
            source_lengths (List[int]): 一批源句子的实际长度

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: (enc_hiddens, dec_init_state)
        """
        # source_padded: (max_sen_len, batch_size)
        # enc_embedding: (max_sen_len, batch_size, embed_size)
        enc_embedding = self.model_embeddings.src_embedding(source_padded)
        # 执行压紧操作后的embedding
        packed_enc_embedding = pack_padded_sequence(enc_embedding, source_lengths)
        outputs, (last_hidden, last_cell) = self.encoder(packed_enc_embedding)
        outputs, _ = pad_packed_sequence(outputs)
        batch_size = len(source_lengths)
        src_len = source_lengths[0]
        # enc_hiddens: (batch_size, max_sen_len, hidden_size * 2)
        enc_hiddens = outputs.permute(1, 0, 2).contiguous().view(batch_size, src_len, -1)

        # last_hidden: (2, batch_size, hidden_size), last_cell: (2, batch_size, hidden_size)
        # init_decoder_hidden: (batch_size, hidden_size)
        init_decoder_hidden = self.h_projection(torch.cat([last_hidden[0], last_hidden[1]], 1))
        # init_decoder_cell: (batch_size, hidden_size)
        init_decoder_cell = self.c_projection(torch.cat([last_cell[0], last_cell[1]], 1))
        dec_init_state = (init_decoder_hidden, init_decoder_cell)
        return enc_hiddens, dec_init_state

    def decode(self, enc_hiddens: torch.Tensor, enc_masks: Any,
               dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:
        """
        使用encoder隐藏层状态，初始化decoder的隐藏层状态、Cell状态，得到联合输出

        Args:
            enc_hiddens (Tensor): (batch_size, max_src_sen_len, hidden_size * 2)
            enc_masks (Tensor): (batch_size, max_sen_len)
            dec_init_state (Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): (enc_hiddens, dec_init_state)
            target_padded (Tensor): (max_tgt_sen_len, batch_size)

        Returns:
            Tensor: 联合输出 (max_tgt_sen_len, batch_size, hidden_size)
        """
        # 去除<end>
        target_padded = target_padded[:-1]
        batch_size = enc_hiddens.size()[0]
        # enc_hiddens_proj: (batch_size, src_len, hidden_size)
        enc_hiddens_proj = self.att_projection(enc_hiddens)
        # dec_embedding: (max_target_sen_len, batch_size, embed_size)
        dec_embedding = self.model_embeddings.dst_embedding(target_padded)

        combined_outputs = []
        # 初始化上个时间步的输出为0向量
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)
        dec_state = dec_init_state
        for sen_embedding in torch.split(dec_embedding, split_size_or_sections=1):
            # sen_embedding: (1, batch_size, embed_size), embed_squeezed: (batch_size, embed_size)
            embed_squeezed = sen_embedding.squeeze(0)
            # combined_input: (batch_size, embed_size + hidden_size)
            combined_input = torch.cat([embed_squeezed, o_prev], 1)
            dec_state, combined_output, _ = self.step(combined_input, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            combined_outputs.append(combined_output)
            o_prev = combined_output

        # 长度为tgt_len的List[(batch_size, hidden_size)]转换为矩阵(tgt_len, batch_size, hidden_size)
        # torch,stack()的作用是增加维度扩展；与torch.cat()不改变维度，只改变shape值相区别
        combined_outputs = torch.stack(combined_outputs, dim=0)
        return combined_outputs

    def step(self, Ybar_t: torch.Tensor,
             dec_state: Tuple[torch.Tensor, torch.Tensor],
             enc_hiddens: torch.Tensor,
             enc_hiddens_proj: torch.Tensor,
             enc_masks: Any) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """
        计算decoder的单步输出

        Args:
            Ybar_t (torch.Tensor): 当前时间步的输入Y_t与上个时间步的输出o_perv的联合，大小为(batch_size, embed_size + hidden_size)
            dec_state (Tuple[torch.Tensor, torch.Tensor]): 当前时间步decoder的状态，包括hidden state、cell state，大小为(batch_size, hidden_size)
            enc_hiddens (torch.Tensor): encoder的hidden state，大小为(batch_size, max_sen_len, hidden_size * 2)
            enc_hiddens_proj (torch.Tensor): enc_hiddens的投影，大小为(batch_size, max_sen_len, hidden_size)
            enc_masks (Any): masks, 其中1表示<pad>，0表示对应位置存在有意义的词

        Returns:
            dec_state (Tuple[torch.Tensor, torch.Tensor]): 下一个时间步decoder的状态
            O_t (torch.Tensor): 大小为(batch_size, hidden_size)
            e_t (torch.Tensor): attention分数的分布，大小为(batch_size, max_sen_len)
        """
        dec_state = self.decoder(Ybar_t, dec_state)
        dec_hidden, dec_cell = dec_state
        # attention的权重分布：由encoder的隐藏层状态和decoder的隐藏层状态对齐计算得到(batch_size, max_sen_len)
        # (batch_size, max_sen_len, 1) = (batch_size, max_sen_len, hidden_size) * (batch_size, hidden_size, 1)
        e_t = torch.bmm(enc_hiddens_proj, dec_hidden.unsqueeze(2)).squeeze(2)

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks, -float('inf'))

        # 应用softmax的attention权重分布, (batch_size, max_sen_len)
        alpha_t = F.softmax(e_t, dim=1)
        # 计算attention层输入向量: 由alpha_t和enc_hiddens计算得到(batch_size, 2 * hidden_size)
        # (batch_size, 1, 2 * hidden_size) = (batch_size, 1, max_sen_len) * (batch_size, max_sen_len, 2 * hidden_size)
        a_t = torch.bmm(alpha_t.unsqueeze(1), enc_hiddens).squeeze(1)
        # 拼接a_t与dec_hidden计算U_t, (batch_size, 2 * hidden_size + hiddens_size)
        U_t = torch.cat([a_t, dec_hidden], 1)
        # V_t: (batch_size, hidden_size)
        V_t = self.combined_output_projection(U_t)
        O_t = self.dropout(torch.tanh(V_t))
        return dec_state, O_t, e_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """
        Generate sentence masks for encoder hidden states.

        Args:
            enc_hiddens (torch.Tensor): (batch_size, max_sen_len)
            source_lengths (List[int]): 句子的实际的长度

        Returns:
            enc_masks (torch.Tensor): (batch_size, max_sen_len)
        """
        batch_size = enc_hiddens.size()[0]
        max_sen_len = enc_hiddens.size()[1]
        enc_masks = torch.zeros(batch_size, max_sen_len, dtype=torch.bool)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = True
        return enc_masks.to(self.device)

    def beam_search(self, src_sent: List[str], beam_size: int = 5,
                    max_decoding_time_step: int = 70) -> List[Hypothesis]:
        """
        给定源句子，由beam search生成目标语言的翻译结果

        Args:
            src_sent (List[str]): 源句子
            beam_size (int): beam size
            max_decoding_time_step: 展开解码器RNN的最大步长

        Returns:
            hypotheses (List[Hypothesis]): 预测生成的句子。 每个Hypothesis包括生成的句子，以及得分，按照得分从大到小排列
            value (List[str]): 给定的目标句子
            score (float): 生成目标句子的概率
        """
        src_sents_var = self.src_vocab.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        hypotheses = [[Vocabulary.start_token()]]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size()[1],
                                                     src_encodings.size()[2])

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.dst_vocab(hyp[-1]) for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.model_embeddings.dst_embedding(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _ = self.step(x, h_tm1,
                                                exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.dst_vocab)
            hyp_word_ids = top_cand_hyp_pos % len(self.dst_vocab)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.dst_vocab.idx2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == Vocabulary.end_token():
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @staticmethod
    def load(model_path: str, device: torch.device):
        """
        加载模型

        Args:
            model_path (str): 模型保存的路径
            device:
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(src_vocab=params['src_vocab'], dst_vocab=params['dst_vocab'], device=device, **args).to(device)
        model.load_state_dict(params['state_dict'])
        return model

    def save(self, path: str):
        """
        保存模型

        Args:
            path (str): 保存逻辑的路径
        """
        print('save model parameters to {}'.format(path))
        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size,
                         dropout_rate=self.dropout_rate),
            'src_vocab': self.src_vocab,
            'dst_vocab': self.dst_vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)
