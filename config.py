# -*- encoding: utf-8 -*-
"""

"""
# from model import ConvLstm
import torch
import numpy as np

class Config():
    def __init__(self, seq_in, seq_out):
        # (type, activation, in_ch, out_ch, kernel_size, padding, stride)
        self.sequences_in = seq_in
        self.sequences_out = seq_out
        self.sequences_length = self.sequences_in + self.sequences_out

        self.src_len = seq_in  # length of source
        self.tgt_len = seq_out  # length of target
        self.d_model = 12  # Embedding Size
        self.d_ff = 512  # 256  # 2048  # FeedForward dimension
        self.d_k = 256  # 64  # dimension of K(=Q), V;  = d_v
        self.d_v = 256  # 64
        self.n_layers = 6  # 6 # 12  # 8 # 4  # 2 # number of Encoder of Decoder Layer
        self.n_heads = 8  # 2 # 12  # 4  # number of heads in Multi-Head Attention
        self.hidden_size = 900 # 540  # 256  # 1440  # 64  # LSTM 隐藏状态的特征维度


        self.percent = 0.8  # 0.6

        self.batch = 20 # 1 # 1  # 20  # 5  # 20  # 3  # 200  # 16  # 16  # 500  # 2  #

        self.input_size = 12  # 输入数据的特征维度
        # self.hidden_size = 1440 #  768 # 128  # 64  # LSTM 隐藏状态的特征维度
        self.num_layers = 12  # 12  # LSTM 层数
        self.output_size = 12  # 输出数据的特征维度
        self.drop_rate = 0.1

        self.num_workers = 0  # 多线程/ windows必须设置为0
        self.num_epochs = 400  # 20  # 3  # 40  # 训练次数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # prepare for earlystopping
        self.patience = 15  # 3  # 7



config = Config(seq_in=6, seq_out=1)
# print(config.encoder)
# print(config.decoder)
# print(config.sequences)
# print(config.time_start, config.time_end)
print(config.batch, config.sequences_in, config.sequences_out, config.percent)
