import torch
import torch.nn as nn
import numpy as np
from networks.layers import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        self.register_buffer('positional_encoding', self.encoding)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        return self.encoding[:seq_len, :].clone().detach().to(x.device) + x

class EncoderV2(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, n_head, d_k, d_v, d_inner, dropout=0.1, n_position=40):
        super(EncoderV2, self).__init__()
        self.position_enc = PositionalEncoding(d_model, max_len=n_position)
        self.src_projection = nn.Linear(input_dim, d_model)  # 输入投影层
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_mask):
        src_seq = self.src_projection(src_seq)  # 投影输入到 d_model
        src_seq = self.position_enc(src_seq)
        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(enc_output, slf_attn_mask=src_mask)

        return enc_output


class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, n_layers, n_head, d_k, d_v, d_inner, dropout=0.1, n_position=40):
        super(Decoder, self).__init__()
        self.position_enc = PositionalEncoding(d_model, max_len=n_position)
        self.trg_projection = nn.Linear(output_dim, d_model)  # 输入投影层
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.output_projection = nn.Linear(d_model, output_dim)  # 输出投影层

    def forward(self, trg_seq, trg_mask, enc_output, src_mask):
        trg_seq = self.trg_projection(trg_seq)  # 投影输入到 d_model
        trg_seq = self.position_enc(trg_seq)
        dec_output = trg_seq

        for dec_layer in self.layer_stack:
            dec_output, _, _ = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)

        dec_output = self.output_projection(dec_output)  # 投影到目标维度
        return dec_output


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_seq, src_mask, trg_seq, trg_mask):
        enc_output = self.encoder(src_seq, src_mask)
        dec_output = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        return dec_output